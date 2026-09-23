# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Worker server for the TensorRT runtime subprocess."""
import argparse
import ctypes
import gc
import logging
import sys
import traceback
from multiprocessing.connection import Listener
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# noinspection PyPackageRequirements
import numpy as np  # ships in conecsa-os-base:base

from .tensor_shm import TensorBuffer, layout_for, size_fd


def _malloc_trim() -> None:
    """
    Force glibc to return free heap pages to the kernel.

    Without this, the worker's RSS climbs monotonically across model
    swaps and engine rebuilds because glibc keeps freed chunks pooled
    in arenas (especially severe on the Yocto host which has no
    zram/swap to absorb the fragmentation).
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        # Non-glibc libc (e.g. musl) - silently skip.
        pass


AUTH_KEY = b"conecsa"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def _send_response(conn, status: str, **kwargs) -> None:
    """
    Send response to client.

    Args:
        conn: Connection object
        status: Response status ("ok" or "error")
        **kwargs: Additional response fields
    """
    response = {"status": status, **kwargs}
    try:
        conn.send(response)
    except Exception as e:
        logger.error(f"Failed to send response: {e}")


def _send_error(conn, error_message: str) -> None:
    """
    Send error response to client.

    Args:
        conn: Connection object
        error_message: Error message
    """
    _send_response(conn, "error", error=error_message)


def _send_success(conn, **kwargs) -> None:
    """
    Send success response to client.

    Args:
        conn: Connection object
        **kwargs: Additional response fields
    """
    _send_response(conn, "ok", **kwargs)


def _create_runtime():
    """Create the TensorRT runtime instance."""
    logger.info("Creating TensorRT runtime")
    try:
        from .tensorrt_runtime import TensorRTRuntime
        return TensorRTRuntime()
    except Exception as e:
        logger.error(f"Failed to create TensorRT runtime: {e}")
        logger.error(traceback.format_exc())
        raise


def _serialize_outputs(interpreter) -> List[Dict[str, Any]]:
    """
    Serialize output tensors from interpreter.

    Uses the tensor index stored in each output_detail entry. Some model graphs
    use raw tensor indices rather than sequential 0,1,2...

    Args:
        interpreter: Interpreter instance

    Returns:
        List of serialized output dictionaries
    """
    outputs = []
    output_details = interpreter.get_output_details()
    for detail in output_details:
        tensor_index = detail['index']
        out = interpreter.get_tensor(tensor_index)
        outputs.append({
            "dtype": str(out.dtype),
            "shape": list(out.shape),
            "data": out.tobytes(),
        })
    return outputs


def _deserialize_input(input_info: Dict[str, Any]) -> np.ndarray:
    """
    Deserialize input data from client.

    Args:
        input_info: Input info dictionary with dtype, shape, and data

    Returns:
        Numpy array
    """
    dtype = np.dtype(input_info["dtype"])
    data = np.frombuffer(input_info["data"], dtype=dtype)
    return data.reshape(input_info["shape"])


def _unmap_buffer(shm) -> None:
    """Drop the worker's view of the shared tensor buffer, if any."""
    if shm is not None and shm.buffer is not None:
        shm.buffer.close()
        shm.buffer = None


def _map_buffer(shm, interpreter) -> Optional[Dict[str, Any]]:
    """Lay the loaded engine out in the shared buffer; returns the layout for the client.

    ``None`` (tensors keep going over the connection) without a shared file,
    for an engine with a dynamic shape, or when the file cannot be sized or
    mapped. The file only ever grows: the client may still map a previous,
    larger layout until it reads this answer, and shrinking under it would
    fault on access.
    """
    _unmap_buffer(shm)
    if shm is None or shm.fd is None:
        return None
    layout = layout_for(interpreter.get_input_details(), interpreter.get_output_details())
    if layout is None:
        logger.info("Engine has a dynamic tensor shape; tensors go over the connection")
        return None
    try:
        size_fd(shm.fd, layout["size"])
        shm.buffer = TensorBuffer(shm.fd, layout)
    except (OSError, ValueError) as exc:
        logger.warning(f"Shared tensor buffer unavailable ({exc}); tensors go over the connection")
        shm.buffer = None
        return None
    return layout


def _handle_load_command(
    conn,
    msg: Dict[str, Any],
    prev_runtime: Optional[Any],
    prev_interpreter: Optional[Any],
    shm: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Handle load model command.

    Tears down any previously-loaded runtime/interpreter explicitly before
    creating new ones. Without explicit teardown, TRT engine <-> context <->
    pycuda bindings form reference cycles that defeat CPython refcount-based
    GC, so each model swap would leak the previous engine's device memory.

    With a shared tensor buffer (``shm``), the answer also carries the new
    engine's ``shm_layout`` (``None`` when the buffer cannot serve it).

    Args:
        conn: Connection object
        msg: Message dictionary
        prev_runtime: Previously-loaded runtime (or None on first load)
        prev_interpreter: Previously-loaded interpreter (or None on first load)
        shm: The worker's shared-buffer state (``fd``, ``buffer``), if any

    Returns:
        Tuple of (runtime, interpreter)
    """
    model_path = msg.get("model_path")
    if not model_path:
        _send_error(conn, "model_path missing")
        return prev_runtime, prev_interpreter

    # Explicit teardown of the prior model before allocating the new one,
    # so peak memory does not double during the swap.
    _unmap_buffer(shm)
    if prev_interpreter is not None:
        try:
            close = getattr(prev_interpreter, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.exception("Error closing previous interpreter (continuing)")
    prev_interpreter = None
    prev_runtime = None
    gc.collect()
    _malloc_trim()

    try:
        logger.info(f"Loading model: {model_path}")
        runtime = _create_runtime()
        interpreter = runtime.create_interpreter(model_path)
        logger.info(f"Model loaded successfully: {model_path}")

        _send_success(
            conn,
            input_details=interpreter.get_input_details(),
            output_details=interpreter.get_output_details(),
            shm_layout=_map_buffer(shm, interpreter),
        )
        # Trim again after a successful load — the TRT deserialize step
        # allocates and frees substantial transient buffers.
        _malloc_trim()
        return runtime, interpreter
    except Exception as exc:
        logger.error(f"Failed to load model {model_path}: {exc}")
        logger.error(traceback.format_exc())
        _send_error(conn, str(exc))
        _malloc_trim()
        return None, None


def _handle_infer_command(conn, msg: Dict[str, Any], interpreter) -> None:
    """
    Handle inference command.

    Args:
        conn: Connection object
        msg: Message dictionary
        interpreter: Interpreter instance
    """
    if interpreter is None:
        _send_error(conn, "model not loaded")
        return

    input_info = msg.get("input")
    if input_info is None:
        _send_error(conn, "input missing")
        return

    try:
        data = _deserialize_input(input_info)
        interpreter.set_tensor(0, data)
        interpreter.invoke()
        _send_success(conn, outputs=_serialize_outputs(interpreter))
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        logger.error(traceback.format_exc())
        _send_error(conn, str(exc))


def _handle_infer_shm_command(conn, interpreter, shm) -> None:
    """
    Handle an inference whose tensors live in the shared buffer.

    The client has written the input slot; the outputs are copied into
    their slots and the answer is ``shm=True``. Should an output not match
    its slot, the outputs travel in the answer instead (``shm=False``).
    """
    if interpreter is None:
        _send_error(conn, "model not loaded")
        return
    buffer = shm.buffer if shm is not None else None
    if buffer is None:
        _send_error(conn, "shared tensor buffer not mapped")
        return

    try:
        interpreter.set_tensor(0, buffer.input)
        interpreter.invoke()
        outputs = [interpreter.get_tensor(detail['index'])
                   for detail in interpreter.get_output_details()]
        slots = buffer.outputs
        if len(outputs) == len(slots) and all(
                out.shape == slot.shape and out.dtype == slot.dtype
                for out, slot in zip(outputs, slots, strict=True)):
            for out, slot in zip(outputs, slots, strict=True):
                np.copyto(slot, out)
            _send_success(conn, shm=True)
        else:
            _send_success(conn, shm=False, outputs=_serialize_outputs(interpreter))
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        logger.error(traceback.format_exc())
        _send_error(conn, str(exc))


def _handle_build_engine_command(conn, msg: Dict[str, Any], interpreter) -> None:
    """
    Build a TensorRT .engine from an .onnx file.

    Fast path: if a TensorRTInterpreter is already loaded in this worker, its
    active CUDA context is reused and ``build_engine_from_onnx`` is called
    directly.

    Cold-start path: if no interpreter is loaded yet, the standalone
    ``build_engine_from_onnx`` function is used.  It initializes CUDA via
    ``pycuda.autoinit`` without requiring any pre-existing .engine file.
    """
    import os as _os

    onnx_path = msg.get("onnx_path")
    engine_path = msg.get("engine_path")
    workspace_mb = int(msg.get("workspace_mb", 256))

    if not onnx_path or not engine_path:
        _send_error(conn, "onnx_path and engine_path are required")
        return

    try:
        _os.environ["TENSORRT_WORKSPACE_MB"] = str(workspace_mb)
        logger.info(f"Building TensorRT engine: {onnx_path} → {engine_path}")

        try:
            from .tensorrt_interpreter import (  # type: ignore
                TensorRTInterpreter,
                build_engine_from_onnx,
            )
        except ImportError as import_err:
            logger.error(f"Failed to import TensorRT components: {import_err}")
            _send_error(conn, f"TensorRT not available: {import_err}")
            return

        if isinstance(interpreter, TensorRTInterpreter):
            # Fast path: reuse active CUDA context from loaded interpreter
            logger.info("Using fast path with existing TensorRT interpreter")
            interpreter.build_engine_from_onnx(
                onnx_path=onnx_path,
                engine_cache_path=engine_path,
            )
        else:
            # Cold-start path: initialize CUDA via pycuda.autoinit
            logger.info("No interpreter loaded — using standalone builder (pycuda.autoinit)")
            build_engine_from_onnx(onnx_path, engine_path, workspace_mb)

        logger.info(f"Engine saved: {engine_path}")
        _send_success(conn, engine_path=engine_path)

    except Exception as exc:
        logger.error(f"build_engine failed: {exc}")
        logger.error(traceback.format_exc())
        _send_error(conn, str(exc))
    finally:
        # The TRT builder allocates large transient buffers (workspace,
        # ONNX parser graph, serialized engine). Return the heap pages to
        # the kernel so peak RSS does not stick around.
        gc.collect()
        _malloc_trim()


def run_server(host: str, port: int, shm_fd: Optional[int] = None) -> None:
    """
    Run the TensorRT worker server.

    Args:
        host: Host address to bind to
        port: Port to bind to
        shm_fd: Inherited shared-memory file for the tensors, if the client made one
    """
    logger.info(f"Starting TensorRT worker server on {host}:{port}")

    try:
        listener = Listener((host, port), authkey=AUTH_KEY)
        logger.info(f"Server listening on {host}:{port}")
    except Exception as e:
        logger.error(f"Failed to create listener on {host}:{port}: {e}")
        logger.error(traceback.format_exc())
        raise

    interpreter = None
    runtime = None
    shm = SimpleNamespace(fd=shm_fd, buffer=None)

    try:
        while True:
            logger.info("Waiting for client connection...")
            conn = listener.accept()
            logger.info("Client connected")

            try:
                while True:
                    try:
                        msg = conn.recv()
                        cmd = msg.get("cmd")
                        logger.debug(f"Received command: {cmd}")

                        if cmd == "load":
                            runtime, interpreter = _handle_load_command(
                                conn, msg, runtime, interpreter, shm
                            )

                        elif cmd == "infer_shm":
                            _handle_infer_shm_command(conn, interpreter, shm)

                        elif cmd == "infer":
                            _handle_infer_command(conn, msg, interpreter)

                        elif cmd == "build_engine":
                            _handle_build_engine_command(conn, msg, interpreter)

                        elif cmd == "close":
                            logger.info("Received close command, shutting down")
                            _send_success(conn)
                            conn.close()
                            return

                        else:
                            logger.warning(f"Unknown command: {cmd}")
                            _send_error(conn, "unknown command")

                    except Exception as e:
                        logger.error(f"Error processing command: {e}")
                        logger.error(traceback.format_exc())
                        try:
                            _send_error(conn, str(e))
                        except Exception as send_err:
                            logger.error(f"Failed to send error response: {send_err}")
                            break

            except EOFError:
                logger.info("Client disconnected")
                conn.close()
            except Exception as e:
                logger.error(f"Connection error: {e}")
                logger.error(traceback.format_exc())
                try:
                    conn.close()
                except Exception as close_err:
                    logger.error(f"Failed to close connection: {close_err}")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Server shutting down")
        _unmap_buffer(shm)
        try:
            listener.close()
        except Exception as close_err:
            logger.error(f"Failed to close listener: {close_err}")


def main() -> None:
    """Main entry point for the worker server."""
    parser = argparse.ArgumentParser(description="TensorRT worker server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--shm-fd", type=int, default=-1,
                        help="Inherited shared-memory file descriptor for the tensors")
    args = parser.parse_args()

    try:
        run_server(args.host, args.port, args.shm_fd if args.shm_fd >= 0 else None)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
