//! POSIX shared-memory producer: a double-buffered frame region plus
//! protobuf-encoded config/health regions in the header (the camera SHM ring
//! consumed by the inference-service and api-gateway).
//!
//! # Publication protocol (header version 2) — must match
//! `os-base/conecsa_shm/camera_ring.py`
//!
//! `FRAME_WRITE_SEQ` is an odd/even generation counter (a seqlock):
//!
//! 1. writer: `seq += 1` (→ odd, "publication open"), release-ordered;
//! 2. writer: payload into the inactive slot, then ALL metadata
//!    (`WIDTH`/`HEIGHT` on change, `FORMAT_FLAG`, `FRAME_SIZE`,
//!    `ACTIVE_SLOT`) — everything a reader pairs with this frame;
//! 3. writer: `seq += 1` (→ even, "published"), release-ordered.
//!
//! Readers load the seq (acquire); an odd value means a write is in flight.
//! On even, they copy the frame and metadata, re-load the seq, and accept
//! only when it is unchanged — otherwise the copy may be torn (the writer
//! can lap a slow reader with only two slots) and they retry. The Python
//! side has no fences; its contract is aligned 8-byte accesses plus that
//! double-read validation.

use prost::Message;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

/// Magic number identifying a valid conecsa SHM segment.
const SHM_MAGIC: u32 = 0xC04E_5A01;
/// Version 2 = seqlock publication protocol (see the module docs). Readers
/// reject a mismatched version, so mixed old/new services fail loudly
/// instead of tearing frames.
const SHM_VERSION: u32 = 2;
const HEADER_SIZE: usize = 256;
const CONFIG_PAYLOAD_MAX: usize = 128;
const HEALTH_PAYLOAD_MAX: usize = 64;
const SHM_SLOT_MIN_BYTES_DEFAULT: usize = 8 * 1024 * 1024;

/// Offsets into the SHM header (see plan for full layout).
mod off {
    pub const MAGIC: usize = 0;
    pub const VERSION: usize = 4;
    pub const WIDTH: usize = 8;
    pub const HEIGHT: usize = 12;
    pub const CHANNELS: usize = 16;
    pub const MAX_FRAME_BYTES: usize = 20;
    pub const FRAME_WRITE_SEQ: usize = 24;
    pub const ACTIVE_SLOT: usize = 32;
    pub const FORMAT_FLAG: usize = 36;
    pub const FRAME_SIZE: usize = 40;
    // Config region
    pub const CONFIG_WRITE_SEQ: usize = 44;
    pub const CONFIG_SIZE: usize = 48;
    pub const CONFIG_PAYLOAD: usize = 52;
    // Health region (52 + 128 = 180)
    pub const HEALTH_WRITE_SEQ: usize = 180;
    pub const HEALTH_SIZE: usize = 184;
    pub const HEALTH_PAYLOAD: usize = 188;
}

/// Include prost-generated protobuf types.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/conecsa.rs"));
}

/// Format flags written to the SHM header.
pub const FORMAT_RAW_RGB: u32 = 0;
pub const FORMAT_JPEG: u32 = 1;

/// POSIX shared-memory producer.
///
/// Creates a named SHM segment with a double-buffered frame region and
/// protobuf-encoded config/health regions in the header.
pub struct ShmProducer {
    ptr: *mut u8,
    total_size: usize,
    slot_size: usize,
    shm_name: std::ffi::CString,
    last_config_seq: AtomicU32,
}

// SAFETY: `ptr` is a private mmap'd region that outlives the struct (unmapped
// only in Drop); all access to it goes through raw-pointer/atomic operations
// that never materialize a `&mut`, and the only local mutable state
// (`last_config_seq`) is atomic. Cross-process synchronization is handled by
// the release/acquire atomics in the mapped header.
unsafe impl Send for ShmProducer {}
unsafe impl Sync for ShmProducer {}

impl ShmProducer {
    /// Read slot min bytes.
    fn read_slot_min_bytes() -> usize {
        std::env::var("SHM_SLOT_MIN_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v >= 1_500_000)
            .unwrap_or(SHM_SLOT_MIN_BYTES_DEFAULT)
    }

    /// Create (or re-create) the named shared-memory segment.
    pub fn new(name: &str, width: u32, height: u32) -> Result<Self, String> {
        // Compute frame size in bytes using usize and checked arithmetic to avoid overflow.
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| "frame dimensions too large".to_string())?;
        let frame_bytes = pixels
            .checked_mul(3)
            .ok_or_else(|| "frame size too large".to_string())?;
        let slot_size =
            std::cmp::max(frame_bytes, Self::read_slot_min_bytes());
        let total_size = slot_size
            .checked_mul(2)
            .and_then(|v| HEADER_SIZE.checked_add(v))
            .ok_or_else(|| "shared memory segment size too large".to_string())?;

        let shm_name = std::ffi::CString::new(format!("/{name}"))
            .map_err(|e| format!("invalid shm name: {e}"))?;

        unsafe {
            // Remove stale segment if any.
            libc::shm_unlink(shm_name.as_ptr());

            let fd = libc::shm_open(
                shm_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_EXCL,
                0o666,
            );
            if fd < 0 {
                return Err(format!(
                    "shm_open failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            if libc::ftruncate(fd, total_size as libc::off_t) != 0 {
                libc::close(fd);
                libc::shm_unlink(shm_name.as_ptr());
                return Err(format!(
                    "ftruncate failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            let ptr = libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);

            if ptr == libc::MAP_FAILED {
                libc::shm_unlink(shm_name.as_ptr());
                return Err(format!("mmap failed: {}", std::io::Error::last_os_error()));
            }

            let ptr = ptr as *mut u8;

            // Zero the entire region.
            std::ptr::write_bytes(ptr, 0, total_size);

            // Write header fields.
            Self::write_u32(ptr, off::MAGIC, SHM_MAGIC);
            Self::write_u32(ptr, off::VERSION, SHM_VERSION);
            Self::write_u32(ptr, off::WIDTH, width);
            Self::write_u32(ptr, off::HEIGHT, height);
            Self::write_u32(ptr, off::CHANNELS, 3);
            Self::write_u32(ptr, off::MAX_FRAME_BYTES, slot_size as u32);

            Ok(Self {
                ptr,
                total_size,
                slot_size,
                shm_name,
                last_config_seq: AtomicU32::new(0),
            })
        }
    }

    /// Publish a raw RGB frame (format_flag = 0).
    pub fn publish_frame_rgb(&self, data: &[u8], w: u32, h: u32) {
        // Never publish dimensions that cannot fit in one slot; that breaks
        // the consumer reshape path for RAW frames.
        if data.len() > self.slot_size {
            eprintln!(
                "[webcam] RAW frame dropped: {} bytes exceeds SHM slot {} bytes ({}x{})",
                data.len(),
                self.slot_size,
                w,
                h
            );
            return;
        }

        // Dimensions are written inside the publication window (see
        // publish_frame) so a reader can never pair them with another frame.
        self.publish_frame(data, FORMAT_RAW_RGB, Some((w, h)));
    }

    /// Publish a JPEG frame (format_flag = 1).
    pub fn publish_frame_jpeg(&self, data: &[u8]) {
        self.publish_frame(data, FORMAT_JPEG, None);
    }

    /// Publish frame — the seqlock writer (see the module docs).
    fn publish_frame(&self, data: &[u8], format: u32, dims: Option<(u32, u32)>) {
        let len = data.len().min(self.slot_size);

        unsafe {
            let seq = self.atomic_u64(off::FRAME_WRITE_SEQ);
            // Open the publication window: odd seq. Single writer, so a plain
            // load+store replaces an RMW; the release fence orders the odd
            // store before every data write below.
            let s = seq.load(Ordering::Relaxed);
            debug_assert!(s.is_multiple_of(2), "publication window left open");
            seq.store(s + 1, Ordering::Relaxed);
            fence(Ordering::Release);

            // Read current active slot, toggle to the other one.
            let cur = self.atomic_u32(off::ACTIVE_SLOT).load(Ordering::Relaxed);
            let new_slot = 1 - cur;

            // Write frame data into the NEW slot.
            let slot_offset = HEADER_SIZE + (new_slot as usize) * self.slot_size;
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(slot_offset), len);

            // All metadata belonging to this frame, inside the window.
            if let Some((w, h)) = dims {
                Self::write_u32(self.ptr, off::WIDTH, w);
                Self::write_u32(self.ptr, off::HEIGHT, h);
            }
            Self::write_u32(self.ptr, off::FORMAT_FLAG, format);
            Self::write_u32(self.ptr, off::FRAME_SIZE, len as u32);
            self.atomic_u32(off::ACTIVE_SLOT)
                .store(new_slot, Ordering::Relaxed);

            // Close the window: even seq publishes everything above.
            seq.store(s + 2, Ordering::Release);
        }
    }

    /// Check whether the consumer has written a new config.  Returns the
    /// deserialized `CameraConfig` if the sequence counter advanced.
    /// Takes `&self`: the polling cursor is atomic, so no exclusive reference
    /// (and none of the old `Arc::as_ptr` → `&mut` casts) is needed.
    pub fn poll_config(&self) -> Option<proto::CameraConfig> {
        unsafe {
            let seq = self.atomic_u32(off::CONFIG_WRITE_SEQ).load(Ordering::Acquire);
            if seq == self.last_config_seq.load(Ordering::Relaxed) {
                return None;
            }
            self.last_config_seq.store(seq, Ordering::Relaxed);

            let size = Self::read_u32(self.ptr, off::CONFIG_SIZE) as usize;
            if size == 0 || size > CONFIG_PAYLOAD_MAX {
                return None;
            }

            let payload =
                std::slice::from_raw_parts(self.ptr.add(off::CONFIG_PAYLOAD), size);
            proto::CameraConfig::decode(payload).ok()
        }
    }

    /// Write health status into the SHM header.
    pub fn publish_health(&self, status: &proto::HealthStatus) {
        let buf = status.encode_to_vec();
        if buf.len() > HEALTH_PAYLOAD_MAX {
            return;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                self.ptr.add(off::HEALTH_PAYLOAD),
                buf.len(),
            );
            Self::write_u32(self.ptr, off::HEALTH_SIZE, buf.len() as u32);
            self.atomic_u32(off::HEALTH_WRITE_SEQ)
                .fetch_add(1, Ordering::Release);
        }
    }

    // ── helpers ──────────────────────────────────────────────────────

    unsafe fn write_u32(base: *mut u8, offset: usize, val: u32) {
        unsafe { (base.add(offset) as *mut u32).write(val) };
    }

    unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
        unsafe { (base.add(offset) as *const u32).read() }
    }

    unsafe fn atomic_u32(&self, offset: usize) -> &AtomicU32 {
        unsafe { &*(self.ptr.add(offset) as *const AtomicU32) }
    }

    unsafe fn atomic_u64(&self, offset: usize) -> &AtomicU64 {
        unsafe { &*(self.ptr.add(offset) as *const AtomicU64) }
    }
}

impl Drop for ShmProducer {
    /// Drop.
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.total_size);
            libc::shm_unlink(self.shm_name.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn poll_config_works_through_a_shared_arc() {
        // poll_config takes &self (atomic cursor), so two handles to the same
        // producer can poll without any exclusive-reference cast.
        let name = format!("conecsa-shm-test-{}", std::process::id());
        let shm = Arc::new(ShmProducer::new(&name, 64, 64).expect("create segment"));
        let clone = Arc::clone(&shm);
        assert!(shm.poll_config().is_none());
        assert!(clone.poll_config().is_none());
        shm.publish_frame_jpeg(b"\xff\xd8fake-jpeg");
    }

    #[test]
    fn published_frames_leave_an_even_generation() {
        // The seqlock contract readers depend on: seq is even between
        // publications and advances by exactly 2 per frame.
        let name = format!("conecsa-shm-seq-test-{}", std::process::id());
        let shm = ShmProducer::new(&name, 64, 64).expect("create segment");
        let seq_at = |shm: &ShmProducer| unsafe {
            shm.atomic_u64(off::FRAME_WRITE_SEQ).load(Ordering::Acquire)
        };
        assert_eq!(seq_at(&shm), 0);
        shm.publish_frame_jpeg(b"one");
        assert_eq!(seq_at(&shm), 2);
        shm.publish_frame_rgb(&[0u8; 64 * 64 * 3], 64, 64);
        assert_eq!(seq_at(&shm), 4);
    }

    #[test]
    fn the_header_declares_protocol_version_2() {
        let name = format!("conecsa-shm-ver-test-{}", std::process::id());
        let shm = ShmProducer::new(&name, 64, 64).expect("create segment");
        let version = unsafe { ShmProducer::read_u32(shm.ptr, off::VERSION) };
        assert_eq!(version, 2, "readers key mixed-version rejection on this");
    }
}
