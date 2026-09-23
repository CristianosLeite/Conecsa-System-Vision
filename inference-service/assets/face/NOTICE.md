# Face recognition models (third party)

The face application type runs two pretrained models that are **not** part of
this repository. `scripts/fetch-face-models.sh` downloads them into this
directory against the checksums in `scripts/face-models.pin`, and the
inference-service image verifies them again at build time. Neither file is
committed, and neither is redistributed by the open-source mirror.

| Model | File | License | Source |
|---|---|---|---|
| YuNet (face detection, 5 landmarks) | `face_detection_yunet_2023mar.onnx` | MIT — © Shiqi Yu and contributors | <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet> |
| SFace (face recognition embeddings) | `face_recognition_sface_2021dec.onnx` | Apache-2.0 | <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface> |

Both come from the OpenCV Model Zoo. The decode and alignment in
`api/postprocess/_yunet.py` and `api/postprocess/_face_align.py` follow
OpenCV's own `FaceDetectorYN` and `FaceRecognizerSF` implementations (Apache-2.0),
so enrolled and live embeddings match what OpenCV would produce.

Conecsa reviewed the licenses of both models together with the provenance of
the data they were trained on and cleared them, as bundled here, for
commercial use in Conecsa System Vision (review closed 2026-09-22).

These models perform no liveness detection: a printed photo or a screen can be
recognized as the person. Face recognition must not be the only factor
protecting anything valuable.
