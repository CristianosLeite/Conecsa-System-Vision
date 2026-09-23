# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""conecsa_shm — shared POSIX shared-memory ring helpers.

Installed into the `conecsa-os-base:base` image so every service built on it
(inference-service, api-gateway, training-service) imports one implementation
of the camera and processed-frame ring layouts.

- ``camera_ring.CameraRingReader``    — reads the webcam-server camera ring.
- ``processed_ring.ProcessedFrameWriter`` / ``ProcessedFrameReader`` — the
  inference→gateway processed-JPEG ring.
- ``stereo.combine_stereo``           — side-by-side stereo blend (pure
  function), so dataset capture / previews match the live detector's view.

Pure struct + numpy/cv2; no protobuf dependency (the camera config/health
payloads cross as opaque bytes — the caller owns the schema).
"""
