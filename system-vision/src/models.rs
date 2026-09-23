// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Serde data structures shared across the frontend (status, models, stats, …).

use serde::{Deserialize, Serialize};

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub is_running: bool,
    pub model: String,
    pub confidence_threshold: f32,
    pub overlay_threshold: f32,
    #[serde(default)]
    pub acceleration_type: String,
    /// False while the webcam-server reports no streaming camera: detection
    /// cannot be started and the stream shows the disconnected placeholder.
    /// Defaults to true so older firmware without the field never makes the
    /// UI claim the camera is gone.
    #[serde(default = "camera_connected_default")]
    pub camera_connected: bool,
    /// The device's application task ("detect", "classify", "segment", "face");
    /// `None` while none is chosen. The authoritative read is
    /// `GET /api/v1/application`; the 5 s status poll only notices changes.
    #[serde(default)]
    pub task: Option<String>,
    /// Segmentation instance limit in effect (1..255); `None` from older
    /// firmware.
    #[serde(default)]
    pub segment_max_instances: Option<u32>,
    /// Face recognition: the cosine similarity (0..1) a face needs to take a
    /// person's name; `None` from firmware without face recognition.
    #[serde(default)]
    pub face_match_threshold: Option<f32>,
    /// Face recognition: faces smaller than this (pixels, 0..1024) are ignored.
    #[serde(default)]
    pub face_min_size_px: Option<u32>,
    /// Face recognition: most faces recognized per frame (1..20).
    #[serde(default)]
    pub face_max_faces: Option<u32>,
    pub stats: PerformanceStats,
    pub protocols: ProtocolInfo,
}

fn camera_connected_default() -> bool {
    true
}

/// An application task: what the device's models, datasets and results are
/// for. The ids equal the backend's (`detect`, `classify`, `segment`,
/// `face`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Task {
    Detect,
    Classify,
    Segment,
    /// Face recognition: detection boxes named after the enrolled people.
    Face,
}

impl Task {
    /// Every task, in the order the UI lists them.
    pub const ALL: [Task; 4] = [Task::Detect, Task::Classify, Task::Segment, Task::Face];

    /// The task for a backend id; `None` for an id this UI does not know.
    pub fn parse(id: &str) -> Option<Task> {
        match id {
            "detect" => Some(Task::Detect),
            "classify" => Some(Task::Classify),
            "segment" => Some(Task::Segment),
            "face" => Some(Task::Face),
            _ => None,
        }
    }

    /// The backend id.
    pub fn id(self) -> &'static str {
        match self {
            Task::Detect => "detect",
            Task::Classify => "classify",
            Task::Segment => "segment",
            Task::Face => "face",
        }
    }

    /// Whether a dataset of this task labels a whole image with one class:
    /// classification, and face recognition (one class per person).
    pub fn uses_image_class(self) -> bool {
        matches!(self, Task::Classify | Task::Face)
    }
}

/// `GET/PUT /api/v1/application`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApplicationInfo {
    /// `None` while no application has been chosen.
    #[serde(default)]
    pub task: Option<String>,
    /// Tasks the device's build can run; the others are shown disabled.
    #[serde(default)]
    pub supported_tasks: Vec<String>,
    /// True when an upgrade set the task, not an administrator.
    #[serde(default)]
    pub migrated: bool,
}

impl ApplicationInfo {
    /// Whether the device can run `task`.
    pub fn supports(&self, task: Task) -> bool {
        self.supported_tasks.iter().any(|t| t == task.id())
    }
}

/// What the UI knows about the device's application.
///
/// `Loading` is distinct from `Unset` so the selector never flashes before the
/// first read answers, and `Error` (the read failed) never shows it either.
/// An id this UI does not know comes from a newer device: `Unsupported`, never
/// `Unset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppState {
    Loading,
    Unset,
    Set(Task),
    Unsupported(String),
    Error,
}

impl AppState {
    /// The state a reported task id stands for.
    pub fn from_task(task: Option<&str>) -> AppState {
        match task.map(str::trim) {
            None | Some("") => AppState::Unset,
            Some(id) => Task::parse(id)
                .map(AppState::Set)
                .unwrap_or_else(|| AppState::Unsupported(id.to_string())),
        }
    }

    /// The chosen task, when this UI knows it.
    pub fn task(&self) -> Option<Task> {
        match self {
            AppState::Set(task) => Some(*task),
            _ => None,
        }
    }

    /// True when the dashboard must give way to the application selector.
    pub fn gated(&self) -> bool {
        matches!(self, AppState::Unset | AppState::Unsupported(_))
    }

    /// True once a read has answered (the state is authoritative).
    pub fn is_known(&self) -> bool {
        !matches!(self, AppState::Loading | AppState::Error)
    }
}

/// `GET /api/v1/detections/snapshot`: the device's latest result.
///
/// Only what the UI reads; the frames are never requested by it. For
/// classification `detections` holds at most the class above the threshold
/// (an item without `bbox`) and `candidates` the top-k.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub total: u32,
    #[serde(default)]
    pub detections: Vec<SnapshotItem>,
    /// Classification top-k, highest first; absent for other tasks.
    #[serde(default)]
    pub candidates: Option<Vec<Candidate>>,
    /// Set when the device dropped polygon rings to keep the snapshot small.
    #[serde(default)]
    pub polygons_truncated: bool,
}

/// One result of a snapshot: a detected object, or a frame's class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotItem {
    pub class_name: String,
    #[serde(default)]
    pub confidence: f32,
    /// The class color as `#rrggbb`, matching the burned-in overlay.
    #[serde(default)]
    pub color: Option<String>,
    /// Normalized corners `[x1, y1, x2, y2]`; absent for a classification.
    #[serde(default)]
    pub bbox: Option<[f32; 4]>,
    /// A segmentation instance's exterior rings as normalized `[x, y]`
    /// vertices; absent for other tasks.
    #[serde(default)]
    pub polygons: Option<Vec<Vec<[f32; 2]>>>,
}

/// One classification candidate (a class and its probability).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
}

impl Snapshot {
    /// The class above the threshold of a classification frame, if any.
    pub fn top_class(&self) -> Option<&SnapshotItem> {
        self.detections.first()
    }
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub fps: f32,
    pub inference_time: f32,
    pub detections: u32,
    pub frames_with_detections: u64,
}

/// Protocol information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolInfo {
    #[serde(default)]
    pub http_port: u16,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: u64,
    pub modified: f64,
    pub is_active: bool,
    /// The model's task; older firmware sends none, and every such model is
    /// an object-detection model.
    #[serde(default = "detect_task")]
    pub task: String,
}

impl ModelInfo {
    /// A `.faces` enrolment package: listed while its gallery builds (and
    /// left behind by an interrupted build) but never selectable — the
    /// model built from it is the `.engine` of the same name.
    pub fn is_enrollment_package(&self) -> bool {
        self.name.to_ascii_lowercase().ends_with(".faces")
    }
}

fn detect_task() -> String {
    Task::Detect.id().to_string()
}

/// Configuration for the detection system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub capture_device: String,
    pub capture_resolution: (u32, u32),
    pub capture_framerate: u32,
    pub model_path: String,
    pub confidence_threshold: f32,
}

/// System metrics (CPU, RAM, Disk, Temperature)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub ram_usage: f32,
    pub ram_total: u64,
    pub ram_used: u64,
    pub disk_usage: f32,
    pub disk_total: u64,
    pub disk_used: u64,
    pub temperature: Option<f32>,
    #[serde(default)]
    pub gpu_usage: Option<f32>,
    #[serde(default)]
    pub gpu_temperature: Option<f32>,
    #[serde(default)]
    pub gpu_freq_mhz: Option<f32>,
    #[serde(default)]
    pub gpu_max_freq_mhz: Option<f32>,
}

#[cfg(test)]
mod tests;
