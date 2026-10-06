// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Pure math for a zoomable image view: zooming the image about a point,
//! panning it and pinching it, clamped so the image always covers the
//! viewport.
//!
//! The viewport keeps its layout size; the image (and its overlay) is
//! a *stage* laid out inside it, `zoom` times as wide and tall, offset so the
//! normalized image point `(ox, oy)` sits at the viewport's top-left corner.
//! Everything here is in *viewport fractions* (0..1 of the viewport box) or
//! normalized image coordinates, so it needs no measurement: the component
//! turns the transform into CSS custom properties and percentages do the rest.

/// No zoom: the whole image fits the viewport.
pub const ZOOM_MIN: f32 = 1.0;
/// The most a native-resolution frame usefully magnifies on screen.
pub const ZOOM_MAX: f32 = 8.0;
/// Zoom factor of one button press or key.
pub const ZOOM_STEP: f32 = 1.25;
/// Wheel zoom: `factor = exp(-delta_px * RATE)` — a 100 px mouse notch is
/// ≈ 1.28×, a trackpad pinch (a few px per event) stays smooth.
pub const WHEEL_ZOOM_RATE: f32 = 0.0025;
/// Bounds of a single wheel event's factor (a page-mode delta must not jump
/// from 1× to 8×).
const WHEEL_FACTOR_MIN: f32 = 0.5;
const WHEEL_FACTOR_MAX: f32 = 2.0;
/// One `deltaMode` line / page, in px (Firefox reports lines for a mouse).
pub const WHEEL_LINE_PX: f32 = 16.0;
pub const WHEEL_PAGE_PX: f32 = 400.0;

/// A point in viewport fractions (0..1 of the viewport box).
pub type Frac = (f32, f32);

/// The view of the image inside the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewTransform {
    /// Stage size relative to the viewport (1 = fits).
    pub zoom: f32,
    /// Normalized image point shown at the viewport's top-left corner.
    pub ox: f32,
    pub oy: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl ViewTransform {
    /// The whole image, unzoomed.
    pub const IDENTITY: Self = Self {
        zoom: ZOOM_MIN,
        ox: 0.0,
        oy: 0.0,
    };

    /// Not zoomed in at all (a clamped transform at zoom 1 has no offset).
    pub fn is_identity(self) -> bool {
        self.zoom <= ZOOM_MIN
    }

    /// The zoom as a whole percentage, for the toolbar readout.
    pub fn percent(self) -> u32 {
        (self.zoom * 100.0).round() as u32
    }

    /// Zoom within bounds and the offsets inside `[0, 1 - 1/zoom]`, so the
    /// stage always covers the viewport (no gutters) and zoom 1 shows the
    /// whole image with no offset. Non-finite input collapses to identity.
    pub fn clamp(self) -> Self {
        let zoom = if self.zoom.is_finite() {
            self.zoom.clamp(ZOOM_MIN, ZOOM_MAX)
        } else {
            ZOOM_MIN
        };
        let max = 1.0 - 1.0 / zoom;
        let clamp = |o: f32| if o.is_finite() { o.clamp(0.0, max) } else { 0.0 };
        Self {
            zoom,
            ox: clamp(self.ox),
            oy: clamp(self.oy),
        }
    }

    /// Zoom to `zoom`, keeping the image point under `anchor` (a viewport
    /// fraction) where it is — the cursor stays on what it points at. Near an
    /// edge the clamp may shift it, so the image keeps covering the viewport.
    pub fn zoom_to(self, zoom: f32, anchor: Frac) -> Self {
        let zoom = if zoom.is_finite() {
            zoom.clamp(ZOOM_MIN, ZOOM_MAX)
        } else {
            self.zoom
        };
        // The image point under the anchor: p = o + a / z.
        let (px, py) = (self.ox + anchor.0 / self.zoom, self.oy + anchor.1 / self.zoom);
        Self {
            zoom,
            ox: px - anchor.0 / zoom,
            oy: py - anchor.1 / zoom,
        }
        .clamp()
    }

    /// Multiply the zoom by `factor` about `anchor`.
    pub fn zoom_by(self, factor: f32, anchor: Frac) -> Self {
        self.zoom_to(self.zoom * factor, anchor)
    }

    /// The content dragged by `moved` viewport fractions (a drag to the right
    /// reveals more of the left side).
    pub fn pan_by(self, moved: Frac) -> Self {
        Self {
            zoom: self.zoom,
            ox: self.ox - moved.0 / self.zoom,
            oy: self.oy - moved.1 / self.zoom,
        }
        .clamp()
    }

    /// One step of a two-finger gesture: the fingers moved from `prev` to
    /// `next` (viewport fractions). Zooms by the change of their distance
    /// about their previous midpoint, then follows the midpoint.
    pub fn pinch(self, prev: (Frac, Frac), next: (Frac, Frac)) -> Self {
        let (d0, d1) = (distance(prev.0, prev.1), distance(next.0, next.1));
        let factor = if d0 > 1e-4 { d1 / d0 } else { 1.0 };
        let (m0, m1) = (midpoint(prev.0, prev.1), midpoint(next.0, next.1));
        self.zoom_by(factor, m0)
            .pan_by((m1.0 - m0.0, m1.1 - m0.1))
    }
}

fn distance(a: Frac, b: Frac) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn midpoint(a: Frac, b: Frac) -> Frac {
    ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0)
}

/// A wheel delta in px whatever its `deltaMode` (0 px, 1 lines, 2 pages).
pub fn wheel_delta_px(delta: f64, mode: u32) -> f32 {
    let delta = delta as f32;
    match mode {
        1 => delta * WHEEL_LINE_PX,
        2 => delta * WHEEL_PAGE_PX,
        _ => delta,
    }
}

/// The zoom factor of a Ctrl+wheel / pinch step of `dy_px`: wheel up (negative
/// delta) zooms in, bounded per event.
pub fn wheel_zoom_factor(dy_px: f32) -> f32 {
    (-dy_px * WHEEL_ZOOM_RATE)
        .exp()
        .clamp(WHEEL_FACTOR_MIN, WHEEL_FACTOR_MAX)
}

#[cfg(test)]
mod tests;
