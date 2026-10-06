// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the label canvas view math.
use super::*;

fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-4, "{a} != {b}");
}

/// The normalized image point under a viewport fraction.
fn under(v: ViewTransform, a: Frac) -> Frac {
    (v.ox + a.0 / v.zoom, v.oy + a.1 / v.zoom)
}

#[test]
fn default_is_identity() {
    let v = ViewTransform::default();
    assert_eq!(v, ViewTransform::IDENTITY);
    assert!(v.is_identity());
    assert_eq!(v.percent(), 100);
    assert_eq!(v.clamp(), v);
}

#[test]
fn zooming_keeps_the_point_under_the_anchor() {
    let v = ViewTransform::IDENTITY.zoom_by(2.0, (0.5, 0.5));
    approx(v.zoom, 2.0);
    approx(v.ox, 0.25);
    approx(v.oy, 0.25);
    for anchor in [(0.5, 0.5), (0.2, 0.7), (0.9, 0.1)] {
        for factor in [1.5, 2.0, 0.8] {
            // Start from a view well inside the clamp so it does not interfere.
            let start = ViewTransform {
                zoom: 3.0,
                ox: 0.3,
                oy: 0.3,
            };
            let before = under(start, anchor);
            let after = under(start.zoom_by(factor, anchor), anchor);
            approx(before.0, after.0);
            approx(before.1, after.1);
        }
    }
}

#[test]
fn zoom_is_bounded() {
    let v = ViewTransform::IDENTITY.zoom_by(100.0, (0.5, 0.5));
    approx(v.zoom, ZOOM_MAX);
    assert_eq!(v.percent(), 800);
    let v = v.zoom_by(0.0001, (0.5, 0.5));
    assert_eq!(v, ViewTransform::IDENTITY);
    let v = ViewTransform::IDENTITY.zoom_to(f32::NAN, (0.5, 0.5));
    assert_eq!(v, ViewTransform::IDENTITY);
}

#[test]
fn zooming_back_to_one_clears_the_offset() {
    let v = ViewTransform {
        zoom: 4.0,
        ox: 0.6,
        oy: 0.2,
    }
    .zoom_to(1.0, (0.1, 0.9));
    assert_eq!(v, ViewTransform::IDENTITY);
    assert!(v.is_identity());
}

#[test]
fn clamp_keeps_the_image_covering_the_viewport() {
    let v = ViewTransform {
        zoom: 2.0,
        ox: 0.9,
        oy: -0.3,
    }
    .clamp();
    approx(v.ox, 0.5);
    approx(v.oy, 0.0);
    let v = ViewTransform {
        zoom: f32::INFINITY,
        ox: f32::NAN,
        oy: 0.1,
    }
    .clamp();
    assert_eq!(v, ViewTransform::IDENTITY);
}

#[test]
fn pan_moves_the_content_and_clamps_at_both_ends() {
    let v = ViewTransform {
        zoom: 2.0,
        ox: 0.25,
        oy: 0.25,
    };
    // Dragging the content right by a tenth of the viewport reveals the left.
    let p = v.pan_by((0.1, 0.0));
    approx(p.ox, 0.2);
    approx(p.oy, 0.25);
    let p = v.pan_by((5.0, -5.0));
    approx(p.ox, 0.0);
    approx(p.oy, 0.5);
    // At zoom 1 there is nothing to pan.
    assert_eq!(ViewTransform::IDENTITY.pan_by((0.3, 0.3)), ViewTransform::IDENTITY);
}

#[test]
fn pinch_scales_by_the_distance_ratio_and_follows_the_midpoint() {
    let v = ViewTransform {
        zoom: 2.0,
        ox: 0.25,
        oy: 0.25,
    };
    // Fingers spread from 0.2 apart to 0.4 apart around the same midpoint.
    let p = v.pinch(((0.4, 0.5), (0.6, 0.5)), ((0.3, 0.5), (0.7, 0.5)));
    approx(p.zoom, 4.0);
    let before = under(v, (0.5, 0.5));
    let after = under(p, (0.5, 0.5));
    approx(before.0, after.0);
    approx(before.1, after.1);
    // Same spread, midpoint dragged right by 0.1: the content follows.
    let q = v.pinch(((0.4, 0.5), (0.6, 0.5)), ((0.5, 0.5), (0.7, 0.5)));
    approx(q.zoom, 2.0);
    approx(q.ox, 0.2);
    // Fingers on the same spot never divide by zero.
    let r = v.pinch(((0.5, 0.5), (0.5, 0.5)), ((0.4, 0.5), (0.6, 0.5)));
    approx(r.zoom, 2.0);
}

#[test]
fn wheel_deltas_convert_by_mode() {
    approx(wheel_delta_px(100.0, 0), 100.0);
    approx(wheel_delta_px(3.0, 1), 3.0 * WHEEL_LINE_PX);
    approx(wheel_delta_px(1.0, 2), WHEEL_PAGE_PX);
    approx(wheel_delta_px(-2.0, 7), -2.0);
}

#[test]
fn wheel_zoom_factor_zooms_in_on_wheel_up_and_is_bounded() {
    assert!(wheel_zoom_factor(-100.0) > 1.0);
    assert!(wheel_zoom_factor(100.0) < 1.0);
    approx(wheel_zoom_factor(0.0), 1.0);
    approx(wheel_zoom_factor(-100.0) * wheel_zoom_factor(100.0), 1.0);
    approx(wheel_zoom_factor(-100_000.0), WHEEL_FACTOR_MAX);
    approx(wheel_zoom_factor(100_000.0), WHEEL_FACTOR_MIN);
}
