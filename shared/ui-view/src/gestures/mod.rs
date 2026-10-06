// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: Apache-2.0

//! What a pointer or wheel event on the viewport means, decided by pure
//! functions over the tracked pointers (tested natively); the wasm32
//! [`crate::viewport::Gestures`] applies the decision to the view transform.
//!
//! The viewport claims its gestures in the capture phase, before whatever
//! sits below it (the label editor's drawing handlers, a carousel's swipe)
//! sees them: a middle-button or modifier+drag pans, two fingers pinch and
//! drag, Ctrl+wheel (a trackpad pinch arrives the same way) zooms about the
//! cursor and the plain wheel pans a zoomed image.

/// A position in client px (`ActivePointer::client`, `pinch_pairs`).
pub type Client = (f32, f32);

/// A pointer currently down on the viewport (at most two are kept:
/// the pair a pinch needs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivePointer {
    pub id: i32,
    /// Last known position.
    pub client: Client,
    pub touch: bool,
}

/// A view gesture owned by the viewport, during which nothing below reacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewGesture {
    /// One pointer drags the image (middle button, or Space held).
    Pan { id: i32 },
    /// Two fingers zoom and drag the image.
    Pinch,
    /// A finger left over from a pinch: still down, must not pass through.
    Suppress,
}

/// What a press on the viewport is, once its pointer is tracked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Press {
    /// A content press: whatever sits below the viewport gets it (the
    /// editor draws, a carousel swipes).
    Through,
    /// A non-primary pointer (another mouse button): swallowed, never passed through.
    Ignore,
    /// Mid-pan, mid-pinch or a leftover finger: claimed, nothing new starts.
    Claimed,
    /// The second finger: a pinch begins with every tracked pointer.
    Pinch,
    /// Middle button, or the pan modifier held: a pan begins with this pointer.
    Pan,
}

/// What a wheel event does to the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WheelAction {
    /// Zoom about the cursor.
    Zoom,
    /// Pan by these px (the page must not scroll).
    Pan(f32, f32),
    /// At 100 %: the page scrolls as usual.
    Ignore,
}

/// Remember a pointer that went down: one mouse only (a new mouse press
/// replaces any mouse entry), a touch id is never reused while down, and at
/// most two are kept.
pub fn track(ps: &mut Vec<ActivePointer>, p: ActivePointer) {
    ps.retain(|q| q.id != p.id && (p.touch || q.touch));
    if ps.len() < 2 {
        ps.push(p);
    }
}

/// Classify a press once its pointer is tracked in `ps`. `pan_modifier` is
/// "a primary press pans" (the editor: Space held; a viewer: the image is
/// zoomed). A second finger during a one-finger pan turns it into a pinch —
/// a viewer that pans on one finger must still let two fingers zoom.
pub fn classify_press(
    ps: &[ActivePointer],
    gesture: Option<ViewGesture>,
    touch: bool,
    primary: bool,
    middle: bool,
    pan_modifier: bool,
) -> Press {
    let two_fingers = touch && ps.iter().filter(|p| p.touch).count() >= 2;
    match gesture {
        Some(ViewGesture::Pan { .. }) if two_fingers => return Press::Pinch,
        Some(_) => return Press::Claimed,
        None => {}
    }
    if two_fingers {
        return Press::Pinch;
    }
    if !primary {
        return Press::Ignore;
    }
    if middle || pan_modifier {
        Press::Pan
    } else {
        Press::Through
    }
}

/// The gesture left after pointer `id` went up with `remaining` pointers still
/// down: releasing the panning pointer ends the pan, the finger still down
/// after a pinch must not pass through.
pub fn after_release(g: ViewGesture, id: i32, remaining: usize) -> Option<ViewGesture> {
    match g {
        ViewGesture::Pan { id: pan_id } if pan_id == id => None,
        ViewGesture::Pan { .. } => Some(g),
        ViewGesture::Pinch | ViewGesture::Suppress if remaining > 0 => Some(ViewGesture::Suppress),
        ViewGesture::Pinch | ViewGesture::Suppress => None,
    }
}

/// The two fingers of a pinch before and after pointer `id` moved to
/// `client`, in the order they are tracked; `None` unless exactly two
/// pointers are tracked and `id` is one of them.
pub fn pinch_pairs(
    before: &[ActivePointer],
    id: i32,
    client: Client,
) -> Option<((Client, Client), (Client, Client))> {
    let prev = before.iter().find(|p| p.id == id)?.client;
    let [a, b] = before else {
        return None;
    };
    Some(if a.id == id {
        ((prev, b.client), (client, b.client))
    } else {
        ((a.client, prev), (a.client, client))
    })
}

/// Ctrl/Meta+wheel zooms; the plain wheel pans a zoomed image (Shift+wheel on
/// a mouse scrolls sideways) and, at 100 %, is left to the page.
pub fn wheel_action(zoom_key: bool, shift: bool, dx: f32, dy: f32, at_identity: bool) -> WheelAction {
    if zoom_key {
        return WheelAction::Zoom;
    }
    if at_identity {
        return WheelAction::Ignore;
    }
    if shift && dx == 0.0 {
        WheelAction::Pan(dy, 0.0)
    } else {
        WheelAction::Pan(dx, dy)
    }
}

#[cfg(test)]
mod tests;
