# conecsa-ui-view

Shared Leptos building blocks for a zoomable image view, used by the device UI
(`system-vision`, its label editor) and by the hub (`hub-vision`, the records and
dataset image viewer). Apache-2.0, like `shared/styles/`, `shared/i18n/` and `proto/`.

- `transform` — pure math: `ViewTransform { zoom, ox, oy }` in viewport fractions,
  zoom about an anchor, pan, pinch, wheel helpers. No dependencies; tested natively.
- `gestures` — pure decisions over the tracked pointers: what a press, a release or a
  wheel event means (`Press`, `ViewGesture`, `WheelAction`). Tested natively.
- `viewport` (wasm32) — `ViewState` (the signals), `Gestures` (applies the decisions to
  the DOM events in the capture phase) and `watch_viewport` (ResizeObserver plus a
  non-passive wheel listener).
- `zoom_guard` (wasm32) — keeps the whole page from being zoomed (Ctrl+wheel, Ctrl+±,
  WebKit gesture events); installed once per app from the entry point.

The DOM contract: a *viewport* element with `overflow: hidden` and `touch-action: none`
holds a *stage* laid out from three CSS custom properties (`--label-zoom`, `--label-ox`,
`--label-oy`) — `left/top: calc(ox · zoom · -100%)`, `width/height: calc(zoom · 100%)` —
never `transform: scale()`, so the picture stays crisp and an overlay drawn in screen
pixels keeps fixed-size strokes. The CSS lives in `shared/styles/` (`label-editor.css`,
`image-viewer.css`).

```bash
cargo test --manifest-path shared/ui-view/Cargo.toml                 # math + decisions
wasm-pack test --headless --firefox shared/ui-view                   # DOM smoke test
cargo clippy --manifest-path shared/ui-view/Cargo.toml --all-targets -- -D warnings
```
