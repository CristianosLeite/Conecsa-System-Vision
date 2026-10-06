# shared/styles/

Shared Tailwind CSS v4 design system for **system-vision** and **hub-vision**.

`input.css` is the single build entrypoint — an index of `@import`s plus the
Tailwind directives (`@custom-variant`, `@plugin`, `@source`). All actual CSS
lives in the partials below. The Tailwind standalone CLI bundles the imports
itself; no PostCSS or Node toolchain is involved.

## Build

Never edit `system-vision/styles.css` or `hub-vision/styles.css` — they are
generated. All consumers compile from `input.css`:

| Consumer | Command |
|---|---|
| `scripts/build.sh` | `bin/tailwindcss -i shared/styles/input.css -o system-vision/styles.css --minify` |
| `scripts/dev.sh` | same, with `--watch` (watches the partials too) |
| `scripts/build-hub.sh` | `bin/tailwindcss -i shared/styles/input.css -o hub-vision/styles.css --minify` |
| `system-vision/Dockerfile.system-vision[.dev]` | copies `shared/styles/`, builds in-image |
| `hub-vision/Dockerfile.hub-builder` | copies `shared/styles/`, builds in-image |

Utility classes are generated from the sources listed in the `@source` globs in
`input.css` (both crates' `.rs` files + `system-vision/index.html`). Missing
globs are tolerated, so each Docker context can carry only its own crate.

## Files

| File | Contents |
|---|---|
| `theme.css` | Design tokens: `@theme` (Tailwind colors/fonts) + `:root` runtime CSS variables (surfaces, borders, text, accents, state colors, shadows) |
| `fonts.css` | Bundled `@font-face`: Good Times (brand) + Inter (UI; see comment on why it must be bundled) |
| `base.css` | `@layer base` globals — root `font-size` density knob, `html`/`body`, button cursor restore — plus unlayered scrollbar styling |
| `app-shell.css` | Viewport frame (`.app-scale-*`, `.app-shell`), header, brand, product title, status pills, alert slot, header-condense media queries, `status-pulse` keyframes |
| `power.css` | Header power button, dropdown, confirm flow, `power-spin` keyframes |
| `app-layout.css` | `.app-main` scroll region, dashboard grid + tablet/desktop media queries, `.app-panel`, `.app-flow-*`, `.panel-header`/`.panel-title`, `.nav-button*` |
| `ui-core.css` | Core `ui-*` primitives: cards, section headers/titles, labels/help/values, rows, list boxes, dividers, topbar, pin rows, color swatch, dataset gallery cards |
| `auth.css` | Auth gate / login screen |
| `ui-controls.css` | Interactive controls: buttons + variants, icon buttons, badges, inputs/select/textarea/range/radio, tabs, toggle, menu, choice |
| `ui-feedback.css` | Alerts, progress bars, code block, spinner |
| `ui-overlays.css` | Stream-stage overlays (tool panel, drawer + container query), overlay controls, segmented buttons, area chips, thumbs, modals |
| `label-editor.css` | Label-editor canvas (training) SVG presentation |
| `image-viewer.css` | Generic zoomable view (`.ui-zoom-viewport`/`.ui-zoom-stage`, the `conecsa-ui-view` stage contract) and the hub's full-screen image viewer (bar, arrows, viewport sizing, corner chips, notice) |
| `training.css` | Training pages layout: wrapping top bars and panel headers, the dataset editor's responsive grid (1 / 2 / 3 columns), gallery thumbnails, class names |

Component partials wrap their rules in their own `@layer components { … }`
block; Tailwind merges same-named layer blocks across files.

## Ordering constraints

The `@import` order in `input.css` is load-bearing — same-specificity rules
resolve by source order:

- `ui-core.css` must come **before** `auth.css`: `.auth-gate-card` overrides
  `.ui-card`'s border purely by coming later.
- Media-query overrides live in the **same file** as (and after) the base rules
  they override — keep it that way when adding responsive tweaks.
- `@keyframes` are global across files: `.ui-spinner` (ui-feedback.css) reuses
  `power-spin` from power.css.
- The scrollbar rules in `base.css` are intentionally **unlayered**.

## Conventions

- New component classes go in the partial that matches their concern (or a new
  partial, imported from `input.css`), wrapped in `@layer components`.
- Use the `:root` variables from `theme.css` for colors/surfaces instead of
  hard-coded values, so the palette stays swappable in one place.
- Sizing is in `rem`: the `html { font-size: 80% }` knob in `base.css` scales
  the whole UI proportionally.

### Contextual actions

Use icon-only buttons for contextual editing, renaming, removal, and saving or
cancelling an inline edit. Keep text on primary form actions, menu items, and
confirmation buttons. Reuse the existing Heroicons outline style: a 24-unit
grid, current-colour strokes of width 2, and rounded caps and joins. Use a
pencil for editing, a trash can for removal, a check for saving, and an X for
cancelling.

Use `ui-icon-button ui-icon-button-md` with the existing semantic colour
variants; media overlays retain their `ui-tool-button` / `ui-thumb-action`
presentation. Each icon-only button needs a localized `aria-label` for its
action and a localized `title` for its tooltip. A disabled control can explain
its restriction in the tooltip while keeping the action as its accessible
name. Decorative SVGs use `aria-hidden="true"`.

Icon controls have visible keyboard focus and a minimum 32 px hit area,
increased to 44 px when any pointer is coarse. These accessibility minima use
pixels so the root density setting does not shrink them. Thumbnail actions
appear on focus as well as hover, and stay visible for touch input. Two
modifiers keep a compact element usable on touch: `ui-thumb-actions-compact`
holds a thumbnail's actions in one 32 px row along the top edge (the gallery
thumbnails and the dataset cards, where three actions share a small picture),
and `ui-icon-button-compact`
is 36 px for buttons inside a row that is itself the tap target (class rows);
tool strips keep 44 px. A confirmation with a picture uses `ui-modal-lead`
with `ui-modal-preview` beside `ui-modal-lead-text`.
