// SPDX-FileCopyrightText: 2026 Conecsa
//
// SPDX-License-Identifier: AGPL-3.0-only

//! Per-epoch training charts: which metric each chart plots for a task, the
//! pure geometry of the SVG lines (unit-tested), and the chart card itself.

use leptos::prelude::*;

use crate::api::EpochMetrics;
use crate::models::Task;

/// What a chart plots; picks its title and its y-axis scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChartKind {
    /// mAP@50:95 (box for detect, mask for segment).
    Map,
    /// Top-1 accuracy (classify).
    Accuracy,
    BoxLoss,
    ClassLoss,
    DflLoss,
    SegLoss,
    /// The single classification loss.
    Loss,
}

impl ChartKind {
    /// Quality metrics are fractions, drawn on a fixed 0–1 axis; losses
    /// auto-scale to their own range.
    pub(super) fn is_fraction(self) -> bool {
        matches!(self, ChartKind::Map | ChartKind::Accuracy)
    }
}

/// One chart: its kind and the history keys of its lines. A quality metric
/// has a single validation line (`train` is `None`); a loss has both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChartSpec {
    pub kind: ChartKind,
    pub train: Option<&'static str>,
    pub val: &'static str,
}

const fn metric(kind: ChartKind, key: &'static str) -> ChartSpec {
    ChartSpec {
        kind,
        train: None,
        val: key,
    }
}

const fn loss(kind: ChartKind, train: &'static str, val: &'static str) -> ChartSpec {
    ChartSpec {
        kind,
        train: Some(train),
        val,
    }
}

const BOX: ChartSpec = loss(ChartKind::BoxLoss, "train/box_loss", "val/box_loss");
const CLS: ChartSpec = loss(ChartKind::ClassLoss, "train/cls_loss", "val/cls_loss");
const DFL: ChartSpec = loss(ChartKind::DflLoss, "train/dfl_loss", "val/dfl_loss");
const SEG: ChartSpec = loss(ChartKind::SegLoss, "train/seg_loss", "val/seg_loss");

/// The charts a task's training screen shows, in display order. A face
/// gallery has no epochs, so it has none.
pub(super) fn charts_for(task: Task) -> Vec<ChartSpec> {
    match task {
        Task::Detect => vec![metric(ChartKind::Map, "metrics/mAP50-95(B)"), BOX, CLS, DFL],
        Task::Segment => vec![metric(ChartKind::Map, "metrics/mAP50-95(M)"), BOX, CLS, SEG],
        Task::Classify => vec![
            metric(ChartKind::Accuracy, "metrics/accuracy_top1"),
            loss(ChartKind::Loss, "train/loss", "val/loss"),
        ],
        Task::Face => Vec::new(),
    }
}

/// The finite `(epoch, value)` points of one history key, oldest first.
pub(super) fn series(history: &[EpochMetrics], key: &str) -> Vec<(u32, f64)> {
    history
        .iter()
        .filter_map(|row| {
            row.values
                .get(key)
                .filter(|v| v.is_finite())
                .map(|v| (row.epoch, *v))
        })
        .collect()
}

/// Whether any epoch reported a value for this chart (charts the backend
/// never fills — another ultralytics version, say — are hidden).
pub(super) fn has_data(spec: &ChartSpec, history: &[EpochMetrics]) -> bool {
    history.iter().any(|row| {
        row.values.contains_key(spec.val) || spec.train.is_some_and(|k| row.values.contains_key(k))
    })
}

/// The y-axis range for these lines: 0–1 for a fraction, otherwise the
/// lines' own span padded by 5% (a flat line gets a visible band).
pub(super) fn y_range(lines: &[&[(u32, f64)]], fraction: bool) -> (f64, f64) {
    if fraction {
        return (0.0, 1.0);
    }
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &(_, v) in lines.iter().flat_map(|l| l.iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() {
        return (0.0, 1.0);
    }
    let pad = if hi > lo {
        (hi - lo) * 0.05
    } else {
        lo.abs().max(1.0) * 0.1
    };
    ((lo - pad).max(0.0), hi + pad)
}

/// Plot area inside the SVG `viewBox` (left gutter holds the y labels, the
/// bottom one the x labels).
pub(super) const VIEW_W: f64 = 320.0;
pub(super) const VIEW_H: f64 = 150.0;
const PLOT_X: f64 = 40.0;
const PLOT_Y: f64 = 8.0;
const PLOT_W: f64 = VIEW_W - PLOT_X - 8.0;
const PLOT_H: f64 = VIEW_H - PLOT_Y - 22.0;

/// SVG coordinates of an `(epoch, value)` point. The x axis always spans the
/// whole run (epoch 1 … `total`), so the line grows as the run advances.
pub(super) fn to_xy(epoch: u32, value: f64, total: u32, (lo, hi): (f64, f64)) -> (f64, f64) {
    let span = f64::from(total.max(2) - 1);
    let x = PLOT_X + f64::from(epoch.saturating_sub(1)) / span * PLOT_W;
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    (x, PLOT_Y + (1.0 - t) * PLOT_H)
}

/// The `points` attribute of a `<polyline>` for one line.
pub(super) fn polyline_points(points: &[(u32, f64)], total: u32, range: (f64, f64)) -> String {
    points
        .iter()
        .map(|&(e, v)| {
            let (x, y) = to_xy(e, v, total, range);
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The rows of a chart's screen-reader table: every epoch either line has a
/// point for, oldest first, with each line's value (if any) at that epoch.
pub(super) fn data_rows(
    train: &[(u32, f64)],
    val: &[(u32, f64)],
) -> Vec<(u32, Option<f64>, Option<f64>)> {
    let mut epochs: Vec<u32> = train.iter().chain(val).map(|&(e, _)| e).collect();
    epochs.sort_unstable();
    epochs.dedup();
    let at = |s: &[(u32, f64)], e| s.iter().find(|&&(x, _)| x == e).map(|&(_, v)| v);
    epochs.into_iter().map(|e| (e, at(train, e), at(val, e))).collect()
}

/// A value as the chart header and the y axis print it.
pub(super) fn format_value(v: f64) -> String {
    if v.abs() >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.3}")
    }
}

/// The translated texts a chart card needs.
pub(super) struct ChartLabels {
    pub title: String,
    pub train: String,
    pub val: String,
    pub waiting: String,
    pub epoch: String,
}

/// One chart card: title with the latest values, then the line chart (or a
/// placeholder before the first epoch).
pub(super) fn metric_chart(
    spec: ChartSpec,
    history: &[EpochMetrics],
    total: u32,
    labels: ChartLabels,
) -> impl IntoView {
    let train = spec.train.map(|k| series(history, k)).unwrap_or_default();
    let val = series(history, spec.val);
    let empty = train.is_empty() && val.is_empty();
    let range = y_range(&[&train, &val], spec.kind.is_fraction());
    let total = total.max(history.iter().map(|r| r.epoch).max().unwrap_or(1));
    let latest = |s: &[(u32, f64)]| s.last().map(|&(_, v)| format_value(v));
    let (lo, hi) = range;
    let mid = (lo + hi) / 2.0;
    let y_ticks = [hi, mid, lo].map(|v| (to_xy(1, v, total, range).1, format_value(v)));
    let x_left = PLOT_X;
    // The plot's right edge, not `to_xy(total, …)`: a one-epoch run would
    // collapse the grid and stack the x labels at the left edge.
    let x_right = PLOT_X + PLOT_W;
    let axis_y = VIEW_H - 6.0;
    let last_dot = |s: &[(u32, f64)]| s.last().map(|&(e, v)| to_xy(e, v, total, range));
    let train_dot = last_dot(&train);
    let val_dot = last_dot(&val);
    let train_points = polyline_points(&train, total, range);
    let val_points = polyline_points(&val, total, range);
    let has_train = spec.train.is_some();
    let ChartLabels {
        title,
        train: train_label,
        val: val_label,
        waiting,
        epoch,
    } = labels;
    // The SVG is decorative to assistive technology; this table carries the
    // same values (the header's title doubles as the caption).
    let rows = data_rows(&train, &val);
    let caption = title.clone();
    let val_header = if has_train { val_label.clone() } else { title.clone() };
    let train_header = train_label.clone();
    let epoch_header = epoch.clone();
    let cell = |v: Option<f64>| v.map(format_value).unwrap_or_default();

    view! {
        <section class="ui-card ui-chart-card">
            <header class="ui-chart-header">
                <h3 class="ui-chart-title">{title}</h3>
                <div class="ui-chart-latest">
                    {has_train.then(|| latest(&train).map(|v| view! {
                        <span class="ui-chart-latest-train">{v}</span>
                    }))}
                    {latest(&val).map(|v| view! { <span class="ui-chart-latest-val">{v}</span> })}
                </div>
            </header>
            {if empty {
                view! { <p class="ui-chart-empty ui-help">{waiting}</p> }.into_any()
            } else {
                view! {
                    <svg
                        class="ui-chart-svg"
                        viewBox=format!("0 0 {VIEW_W} {VIEW_H}")
                        aria-hidden="true"
                    >
                        {y_ticks.into_iter().map(|(y, text)| view! {
                            <line class="ui-chart-grid" x1=x_left x2=x_right y1=y y2=y/>
                            <text class="ui-chart-tick" x=x_left - 4.0 y=y + 3.0 text-anchor="end">
                                {text}
                            </text>
                        }).collect_view()}
                        <text class="ui-chart-tick" x=x_left y=axis_y text-anchor="start">
                            "1"
                        </text>
                        <text class="ui-chart-tick" x=(x_left + x_right) / 2.0 y=axis_y text-anchor="middle">
                            {epoch}
                        </text>
                        {(total > 1).then(|| view! {
                            <text class="ui-chart-tick" x=x_right y=axis_y text-anchor="end">
                                {total.to_string()}
                            </text>
                        })}
                        {has_train.then(|| view! {
                            <polyline class="ui-chart-line ui-chart-line-train" points=train_points/>
                        })}
                        <polyline
                            class=if has_train { "ui-chart-line ui-chart-line-val" } else { "ui-chart-line ui-chart-line-metric" }
                            points=val_points
                        />
                        {train_dot.filter(|_| has_train).map(|(x, y)| view! {
                            <circle class="ui-chart-dot ui-chart-dot-train" cx=x cy=y r="2.5"/>
                        })}
                        {val_dot.map(|(x, y)| view! {
                            <circle
                                class=if has_train { "ui-chart-dot ui-chart-dot-val" } else { "ui-chart-dot ui-chart-dot-metric" }
                                cx=x cy=y r="2.5"
                            />
                        })}
                    </svg>
                    <table class="ui-chart-data">
                        <caption>{caption}</caption>
                        <thead>
                            <tr>
                                <th scope="col">{epoch_header}</th>
                                {has_train.then(|| view! { <th scope="col">{train_header}</th> })}
                                <th scope="col">{val_header}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.into_iter().map(|(e, t, v)| view! {
                                <tr>
                                    <th scope="row">{e.to_string()}</th>
                                    {has_train.then(|| view! { <td>{cell(t)}</td> })}
                                    <td>{cell(v)}</td>
                                </tr>
                            }).collect_view()}
                        </tbody>
                    </table>
                }.into_any()
            }}
            {has_train.then(|| view! {
                <footer class="ui-chart-legend">
                    <span class="ui-chart-key ui-chart-key-train">{train_label}</span>
                    <span class="ui-chart-key ui-chart-key-val">{val_label}</span>
                </footer>
            })}
        </section>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn history(v: serde_json::Value) -> Vec<EpochMetrics> {
        serde_json::from_value(v).unwrap()
    }

    #[wasm_bindgen_test]
    fn each_task_charts_its_own_metrics() {
        let keys = |t| charts_for(t).iter().map(|c| c.val).collect::<Vec<_>>();
        assert_eq!(
            keys(Task::Detect),
            [
                "metrics/mAP50-95(B)",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss"
            ]
        );
        assert_eq!(
            keys(Task::Segment),
            [
                "metrics/mAP50-95(M)",
                "val/box_loss",
                "val/cls_loss",
                "val/seg_loss"
            ]
        );
        assert_eq!(keys(Task::Classify), ["metrics/accuracy_top1", "val/loss"]);
        assert!(charts_for(Task::Face).is_empty());
        // The quality metric is a single line; every loss has train + val.
        for t in [Task::Detect, Task::Segment, Task::Classify] {
            let charts = charts_for(t);
            assert!(charts[0].train.is_none() && charts[0].kind.is_fraction());
            assert!(
                charts[1..]
                    .iter()
                    .all(|c| c.train.is_some() && !c.kind.is_fraction())
            );
        }
    }

    #[wasm_bindgen_test]
    fn a_job_without_history_still_deserializes() {
        let j: crate::api::TrainingJobStatus =
            serde_json::from_value(serde_json::json!({ "status": "training" })).unwrap();
        assert!(j.history.is_empty());
    }

    #[wasm_bindgen_test]
    fn series_follows_the_epochs_and_skips_missing_keys() {
        let h = history(serde_json::json!([
            { "epoch": 1, "train/box_loss": 1.5, "val/box_loss": 1.8 },
            { "epoch": 2, "train/box_loss": 1.2 },
            { "epoch": 3, "train/box_loss": 1.0, "val/box_loss": 1.4 },
        ]));
        assert_eq!(series(&h, "train/box_loss"), [(1, 1.5), (2, 1.2), (3, 1.0)]);
        assert_eq!(series(&h, "val/box_loss"), [(1, 1.8), (3, 1.4)]);
        assert!(has_data(&BOX, &h));
        assert!(!has_data(&DFL, &h));
    }

    #[wasm_bindgen_test]
    fn fractions_use_a_fixed_axis_and_losses_their_own_span() {
        let line = [(1, 0.2), (2, 0.4)];
        assert_eq!(y_range(&[&line], true), (0.0, 1.0));
        let (lo, hi) = y_range(&[&[(1, 2.0), (2, 1.0)], &[(1, 3.0)]], false);
        assert!((lo - 0.9).abs() < 1e-9 && (hi - 3.1).abs() < 1e-9);
        // A flat line still gets a visible band, and the floor never goes below 0.
        let (lo, hi) = y_range(&[&[(1, 0.5), (2, 0.5)]], false);
        assert!(lo < 0.5 && hi > 0.5 && lo >= 0.0);
        assert_eq!(y_range(&[&[]], false), (0.0, 1.0));
    }

    #[wasm_bindgen_test]
    fn points_span_the_whole_run_and_invert_the_y_axis() {
        let range = (0.0, 1.0);
        let (x1, y_top) = to_xy(1, 1.0, 10, range);
        let (x10, y_bottom) = to_xy(10, 0.0, 10, range);
        assert_eq!(x1, PLOT_X);
        assert!((x10 - (PLOT_X + PLOT_W)).abs() < 1e-9);
        assert_eq!(y_top, PLOT_Y);
        assert!((y_bottom - (PLOT_Y + PLOT_H)).abs() < 1e-9);
        // Halfway through a 50-epoch run the line ends mid-plot.
        let (x, _) = to_xy(25, 0.5, 49, range);
        assert!((x - (PLOT_X + PLOT_W / 2.0)).abs() < 1e-9);
        // A one-epoch run and out-of-range values stay inside the plot.
        let (x, y) = to_xy(1, 5.0, 1, range);
        assert_eq!((x, y), (PLOT_X, PLOT_Y));
    }

    #[wasm_bindgen_test]
    fn polyline_lists_one_point_per_epoch() {
        let pts = polyline_points(&[(1, 1.0), (2, 0.0)], 2, (0.0, 1.0));
        assert_eq!(
            pts,
            format!(
                "{:.1},{:.1} {:.1},{:.1}",
                PLOT_X,
                PLOT_Y,
                PLOT_X + PLOT_W,
                PLOT_Y + PLOT_H
            )
        );
        assert_eq!(polyline_points(&[], 10, (0.0, 1.0)), "");
    }

    #[wasm_bindgen_test]
    fn the_data_table_lists_every_epoch_of_either_line() {
        let train = [(1, 1.5), (2, 1.2), (3, 1.0)];
        let val = [(1, 1.8), (3, 1.4), (4, 1.3)];
        assert_eq!(
            data_rows(&train, &val),
            [
                (1, Some(1.5), Some(1.8)),
                (2, Some(1.2), None),
                (3, Some(1.0), Some(1.4)),
                (4, None, Some(1.3)),
            ]
        );
        assert_eq!(data_rows(&[], &[(1, 0.4)]), [(1, None, Some(0.4))]);
        assert!(data_rows(&[], &[]).is_empty());
    }

    #[wasm_bindgen_test]
    fn values_print_with_enough_precision() {
        assert_eq!(format_value(0.51234), "0.512");
        assert_eq!(format_value(1.5), "1.500");
        assert_eq!(format_value(12.345), "12.3");
    }
}
