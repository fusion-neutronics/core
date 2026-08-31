//! Shared helpers for building self-contained HTML for Jupyter `_repr_html_`.
//!
//! The core data objects (`Material`, `SimulationResults`, `TallyResult`,
//! `StatisticalChecks`, and the probability distributions) render rich tables
//! in notebooks via these helpers. Output uses only inline `style="..."`
//! attributes -- no `<style>` blocks or external CSS -- so it renders
//! identically in Jupyter, JupyterLab, nbconvert and the VS Code notebook
//! viewer, none of which is guaranteed to preserve a `<style>` element in
//! cell output.

const FONT: &str =
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif";

/// Escape the HTML-significant characters in untrusted text.
pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Compact numeric formatting: trimmed fixed-point for human-scale
/// magnitudes, scientific for very large / very small, exact for zero and
/// non-finite values.
pub fn num(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let a = x.abs();
    if (1e-3..1e7).contains(&a) {
        let s = format!("{x:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        format!("{x:.4e}")
    }
}

/// A PASS / FAIL / n/a pill. `None` renders as a grey "n/a".
pub fn badge(ok: Option<bool>) -> String {
    let (label, bg) = match ok {
        Some(true) => ("PASS", "#1a7f37"),
        Some(false) => ("FAIL", "#cf222e"),
        None => ("n/a", "#8c959f"),
    };
    format!(
        "<span style=\"display:inline-block;padding:1px 7px;border-radius:9px;\
         font-size:11px;font-weight:600;color:#ffffff;background:{bg};\">{label}</span>"
    )
}

/// Wrap inner HTML (`body`, already-formed HTML) in a titled card. `title`
/// and `subtitle` are plain text and are escaped here; `subtitle` may be
/// empty.
pub fn card(title: &str, subtitle: &str, body: &str) -> String {
    let sub = if subtitle.is_empty() {
        String::new()
    } else {
        format!(
            "<span style=\"color:#656d76;font-weight:400;font-size:12px;\"> &middot; {}</span>",
            esc(subtitle)
        )
    };
    format!(
        "<div style=\"{FONT};display:inline-block;border:1px solid #d0d7de;\
         border-radius:6px;overflow:hidden;margin:2px 0;font-size:13px;\
         color:#1f2328;background:#ffffff;vertical-align:top;\">\
           <div style=\"background:#f6f8fa;border-bottom:1px solid #d0d7de;\
            padding:5px 10px;font-weight:600;\">{title}{sub}</div>\
           <div style=\"padding:8px 10px;\">{body}</div>\
         </div>",
        title = esc(title)
    )
}

/// Build a table. Columns at index >= `numeric_from` are right-aligned.
/// Header text is escaped here; row cells are treated as ready HTML (so they
/// may contain badges) -- escape any plain-text cells with [`esc`] at the
/// call site.
pub fn table(headers: &[&str], rows: &[Vec<String>], numeric_from: usize) -> String {
    let mut head = String::new();
    for (i, h) in headers.iter().enumerate() {
        let align = if i >= numeric_from { "right" } else { "left" };
        head.push_str(&format!(
            "<th style=\"text-align:{align};padding:3px 12px 3px 0;\
             border-bottom:1px solid #d0d7de;color:#656d76;font-weight:600;\
             font-size:12px;white-space:nowrap;\">{}</th>",
            esc(h)
        ));
    }
    let mut body = String::new();
    for row in rows {
        body.push_str("<tr>");
        for (i, cell) in row.iter().enumerate() {
            let align = if i >= numeric_from { "right" } else { "left" };
            body.push_str(&format!(
                "<td style=\"text-align:{align};padding:3px 12px 3px 0;\
                 border-bottom:1px solid #eaeef2;white-space:nowrap;\
                 font-variant-numeric:tabular-nums;\">{cell}</td>"
            ));
        }
        body.push_str("</tr>");
    }
    format!(
        "<table style=\"border-collapse:collapse;\"><thead><tr>{head}</tr>\
         </thead><tbody>{body}</tbody></table>"
    )
}

/// A two-column key / value table. Keys (plain text) are escaped here; values
/// are treated as ready HTML -- escape plain-text values with [`esc`] at the
/// call site.
pub fn kv(rows: &[(&str, String)]) -> String {
    let mut body = String::new();
    for (k, v) in rows {
        body.push_str(&format!(
            "<tr><td style=\"padding:2px 12px 2px 0;color:#656d76;\
             white-space:nowrap;\">{}</td>\
             <td style=\"padding:2px 0;font-variant-numeric:tabular-nums;\">{}</td></tr>",
            esc(k),
            v
        ));
    }
    format!("<table style=\"border-collapse:collapse;\"><tbody>{body}</tbody></table>")
}
