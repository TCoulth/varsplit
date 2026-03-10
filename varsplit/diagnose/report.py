"""
Report assembly for varsplit.diagnose.
Combines metrics, plots, and recommendation into HTML or PDF output.
"""

import io
import base64
from datetime import datetime
from matplotlib.figure import Figure


def _fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _img_tag(fig: Figure, caption: str = "") -> str:
    b64 = _fig_to_base64(fig)
    cap = f'<p class="caption">{caption}</p>' if caption else ""
    return f'<div class="figure"><img src="data:image/png;base64,{b64}">{cap}</div>'


def _metric_table(rows: list[tuple]) -> str:
    html = '<table class="metrics">'
    for label, value in rows:
        html += f"<tr><td>{label}</td><td><strong>{value}</strong></td></tr>"
    html += "</table>"
    return html


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 960px; margin: 40px auto; padding: 0 24px; color: #222; }
h1   { font-size: 1.6em; border-bottom: 2px solid #4878CF; padding-bottom: 8px; }
h2   { font-size: 1.2em; color: #4878CF; margin-top: 36px; }
h3   { font-size: 1em; color: #555; margin-top: 16px; }
.rec-box { background: #f0f4ff; border-left: 4px solid #4878CF;
           padding: 16px 20px; border-radius: 4px; margin: 16px 0; }
.rec-box .strategy { font-size: 1.4em; font-weight: bold; color: #4878CF; }
.dim-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin-top: 12px; }
.dim-table th { background: #f5f5f5; padding: 8px 12px; text-align: left;
                font-weight: 600; }
.dim-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
.bar-cell  { width: 160px; }
.bar-bg    { background: #eee; border-radius: 4px; height: 12px; }
.bar-fill  { background: #4878CF; border-radius: 4px; height: 12px; }
.bar-fill.primary { background: #E8762B; }
.primary-tag { color: #E8762B; font-weight: bold; font-size: 0.85em; }
.metrics { border-collapse: collapse; font-size: 0.9em; margin: 8px 0 16px 0; }
.metrics td { padding: 5px 20px 5px 0; vertical-align: top; }
.figure  { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
.caption { color: #888; font-size: 0.85em; margin-top: 6px; }
.summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
footer { margin-top: 48px; color: #aaa; font-size: 0.8em;
         border-top: 1px solid #eee; padding-top: 12px; }
"""


def build_html_report(
    summary: dict,
    positional: dict,
    mutational: dict,
    order: dict,
    fitness: dict | None,
    recommendation,
    figures: dict,
    label_A: str = "train",
    label_B: str = "test",
    preset: str = "full",
) -> str:
    _DEVIATION_THRESHOLD = 0.05

    def should_include(dim_name: str) -> bool:
        if preset == "full":
            return True
        if dim_name not in recommendation.dimensions:
            return False
        return recommendation.dimensions[dim_name].novelty >= _DEVIATION_THRESHOLD

    # --- Recommendation box ---
    dims = recommendation.dimensions
    dim_rows = ""
    for name, dim in sorted(dims.items(), key=lambda x: -x[1].novelty):
        is_primary = name == recommendation.recommended_strategy
        pct = int(dim.novelty * 100)
        fill_class = "bar-fill primary" if is_primary else "bar-fill"
        tag = '<span class="primary-tag">PRIMARY</span>' if is_primary else ""
        dim_rows += f"""
        <tr>
          <td>{name.capitalize()}</td>
          <td>{dim.novelty:.0%}</td>
          <td class="bar-cell">
            <div class="bar-bg">
              <div class="{fill_class}" style="width:{pct}%"></div>
            </div>
          </td>
          <td>{dim.note}</td>
          <td>{tag}</td>
        </tr>"""

    rec_html = f"""
    <div class="rec-box">
      <div><span class="strategy">{recommendation.recommended_strategy.upper()}</span></div>
      <p style="margin:8px 0 12px 0;">{recommendation.rationale}</p>
      <table class="dim-table">
        <thead><tr>
          <th>Dimension</th><th>Novelty</th><th></th><th>Note</th><th></th>
        </tr></thead>
        <tbody>{dim_rows}</tbody>
      </table>
      <p style="margin-top:14px;font-size:0.9em;">
        Suggested within-<em>{label_A}</em> split:
        <code>VarSplit(strategy="{recommendation.recommended_strategy}")</code>
      </p>
    </div>"""

    # --- Dataset summary ---
    summary_html = f"""
    <div class="summary-grid">
      <div>
        <h3>{label_A.capitalize()}</h3>
        {_metric_table([
            ("Variants", summary["n_variants_A"]),
            ("Unique positions", positional["n_positions_A"]),
            ("Unique mutations", mutational["n_subs_A"]),
            ("Mean mutations per variant",
             f"{order['mean_A']:.2f} ± {order['std_A']:.2f}"),
        ])}
      </div>
      <div>
        <h3>{label_B.capitalize()}</h3>
        {_metric_table([
            ("Variants", summary["n_variants_B"]),
            ("Unique positions", positional["n_positions_B"]),
            ("Unique mutations", mutational["n_subs_B"]),
            ("Mean mutations per variant",
             f"{order['mean_B']:.2f} ± {order['std_B']:.2f}"),
        ])}
      </div>
    </div>"""

    # --- Sections ---
    sections_html = ""

    # Mutations per variant
    if should_include("order") and "order" in figures:
        shift = order["mean_shift"]
        ratio = order["mean_B"] / order["mean_A"] if order["mean_A"] > 0 else float("inf")
        sections_html += f"<h2>Mutations per Variant</h2>"
        sections_html += _metric_table([
            (f"Mean ({label_A})",
             f"{order['mean_A']:.2f} ± {order['std_A']:.2f}"),
            (f"Mean ({label_B})",
             f"{order['mean_B']:.2f} ± {order['std_B']:.2f}"),
            ("Mean shift (B − A)",
             f"{shift:+.2f}  ({ratio:.1f}×)"),
        ])
        sections_html += _img_tag(
            figures["order"],
            "Distribution of mutations per variant in each set."
        )

    # Positional overlap
    if should_include("positional") and "positional" in figures:
        sections_html += f"<h2>Positional Overlap</h2>"
        sections_html += _metric_table([
            (f"Positions in {label_A}", positional["n_positions_A"]),
            (f"Positions in {label_B}", positional["n_positions_B"]),
            ("Shared", positional["n_shared"]),
            (f"New positions in {label_B}",
             f"{positional['n_only_B']} ({positional['pct_B_novel']:.0%})"),
        ])
        sections_html += _img_tag(
            figures["positional"],
            f"Positions shared between sets vs. unique to each."
        )

    # Mutational novelty
    if should_include("mutational") and "mutational" in figures:
        sections_html += f"<h2>Mutational Novelty</h2>"
        sections_html += _metric_table([
            (f"Mutations in {label_A}", mutational["n_subs_A"]),
            (f"Mutations in {label_B}", mutational["n_subs_B"]),
            ("Shared", mutational["n_shared"]),
            (f"New mutations in {label_B}",
             f"{mutational['n_only_B']} ({mutational['pct_B_novel']:.0%})"),
        ])
        sections_html += _img_tag(
            figures["mutational"],
            f"Mutations shared between sets vs. unique to each."
        )

    # Fitness (only if provided)
    if fitness is not None and should_include("fitness") and "fitness" in figures:
        sections_html += f"<h2>Fitness Distribution</h2>"
        sections_html += _metric_table([
            (f"Mean score ({label_A})",
             f"{fitness['mean_A']:.3f} ± {fitness['std_A']:.3f}"),
            (f"Mean score ({label_B})",
             f"{fitness['mean_B']:.3f} ± {fitness['std_B']:.3f}"),
            (f"Score range ({label_A})",
             f"[{fitness['min_A']:.3f}, {fitness['max_A']:.3f}]"),
            (f"Score range ({label_B})",
             f"[{fitness['min_B']:.3f}, {fitness['max_B']:.3f}]"),
            (f"{label_B} outside {label_A} range",
             f"{fitness['pct_B_extrapolates']:.0%}"),
        ])
        sections_html += _img_tag(
            figures["fitness"],
            f"Dashed lines show {label_A} score range. "
            f"Shaded regions indicate extrapolation."
        )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>varsplit diagnose</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>varsplit diagnose</h1>
  <p style="color:#888;font-size:0.9em;margin-top:-8px;">
    {label_A.capitalize()} vs {label_B.capitalize()}
  </p>

  <h2>Recommendation</h2>
  {rec_html}

  <h2>Dataset Summary</h2>
  {summary_html}

  {sections_html}

  <footer>Generated by varsplit &middot; {now}</footer>
</body>
</html>"""


def save_report(html: str, output_path: str) -> None:
    if output_path.endswith(".pdf"):
        try:
            from weasyprint import HTML as WP_HTML
            WP_HTML(string=html).write_pdf(output_path)
        except ImportError:
            raise ImportError(
                "PDF output requires weasyprint: pip install weasyprint. "
                "Alternatively, save as .html and print to PDF from your browser."
            )
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
