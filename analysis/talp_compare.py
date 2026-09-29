#!/usr/bin/env python3
"""Compare multiple DLB/TALP 3.8 JSON reports.

Plotly creates a self-contained interactive HTML report. Matplotlib creates
static PNG figures. Values are always read from the input JSON files.
MPS mode is inferred from "MPS" in each filename.
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

EFF = {
    "parallelEfficiency": "Parallel efficiency",
    "mpiParallelEfficiency": "MPI parallel efficiency",
    "mpiCommunicationEfficiency": "MPI communication efficiency",
    "mpiLoadBalance": "MPI load balance",
    "deviceOffloadEfficiency": "Device offload efficiency",
    "gpuParallelEfficiency": "GPU parallel efficiency",
    "gpuLoadBalance": "GPU load balance",
    "gpuCommunicationEfficiency": "GPU communication efficiency",
    "gpuOrchestrationEfficiency": "GPU orchestration efficiency",
}
TIMES = {
    "elapsedTime": "Elapsed time (s)",
    "usefulTime": "Useful time (s)",
    "mpiTime": "MPI time (s)",
    "gpuRuntimeTime": "GPU runtime time (s)",
    "gpuUsefulTime": "GPU useful time (s)",
    "gpuCommunicationTime": "GPU communication time (s)",
}
OVERVIEW = ["Parallel efficiency", "MPI parallel efficiency",
            "Device offload efficiency", "GPU parallel efficiency"]
MPI_METRICS = ["MPI parallel efficiency", "MPI communication efficiency",
               "MPI load balance"]
GPU_METRICS = ["Device offload efficiency", "GPU parallel efficiency",
               "GPU load balance", "GPU communication efficiency",
               "GPU orchestration efficiency"]


def discover(items):
    found = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            found.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            found.append(p)
        else:
            found.extend(Path(x) for x in sorted(glob.glob(item)))
    answer, seen = [], set()
    for p in found:
        rp = p.resolve()
        if p.suffix.lower() == ".json" and rp not in seen:
            seen.add(rp)
            answer.append(p)
    return answer


def read_report(path):
    with path.open(encoding="utf-8") as f:
        doc = json.load(f)
    try:
        g = doc["Application"]["Global"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing Application.Global") from exc
    resources = doc.get("resources", {})
    ranks = g.get("numMpiRanks", resources.get("numMpiRanks"))
    if ranks is None:
        raise ValueError(f"{path}: missing numMpiRanks")
    mode = "MPS" if "mps" in path.stem.casefold() else "Standard"
    row = {
        "Configuration": f"{int(ranks)} MPS" if mode == "MPS" else str(int(ranks)),
        "Execution mode": mode,
        "MPI ranks": int(ranks),
        "Nodes": g.get("numNodes", resources.get("numNodes")),
        "CPUs": g.get("numCpus", resources.get("numCpus")),
        "Available CPUs": resources.get("numAvailableCpus"),
        "GPUs": g.get("numGpus", resources.get("numGpus")),
        "DLB version": doc.get("dlbVersion", ""),
        "Timestamp": doc.get("timestamp", ""),
        "Source file": str(path),
    }
    for key, name in TIMES.items():
        value = g.get(key)
        row[name] = float(value) / 1e9 if value is not None else np.nan
    for key, name in EFF.items():
        value = g.get(key)
        row[name] = float(value) if value is not None else np.nan
    return row


def load(files):
    rows = []
    for path in files:
        try:
            rows.append(read_report(path))
            print(f"Loaded: {path}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: {exc}", file=sys.stderr)
    if not rows:
        raise RuntimeError("No valid reports loaded")
    df = pd.DataFrame(rows)
    df["_mode"] = df["Execution mode"].map({"Standard": 0, "MPS": 1})
    df = df.sort_values(["MPI ranks", "_mode", "Timestamp", "Source file"],
                        kind="stable").drop(columns="_mode").reset_index(drop=True)
    # Preserve duplicate runs without losing mathematical ordering.
    occurrence = df.groupby("Configuration").cumcount() + 1
    total = df.groupby("Configuration")["Configuration"].transform("size")
    df.loc[total > 1, "Configuration"] += " #" + occurrence[total > 1].astype(str)
    return df


def existing(df, columns):
    return [c for c in columns if c in df and df[c].notna().any()]


def pbar(df, columns, title, percent=True):
    columns = existing(df, columns)
    d = df.melt(id_vars="Configuration", value_vars=columns,
                var_name="Metric", value_name="Value").dropna()
    fig = px.bar(d, x="Configuration", y="Value", color="Metric",
                 barmode="group", title=title,
                 category_orders={"Configuration": df["Configuration"].tolist()},
                 text_auto=".1%" if percent else ".3s")
    fig.update_xaxes(title="MPI ranks / execution mode", type="category")
    fig.update_yaxes(title="Efficiency" if percent else "Seconds",
                     range=[0, 1.05] if percent else None,
                     tickformat=".0%" if percent else None)
    fig.update_layout(template="plotly_white", height=620,
                      margin=dict(l=60, r=30, t=80, b=80), legend_title_text="")
    return fig


def pheatmap(df):
    cols = existing(df, list(EFF.values()))
    z = df.set_index("Configuration")[cols].T
    fig = px.imshow(z, zmin=0, zmax=1, aspect="auto", text_auto=".0%",
                    color_continuous_scale="RdYlGn", title="Efficiency heatmap",
                    labels={"x": "MPI ranks / execution mode", "y": "Metric",
                            "color": "Efficiency"})
    fig.update_layout(template="plotly_white", height=700)
    return fig


def radar_columns(df):
    return existing(df, OVERVIEW + ["MPI communication efficiency", "MPI load balance",
                                    "GPU load balance", "GPU communication efficiency",
                                    "GPU orchestration efficiency"])


def pradar(df):
    cols = radar_columns(df)
    fig = go.Figure()
    for _, row in df.iterrows():
        vals = [row[c] for c in cols]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cols + cols[:1],
                                     fill="toself", name=row["Configuration"]))
    fig.update_layout(title="Efficiency radar", template="plotly_white", height=750,
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                                tickformat=".0%")))
    return fig


def grouped_png(df, columns, title, ylabel, path, dpi, percent=False):
    cols = existing(df, columns)
    x = np.arange(len(df))
    width = 0.82 / len(cols)
    fig, ax = plt.subplots(figsize=(max(11, len(df) * 1.25), 7))
    for i, col in enumerate(cols):
        offset = (i - (len(cols) - 1) / 2) * width
        bars = ax.bar(x + offset, df[col], width, label=col)
        if len(cols) <= 4:
            labels = [f"{v:.0%}" if percent else f"{v:.2f}" for v in df[col]]
            ax.bar_label(bars, labels=labels, padding=2, fontsize=8, rotation=90)
    ax.set(title=title, xlabel="MPI ranks / execution mode", ylabel=ylabel)
    ax.set_xticks(x, df["Configuration"])
    if percent:
        ax.set_ylim(0, 1.08)
        ticks = np.linspace(0, 1, 6)
        ax.set_yticks(ticks, [f"{v:.0%}" for v in ticks])
    ax.grid(axis="y", alpha=.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {path}")


def heatmap_png(df, path, dpi):
    cols = existing(df, list(EFF.values()))
    matrix = df[cols].to_numpy().T
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.15), max(6, len(cols) * .55)))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(df)), df["Configuration"])
    ax.set_yticks(range(len(cols)), cols)
    ax.set_title("Efficiency heatmap")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if not np.isnan(matrix[r, c]):
                ax.text(c, r, f"{matrix[r,c]:.0%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Efficiency")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {path}")


def radar_png(df, path, dpi):
    cols = radar_columns(df)
    angles = np.linspace(0, 2 * math.pi, len(cols), endpoint=False).tolist()
    closed = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"polar": True})
    for _, row in df.iterrows():
        vals = [float(row[c]) for c in cols]
        ax.plot(closed, vals + vals[:1], linewidth=1.8, label=row["Configuration"])
        ax.fill(closed, vals + vals[:1], alpha=.04)
    ax.set_xticks(angles, cols, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([.2, .4, .6, .8, 1], ["20%", "40%", "60%", "80%", "100%"])
    ax.set_title("Efficiency radar", pad=28)
    ax.legend(bbox_to_anchor=(1.25, 1.1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {path}")


def table_html(df):
    table = df.copy()
    for col in existing(table, list(TIMES.values())):
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    for col in existing(table, list(EFF.values())):
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.1%}")
    return table.to_html(index=False, border=0, classes="summary", escape=True)


def write_html(df, figures, path):
    fragments = [fig.to_html(full_html=False, include_plotlyjs=(i == 0),
                             config={"responsive": True, "displaylogo": False})
                 for i, fig in enumerate(figures)]
    cards = "".join('<section class="card">' + x + '</section>' for x in fragments)
    template = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DLB/TALP comparison</title><style>
body{margin:0;padding:2rem;font-family:Arial,sans-serif;background:#f4f6f8;color:#1f2937}
main{max-width:1500px;margin:auto}.card{background:white;border-radius:12px;padding:1rem;
margin:1.2rem 0;box-shadow:0 2px 10px #00000014}.table-wrap{overflow-x:auto}
.summary{border-collapse:collapse;width:100%;font-size:.84rem}.summary th,.summary td{
border-bottom:1px solid #e5e7eb;padding:.5rem;text-align:right;white-space:nowrap}
.summary th{background:#eef2f7}.summary th:first-child,.summary td:first-child{text-align:left}
</style></head><body><main><h1>DLB/TALP comparison report</h1>
<p>RUNS report(s), ordered numerically with Standard before MPS.</p>
<section class="card"><h2>Summary</h2><div class="table-wrap">TABLE</div></section>
CHARTS</main></body></html>"""
    template = template.replace("RUNS", str(len(df))).replace("TABLE", table_html(df))
    path.write_text(template.replace("CHARTS", cards), encoding="utf-8")
    print(f"HTML: {path}")


def main():
    parser = argparse.ArgumentParser(description="DLB/TALP JSON comparison report")
    parser.add_argument("inputs", nargs="+", help="Files, directory, or quoted glob")
    parser.add_argument("-o", "--output-dir", default="talp_output")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    files = discover(args.inputs)
    if not files:
        parser.error("no JSON files found")
    try:
        df = load(files)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "talp_summary.csv", index=False)
        df[["Configuration", "Execution mode", "MPI ranks"] + existing(df, list(EFF.values()))].to_csv(
            out / "talp_efficiencies.csv", index=False)
        df[["Configuration", "Execution mode", "MPI ranks"] + existing(df, list(TIMES.values()))].to_csv(
            out / "talp_timings.csv", index=False)

        figures = [pbar(df, ["Elapsed time (s)"], "Elapsed time", False),
                   pbar(df, OVERVIEW, "Efficiency overview"),
                   pbar(df, MPI_METRICS, "MPI efficiencies"),
                   pbar(df, GPU_METRICS, "GPU efficiencies"),
                   pheatmap(df), pradar(df)]
        write_html(df, figures, out / "talp_report.html")

        grouped_png(df, ["Elapsed time (s)"], "Elapsed time", "Seconds",
                    out / "elapsed_time.png", args.dpi)
        grouped_png(df, OVERVIEW, "Efficiency overview", "Efficiency",
                    out / "efficiency_overview.png", args.dpi, True)
        grouped_png(df, MPI_METRICS, "MPI efficiencies", "Efficiency",
                    out / "mpi_efficiencies.png", args.dpi, True)
        grouped_png(df, GPU_METRICS, "GPU efficiencies", "Efficiency",
                    out / "gpu_efficiencies.png", args.dpi, True)
        heatmap_png(df, out / "efficiency_heatmap.png", args.dpi)
        radar_png(df, out / "efficiency_radar.png", args.dpi)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Done: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
