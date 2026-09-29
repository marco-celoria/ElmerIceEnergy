import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Config ----
results_dir = os.path.expanduser("~/Work/my_ElmerIceEnergy/results/roihu")
output_dir = os.path.expanduser("~/Work/my_ElmerIceEnergy/results/roihu")
mesh_level_to_plot = 3
expected_timesteps = 20  # see inputs/Greenland_SSA/SSA_amgx_ML{n}.sif: "Timestep Intervals(1) = 20"
ranks_to_exclude = [64,128]  # NTASKS_PER_NODE values to drop from the plot

if not os.path.isdir(results_dir):
    raise SystemExit(f"results_dir does not exist: {results_dir}")

Path(output_dir).mkdir(parents=True, exist_ok=True)

# run_Elmer_roihu_N<nodes>_n<ntasks_per_node>_c<cpus_per_task>_ML<mesh_level>[_MPS]_<jobid>
pattern = re.compile(
    r"run_Elmer_roihu_"
    r"N(?P<nodes>\d+)_"
    r"n(?P<ntasks_per_node>\d+)_"
    r"c(?P<cpus_per_task>\d+)_"
    r"ML(?P<mesh_level>\d+)"
    r"(?:_(?P<mps>MPS))?"
    r"_(?P<jobid>\d+)"
)

records = []
skipped = []

for root, dirs, files in os.walk(results_dir):
    if "Scalars_gpu.dat" not in files:
        continue

    match = pattern.search(root)
    if not match:
        continue
    meta = match.groupdict()

    file_path = os.path.join(root, "Scalars_gpu.dat")
    with open(file_path) as f:
        lines = [l for l in f.readlines() if l.strip()]

    label = (
        f"N{meta['nodes']}_n{meta['ntasks_per_node']}_c{meta['cpus_per_task']}"
        f"_ML{meta['mesh_level']}{'_MPS' if meta['mps'] else ''}_{meta['jobid']}"
    )

    if len(lines) != expected_timesteps:
        skipped.append((label, len(lines)))
        continue

    # Columns: time, min dhdt, min ssavelocity, max ssavelocity, cpu time (s), real time (s)
    last_row = lines[-1].split()
    cpu_time = float(last_row[4])

    records.append({
        "NODES": int(meta["nodes"]),
        "NTASKS_PER_NODE": int(meta["ntasks_per_node"]),
        "CPUS_PER_TASK": int(meta["cpus_per_task"]),
        "MESH_LEVEL": int(meta["mesh_level"]),
        "MPS": meta["mps"] == "MPS",
        "JOBID": int(meta["jobid"]),
        "CPU_TIME": cpu_time,
        "SOURCE": root,
    })

if skipped:
    print(f"Skipping {len(skipped)} run(s) with an incomplete Scalars_gpu.dat "
          f"(expected {expected_timesteps} timesteps):")
    for label, n in skipped:
        print(f"  - {label}: only {n}/{expected_timesteps} lines "
              f"-> the run likely did not finish within the job's time limit")

if not records:
    raise SystemExit(
        f"No matching runs with a Scalars_gpu.dat found under {results_dir}. "
        "Check that the results have been synced there and that folder names "
        "match the expected 'run_Elmer_roihu_N.._n.._c.._ML..[_MPS]_<jobid>' pattern."
    )

df = pd.DataFrame(records)

# Keep only "full node" runs: CPUS_PER_TASK == 1 is a separate pure-MPI sweep
# (one core per task) rather than the full-node OpenMP x MPI sweep we want here.
full_node = df[(df["CPUS_PER_TASK"] != 1) & (~df["NTASKS_PER_NODE"].isin(ranks_to_exclude))]

strong_scaling = full_node[full_node["MESH_LEVEL"] == mesh_level_to_plot].sort_values(
    by=["NTASKS_PER_NODE", "MPS"]
)

if strong_scaling.empty:
    print(f"No complete full-node runs found for mesh level {mesh_level_to_plot}; nothing to plot.")
else:
    # Collapse repeated runs of the same test case down to a single CPU_TIME.
    # Repeats normally agree closely; only warn (and fall back to their average)
    # when they disagree by more than `divergence_threshold`.
    divergence_threshold = 0.05  # 5% relative spread between the fastest and slowest repeat

    group_cols = ["NODES", "NTASKS_PER_NODE", "CPUS_PER_TASK", "MPS"]
    collapsed_rows = []
    for key, group in strong_scaling.groupby(group_cols):
        times = group["CPU_TIME"].to_numpy()
        mean_time = times.mean()
        if len(times) > 1:
            spread = (times.max() - times.min()) / mean_time
            if spread > divergence_threshold:
                nodes, ntasks, cpus, mps = key
                print(
                    f"Warning: repeated runs for NODES={nodes}, NTASKS_PER_NODE={ntasks}, "
                    f"CPUS_PER_TASK={cpus}, MPS={mps} disagree by {spread:.1%} "
                    f"(CPU times: {times.tolist()}, jobids: {group['JOBID'].tolist()}) "
                    "-> plotting their average."
                )
        collapsed_rows.append({**dict(zip(group_cols, key)), "CPU_TIME": mean_time})

    strong_scaling = pd.DataFrame(collapsed_rows)

    # ---- Parameters ----
    # With CPUS_PER_TASK == 1 excluded, each NTASKS_PER_NODE value maps to exactly
    # one CPUS_PER_TASK (they multiply out to the same full-node core count), so
    # the grouping collapses to "main group = MPI ranks/node, subgroup = MPS
    # on/off" -- the same shape as the Leonardo strong-scaling plot, where the
    # main group is NODES and the subgroup is MPS on/off.
    groups_lvl1 = sorted(strong_scaling["NTASKS_PER_NODE"].unique().tolist())
    mps_vals = [False, True]

    cpus_per_group = {
        ntasks: strong_scaling.loc[strong_scaling["NTASKS_PER_NODE"] == ntasks, "CPUS_PER_TASK"].iloc[0]
        for ntasks in groups_lvl1
    }

    bar_width = 0.3
    group_width = bar_width * len(mps_vals)
    x_main = np.arange(len(groups_lvl1)) * (group_width + 0.5)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- Plot ----
    for i, ntasks in enumerate(groups_lvl1):
        bars_this_group = {}
        for k, mps in enumerate(mps_vals):
            subset = strong_scaling[
                (strong_scaling["NTASKS_PER_NODE"] == ntasks) &
                (strong_scaling["MPS"] == mps)
            ]

            if subset.empty:
                continue

            y = subset["CPU_TIME"].values[0]
            x = x_main[i] + k * bar_width

            ax.bar(
                x,
                y,
                width=bar_width,
                color=f"C{i}",
                hatch="//" if mps else "",
                label=f"MPI ranks={ntasks}, MPS={'On' if mps else 'Off'}"
            )
            bars_this_group[mps] = (x, y)

        # ---- Speedup annotation (MPS off -> MPS on) ----
        if False in bars_this_group and True in bars_this_group:
            x_off, y_off = bars_this_group[False]
            x_on, y_on = bars_this_group[True]
            speedup = y_off / y_on
            x_mid = (x_off + x_on) / 2 + bar_width / 2
            y_text = max(y_off, y_on) * 1.05
            ax.text(
                x_mid, y_text, f"{speedup:.2f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # ---- X ticks (centered on main groups) ----
    ax.set_xticks(x_main + group_width / 2 - bar_width / 2)
    ax.set_xticklabels([
        f"MPI ranks={g}\nCPUs/task={cpus_per_group[g]}" for g in groups_lvl1
    ])

    ax.set_yscale("log")
    ax.set_ylabel("CPU Time [s]")
    ax.set_title(
        f"Elmer/Ice Greenland SSA (Roihu, Mesh Level {mesh_level_to_plot}): "
        "Execution Time Breakdown"
    )

    # ---- Clean legend (avoid duplicates) ----
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=1)

    plt.tight_layout()
    outfile = os.path.join(output_dir, f"strong_scaling_cputime_fullnode_ml{mesh_level_to_plot}.png")
    plt.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")
