"""Paper figures from saved runs (no new runs).

    python analysis/make_paper_figures.py [ROOT] [OUT_DIR] [--tag paper]

ROOT: the transport_mh folder (default: this script's parent). For each problem, the
script reads runs.csv of the listed result folders, the reference value, and writes
    <OUT_DIR>/fig_<problem>_error_vs_time.pdf   relative RMSE vs mean wall time per run, over horizons L
    <OUT_DIR>/fig_<problem>_level_sd.pdf        per-run s.d. of each level's correction at the finest horizon
    <OUT_DIR>/table_<problem>.csv               the numbers behind both figures
Problems: beskos (1D elliptic), darcy (2D Darcy flow), ns2d (2D Navier--Stokes; needs ref_ns/reference.json).
Folders are given per method; set --tag paper to use the 30-repeat results_paper_* folders.
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv.field_size_limit(10**9)
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ROOT / "paper_figures"
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "current"

# Method styling: fixed categorical order (validated palette, light mode), plus marker and
# line style so identity never relies on colour alone (print, colour-vision deficiency).
STYLE = {
    "TransportMH":        dict(color="#2a78d6", marker="o", ls="-",  label="TransportMH"),
    "pCN + ESS sweeps":   dict(color="#eb6834", marker="s", ls="--", label="MLSMC-pCN + ESS sweeps"),
    "MLSMC-pCN":          dict(color="#1baf7a", marker="^", ls="-.", label="MLSMC-pCN"),
    "LightSB":            dict(color="#eda100", marker="D", ls=":",  label="LightSB (no MH)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

if TAG == "paper":
    PROBLEMS = {
        "darcy": dict(ref="ref_darcy/reference.json", finest="L6", title="2D Darcy flow", sources={
            "TransportMH": [("results_paper_darcy_L5", "TransportMH"), ("results_paper_darcy_L6", "TransportMH")],
            "pCN + ESS sweeps": [("results_paper_darcy_L5", "MLSMC_pCN"), ("results_paper_darcy_L6", "MLSMC_pCN")],
            "MLSMC-pCN": [("results_paper_darcy_L5_pcn_plain", "MLSMC_pCN"), ("results_paper_darcy_L6_pcn_plain", "MLSMC_pCN")],
            "LightSB": [("results_paper_darcy_L5", "LightSB"), ("results_paper_darcy_L6", "LightSB")]}),
        "beskos": dict(ref="ref_is_noise005/reference.json", finest="L9", title=r"1D elliptic (Beskos), $\sigma_\eta=0.05$", sources={
            "TransportMH": [("results_paper_T1_noise005", "TransportMH")],
            "pCN + ESS sweeps": [("results_paper_T1_noise005", "MLSMC_pCN")],
            "MLSMC-pCN": [("results_paper_T1_noise005_pcn_plain", "MLSMC_pCN")],
            "LightSB": [("results_paper_T1_noise005", "LightSB")]}),
    }
else:   # the 10-repeat runs available now (draft styling)
    PROBLEMS = {
        "darcy": dict(ref="ref_darcy/reference.json", finest="L6", title="2D Darcy flow", sources={
            "TransportMH": [("results_darcy_L5", "TransportMH"), ("results_darcy_L6", "TransportMH")],
            "pCN + ESS sweeps": [("results_darcy_L5", "MLSMC_pCN"), ("results_darcy_L6", "MLSMC_pCN")],
            "MLSMC-pCN": [("results_darcy_L5_pcn_plain", "MLSMC_pCN"), ("results_darcy_L6_pcn_plain", "MLSMC_pCN")],
            "LightSB": [("results_darcy_L5", "LightSB"), ("results_darcy_L6", "LightSB")]}),
        "beskos": dict(ref="ref_is_noise005/reference.json", finest="L9", title=r"1D elliptic (Beskos), $\sigma_\eta=0.05$", sources={
            "TransportMH": [("results_L9_noise005_itlev300", "TransportMH")],
            "pCN + ESS sweeps": [("results_L9_noise005_pcn_esssw30", "MLSMC_pCN")],
            "MLSMC-pCN": [("results_L9_noise005", "MLSMC_pCN")],
            "LightSB": [("results_L9_noise005_itlev300", "LightSB")]}),
    }

# Navier--Stokes: one set of 30-repeat runs (horizons L = 1..4 inside the L4 config), same for both tags
PROBLEMS["ns2d"] = dict(ref="ref_ns/reference.json", finest="L4", title="2D Navier\u2013Stokes", sources={
    "TransportMH": [("results_ns2d_L4", "TransportMH")],
    "pCN + ESS sweeps": [("results_ns2d_L4", "MLSMC_pCN")],
    "MLSMC-pCN": [("results_ns2d_L4_pcn_plain", "MLSMC_pCN")],
    "LightSB": [("results_ns2d_L4", "LightSB")]})


def load(sources):
    """{L: list of (estimate, total_seconds, [level contributions])}"""
    out = {}
    for folder, method in sources:
        path = ROOT / folder / "runs.csv"
        if not path.exists():
            continue
        for r in csv.DictReader(open(path)):
            if r["method"] != method:
                continue
            contrib = [t["contribution"] for t in sorted(json.loads(r["contributions"]), key=lambda t: t["level"])]
            out.setdefault(int(r["L"]), []).append((float(r["estimate"]), float(r["total_seconds"]), contrib))
    return out


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.labelcolor": INK, "axes.titlesize": 9, "legend.fontsize": 7.5})
OUT.mkdir(parents=True, exist_ok=True)
CURVES, SDS = {}, {}          # per problem, for the combined one-legend figures at the end

for name, spec in PROBLEMS.items():
    if not any((ROOT / folder / "runs.csv").exists() for srcs in spec["sources"].values() for folder, _ in srcs):
        print(f"{name}: no result folders yet, skipped")
        continue
    ref = json.loads((ROOT / spec["ref"]).read_text())
    R, Rse = ref["value"], ref.get("se") or 0.0
    rows = []
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for method, st in STYLE.items():
        data = load(spec["sources"][method])
        pts = []
        for L in sorted(data):
            if L == 0:
                continue
            e = np.array([x[0] for x in data[L]])
            t = float(np.mean([x[1] for x in data[L]]))
            rel = math.sqrt(np.mean((e - R) ** 2)) / abs(R)
            bias = (e.mean() - R) / abs(R)
            se = math.hypot(e.std(ddof=1) / math.sqrt(len(e)), Rse) / abs(R)
            pts.append((t, rel, L))
            rows.append({"method": method, "L": L, "repeats": len(e), "wall_s": t, "relRMSE": rel,
                         "relbias": bias, "relbias_se": se})
        if not pts:
            continue
        CURVES.setdefault(name, {})[method] = pts
        x, y, Ls = zip(*pts)
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.5, marker=st["marker"], ms=4.5,
                mec="white", mew=0.6, label=st["label"])
        if method == "TransportMH":                      # selective direct labels: first and last two horizons
            for xi, yi, Li in [pts[0]] + pts[-2:]:
                ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=(4, 3), fontsize=6.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean wall time per run (s)")
    ax.set_ylabel("relative RMSE")
    ax.set_title(spec["title"], color=INK)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_{name}_error_vs_time.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_{name}_error_vs_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # per-level correction s.d. at the finest horizon
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    finest = int(spec["finest"][1:])
    for method, st in STYLE.items():
        data = load(spec["sources"][method]).get(finest)
        if not data:
            continue
        C = np.array([x[2] for x in data])               # [repeats, levels]
        sd = C.std(axis=0, ddof=1)[1:]
        lv = np.arange(1, len(sd) + 1)
        SDS.setdefault(name, {})[method] = (lv, sd)
        ax.plot(lv, sd, color=st["color"], ls=st["ls"], lw=1.5, marker=st["marker"], ms=4.5,
                mec="white", mew=0.6, label=st["label"])
        for l, v in zip(lv, sd):
            rows.append({"method": method, "L": finest, "level": int(l), "correction_sd": float(v)})
    ax.set_yscale("log")
    ax.set_xlabel(r"level $l$")
    ax.set_ylabel(r"s.d. of level-$l$ correction (per run)")
    ax.set_title(f"{spec['title']}, horizon {spec['finest']}", color=INK)
    ax.set_xticks(range(1, finest + 1))
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=2.6)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_{name}_level_sd.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"fig_{name}_level_sd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    keys = sorted({k for r in rows for k in r}, key=lambda k: ["method", "L", "level", "repeats", "wall_s", "relRMSE",
                                                                 "relbias", "relbias_se", "correction_sd"].index(k))
    with open(OUT / f"table_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: wrote figures and table to {OUT}")


# Combined figures: one row of panels (test cases 1, 2, 3) with a single shared legend.
ORDER = [n for n in ("beskos", "darcy", "ns2d") if n in CURVES]
PANEL = {"beskos": "Test case 1 (1D elliptic)", "darcy": "Test case 2 (Darcy flow)", "ns2d": "Test case 3 (Navier\u2013Stokes)"}
if len(ORDER) > 1:
    for kind in ("error_vs_time", "level_sd"):
        fig, axes = plt.subplots(1, len(ORDER), figsize=(2.45 * len(ORDER), 2.5))
        for ax, name in zip(axes, ORDER):
            for method, st in STYLE.items():
                if kind == "error_vs_time":
                    pts = CURVES[name].get(method)
                    if not pts:
                        continue
                    x, y, Ls = zip(*pts)
                else:
                    if method not in SDS.get(name, {}):
                        continue
                    x, y = SDS[name][method]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.3, marker=st["marker"], ms=3.8,
                        mec="white", mew=0.5, label=st["label"])
                if kind == "error_vs_time" and method == "TransportMH":
                    for (xi, yi, Li), off in [(pts[0], (-6, -10)), (pts[-1], (4, 3))]:
                        ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=off, fontsize=6.5,
                                    color=MUTED, ha="right" if off[0] < 0 else "left")
            ax.set_yscale("log")
            if kind == "error_vs_time":
                ax.set_xscale("log")
                ax.set_xlabel("mean wall time per run (s)")
            else:
                finest = int(PROBLEMS[name]["finest"][1:])
                ax.set_xticks(range(1, finest + 1))
                ax.set_xlabel(r"level $l$" + f" (horizon L{finest})")
            ax.set_title(PANEL[name], color=INK)
            style_axes(ax)
        axes[0].set_ylabel("relative RMSE" if kind == "error_vs_time" else r"s.d. of level-$l$ correction")
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, handlelength=2.6, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(OUT / f"fig_all_{kind}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"fig_all_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"combined: wrote fig_all_error_vs_time and fig_all_level_sd to {OUT}")


# Stacked figures: one column of panels (1D elliptic, Darcy flow, Navier--Stokes), sized for 10 cm width,
# with a single shared legend below the last panel.
STACK_TITLE = {"beskos": "1D elliptic", "darcy": "Darcy flow", "ns2d": "Navier–Stokes"}
if len(ORDER) > 1:
    W = 10 / 2.54
    for kind in ("error_vs_time", "level_sd"):
        fig, axes = plt.subplots(len(ORDER), 1, figsize=(W, 2.3 * len(ORDER) + 0.35))
        for ax, name in zip(axes, ORDER):
            for method, st in STYLE.items():
                if kind == "error_vs_time":
                    pts = CURVES[name].get(method)
                    if not pts:
                        continue
                    x, y, Ls = zip(*pts)
                else:
                    if method not in SDS.get(name, {}):
                        continue
                    x, y = SDS[name][method]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.3, marker=st["marker"], ms=3.8,
                        mec="white", mew=0.5, label=st["label"])
                if kind == "error_vs_time" and method == "TransportMH":
                    for (xi, yi, Li), off in [(pts[0], (-6, -10)), (pts[-1], (4, 3))]:
                        ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=off, fontsize=6.5,
                                    color=MUTED, ha="right" if off[0] < 0 else "left")
            ax.set_yscale("log")
            if kind == "error_vs_time":
                ax.set_xscale("log")
                ax.set_xlabel("mean wall time per run (s)")
                ax.set_ylabel("relative RMSE")
            else:
                finest = int(PROBLEMS[name]["finest"][1:])
                ax.set_xticks(range(1, finest + 1))
                ax.set_xlabel(r"level $l$" + f" (horizon L{finest})")
                ax.set_ylabel(r"s.d. of level-$l$ correction")
            ax.set_title(STACK_TITLE[name], color=INK)
            style_axes(ax)
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="lower center", ncol=2, handlelength=2.6, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2)
        fig.savefig(OUT / f"fig_stack_{kind}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"fig_stack_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"stacked: wrote fig_stack_error_vs_time and fig_stack_level_sd to {OUT}")


# Grid figures: two panels in a row (1D elliptic, Darcy flow) and the third (Navier--Stokes) centred below,
# sized for the full text width (15.3 cm), with a single shared legend at the bottom.
if len(ORDER) == 3:
    from matplotlib.gridspec import GridSpec
    W = 15.3 / 2.54
    for kind in ("error_vs_time", "level_sd"):
        fig = plt.figure(figsize=(W, 5.3))
        gs = GridSpec(2, 4, figure=fig)
        axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
        for ax, name in zip(axes, ORDER):
            for method, st in STYLE.items():
                if kind == "error_vs_time":
                    pts = CURVES[name].get(method)
                    if not pts:
                        continue
                    x, y, Ls = zip(*pts)
                else:
                    if method not in SDS.get(name, {}):
                        continue
                    x, y = SDS[name][method]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.3, marker=st["marker"], ms=3.8,
                        mec="white", mew=0.5, label=st["label"])
                if kind == "error_vs_time" and method == "TransportMH":
                    for (xi, yi, Li), off in [(pts[0], (-6, -10)), (pts[-1], (4, 3))]:
                        ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=off, fontsize=6.5,
                                    color=MUTED, ha="right" if off[0] < 0 else "left")
            ax.set_yscale("log")
            if kind == "error_vs_time":
                ax.set_xscale("log")
                ax.set_xlabel("mean wall time per run (s)")
                ax.set_ylabel("relative RMSE")
            else:
                finest = int(PROBLEMS[name]["finest"][1:])
                ax.set_xticks(range(1, finest + 1))
                ax.set_xlabel(r"level $l$" + f" (horizon L{finest})")
                ax.set_ylabel(r"s.d. of level-$l$ correction")
            ax.set_title(STACK_TITLE[name], color=INK)
            style_axes(ax)
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, handlelength=2.6, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.0, w_pad=1.5)
        fig.savefig(OUT / f"fig_grid_{kind}.pdf", bbox_inches="tight")
        fig.savefig(OUT / f"fig_grid_{kind}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"grid: wrote fig_grid_error_vs_time and fig_grid_level_sd to {OUT}")


# SIAM row figures: the three test cases in one row, sized for the SIAM text width (370 pt = 5.125 in),
# with a single shared legend below. Used by the SINUM version (seq_eot_v2/sinum).
if len(ORDER) == 3:
    W = 370 / 72.27
    for kind in ("error_vs_time", "level_sd"):
        fig, axes = plt.subplots(1, 3, figsize=(W, 2.2))
        for ax, name in zip(axes, ORDER):
            for method, st in STYLE.items():
                if kind == "error_vs_time":
                    pts = CURVES[name].get(method)
                    if not pts:
                        continue
                    x, y, Ls = zip(*pts)
                else:
                    if method not in SDS.get(name, {}):
                        continue
                    x, y = SDS[name][method]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.1, marker=st["marker"], ms=3.0,
                        mec="white", mew=0.4, label=st["label"])
                if kind == "error_vs_time" and method == "TransportMH":
                    for (xi, yi, Li), off in [(pts[0], (-4, -9)), (pts[-1], (-3, 5))]:
                        ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=off, fontsize=6,
                                    color=MUTED, ha="right" if off[0] < 0 else "left")
            ax.set_yscale("log")
            if kind == "error_vs_time":
                ax.set_xscale("log")
                ax.set_xlabel("wall time per run (s)", fontsize=7.5)
            else:
                finest = int(PROBLEMS[name]["finest"][1:])
                ax.set_xticks(range(1, finest + 1))
                ax.set_xlabel(r"level $l$" + f" (horizon L{finest})", fontsize=7.5)
            ax.set_title(STACK_TITLE[name], color=INK, fontsize=8)
            style_axes(ax)
            ax.tick_params(labelsize=6.5)
            lo, hi = ax.get_ylim()
            if hi / lo < 10:                      # less than a decade: label a few plain ticks, no minor labels
                import matplotlib.ticker as mt
                ax.yaxis.set_minor_formatter(mt.NullFormatter())
                ax.yaxis.set_major_locator(mt.FixedLocator([v for v in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05) if lo <= v <= hi]))
                ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        axes[0].set_ylabel("relative RMSE" if kind == "error_vs_time" else r"s.d. of level-$l$ correction", fontsize=7.5)
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, handlelength=2.2, fontsize=6.5,
                   columnspacing=1.0, bbox_to_anchor=(0.53, 0.005))
        # identical page size and panel boxes for both figures (no tight bounding box), so that the two
        # figures have the same aspect ratio and their panels line up when both are set at \textwidth
        for ax in axes:
            ax.set_box_aspect(1)              # square panels
        fig.subplots_adjust(left=0.095, right=0.99, bottom=0.245, top=0.925, wspace=0.34)
        fig.savefig(OUT / f"fig_siam_{kind}.pdf")
        fig.savefig(OUT / f"fig_siam_{kind}.png", dpi=200)
        plt.close(fig)
    print(f"siam: wrote fig_siam_error_vs_time and fig_siam_level_sd to {OUT}")


# SIAM grid figures: two panels in a row (1D elliptic, Darcy flow) and Navier--Stokes centred below, sized for the
# SIAM text width (370 pt), square panels, one shared legend; identical frames for both figures.
if len(ORDER) == 3:
    from matplotlib.gridspec import GridSpec
    W = 370 / 72.27
    for kind in ("error_vs_time", "level_sd"):
        fig = plt.figure(figsize=(W, 4.15))
        gs = GridSpec(2, 4, figure=fig, left=0.10, right=0.98, bottom=0.145, top=0.95, wspace=0.55, hspace=0.62)
        axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
        for ax, name in zip(axes, ORDER):
            for method, st in STYLE.items():
                if kind == "error_vs_time":
                    pts = CURVES[name].get(method)
                    if not pts:
                        continue
                    x, y, Ls = zip(*pts)
                else:
                    if method not in SDS.get(name, {}):
                        continue
                    x, y = SDS[name][method]
                ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.2, marker=st["marker"], ms=3.4,
                        mec="white", mew=0.5, label=st["label"])
                if kind == "error_vs_time" and method == "TransportMH":
                    for (xi, yi, Li), off in [(pts[0], (-4, -9)), (pts[-1], (-3, 5))]:
                        ax.annotate(f"L{Li}", (xi, yi), textcoords="offset points", xytext=off, fontsize=6.5,
                                    color=MUTED, ha="right" if off[0] < 0 else "left")
            ax.set_yscale("log")
            if kind == "error_vs_time":
                ax.set_xscale("log")
                ax.set_xlabel("mean wall time per run (s)", fontsize=8)
                ax.set_ylabel("relative RMSE", fontsize=8)
            else:
                finest = int(PROBLEMS[name]["finest"][1:])
                ax.set_xticks(range(1, finest + 1))
                ax.set_xlabel(r"level $l$" + f" (horizon L{finest})", fontsize=8)
                ax.set_ylabel(r"s.d. of level-$l$ correction", fontsize=8)
            ax.set_title(STACK_TITLE[name], color=INK, fontsize=8.5)
            style_axes(ax)
            ax.tick_params(labelsize=7)
            ax.set_box_aspect(0.8)          # a little wider than tall
            lo, hi = ax.get_ylim()
            if hi / lo < 10:
                import matplotlib.ticker as mt
                ax.yaxis.set_minor_formatter(mt.NullFormatter())
                ax.yaxis.set_major_locator(mt.FixedLocator([v for v in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05) if lo <= v <= hi]))
                ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        h, lab = axes[-1].get_legend_handles_labels()
        fig.legend(h, lab, frameon=False, loc="lower center", ncol=4, handlelength=2.0, fontsize=6.6,
                   columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
        fig.savefig(OUT / f"fig_siamgrid_{kind}.pdf")
        fig.savefig(OUT / f"fig_siamgrid_{kind}.png", dpi=200)
        plt.close(fig)
    print(f"siam grid: wrote fig_siamgrid_error_vs_time and fig_siamgrid_level_sd to {OUT}")
