"""Slide 10 - "Propagate the whole distribution, not just the mean".

Left  : a reliability block diagram. C1 - (C2a || C2b) - C3.
        Each block is a posterior, not a point.
Right : Monte-Carlo over the component posteriors -> a distribution for system
        reliability over a 7-year life, with its own 90% credible interval.

System logic:  R_sys = R_C1 * [1 - (1 - R_C2a)(1 - R_C2b)] * R_C3
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bayes_style import (apply_style, gamma_fit, reliability, finish,
                         TEAL, BLUE, NAVY, AMBER, TEAL_L)

apply_style()
rng = np.random.default_rng(7)

# --- knobs --------------------------------------------------------------- #
YEARS   = 7
N_DRAWS = 200_000
# component posteriors as Gamma(alpha, beta_per_hour):
COMPONENTS = {
    "C1":  (2.0, 2.75e6),   # ~727 FIT
    "C2a": (3.0, 1.50e6),   # ~2000 FIT (redundant)
    "C2b": (3.0, 1.50e6),
    "C3":  (2.0, 2.75e6),   # ~727 FIT
}
# ------------------------------------------------------------------------- #

def sample_R(name):
    a, b = COMPONENTS[name]
    lam = gamma_fit(a, b).rvs(N_DRAWS, random_state=rng)   # FIT
    return reliability(lam, YEARS)

R1, R2a, R2b, R3 = (sample_R(n) for n in ("C1", "C2a", "C2b", "C3"))
R_sys = R1 * (1 - (1 - R2a) * (1 - R2b)) * R3

mean = R_sys.mean()
lo, hi = np.percentile(R_sys, [5, 95])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.0),
                               gridspec_kw=dict(width_ratios=[1.05, 1.25]))

# ---------------- left: reliability block diagram ------------------------ #
axL.set_xlim(0, 10)
axL.set_ylim(0, 10)
axL.axis("off")
axL.set_title("Each block = a posterior, not a point", fontsize=15, color=NAVY)

def block(ax, cx, cy, text, color, w=2.0, h=1.7):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.28",
        linewidth=2.5, edgecolor=NAVY, facecolor=color, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center",
            color="white", fontsize=16, fontweight="bold", zorder=4)

def wire(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color=NAVY, lw=2.5, zorder=1,
            solid_capstyle="butt")

yc = 5.0
# series wires + nodes
wire(axL, 0.2, yc, 1.4, yc)          # input -> C1
block(axL, 2.4, yc, "C1", TEAL)
wire(axL, 3.4, yc, 4.3, yc)          # C1 -> split node
wire(axL, 8.0, yc, 8.6, yc)          # join node -> C3 already; handled below

# parallel pair
y_top, y_bot = 7.0, 3.0
block(axL, 5.6, y_top, "C2a", BLUE)
block(axL, 5.6, y_bot, "C2b", BLUE)
# split
wire(axL, 4.3, yc, 4.3, y_top); wire(axL, 4.3, y_top, 4.6, y_top)
wire(axL, 4.3, yc, 4.3, y_bot); wire(axL, 4.3, y_bot, 4.6, y_bot)
# join
wire(axL, 6.6, y_top, 6.9, y_top); wire(axL, 6.9, y_top, 6.9, yc)
wire(axL, 6.6, y_bot, 6.9, y_bot); wire(axL, 6.9, y_bot, 6.9, yc)
wire(axL, 6.9, yc, 7.6, yc)
block(axL, 8.6, yc, "C3", TEAL)
wire(axL, 9.6, yc, 9.9, yc)          # C3 -> output

axL.text(5.6, 1.4, "redundancy (parallel)", ha="center",
         color=BLUE, fontsize=13, fontstyle="italic")

# ---------------- right: Monte-Carlo histogram --------------------------- #
axR.hist(R_sys, bins=70, color=TEAL_L, edgecolor="none")
axR.axvline(mean, color=NAVY, lw=2.8, zorder=4)
axR.axvline(lo, color=AMBER, lw=3.0, ls=(0, (6, 4)), zorder=4)
axR.axvline(hi, color=AMBER, lw=3.0, ls=(0, (6, 4)), zorder=4)
axR.text(lo - 0.005, axR.get_ylim()[1] * 0.72,
         f"90% CI\n[{lo:.3f}, {hi:.3f}]",
         color=AMBER, fontsize=13.5, fontweight="bold", ha="right", va="top")

axR.set_title("System reliability (Monte-Carlo)", fontsize=16)
axR.set_xlabel("System R over 7 yr")
axR.set_yticks([])
axR.set_xlim(0.63, 1.0)

fig.suptitle("Propagate the whole distribution, not just the mean",
             fontsize=19, fontweight="bold", color=NAVY, y=1.0)
fig.tight_layout()

print(f"system mean R = {mean:.3f},  90% CI = [{lo:.3f}, {hi:.3f}]")
finish(fig, "slide10_system_montecarlo.png")
