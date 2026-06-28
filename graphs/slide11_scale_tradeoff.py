"""Slide 11 - "Where is inference most informative?".

The aggregation trade-off across three levels (component -> subsystem -> system):
    statistical uncertainty (teal)  - shrinks as you aggregate (more pooled data)
    design resolution        (amber) - also shrinks: you lose the location of the
                                        weak spot

The lines cross near the system level: aggregating buys certainty but costs
resolution.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import apply_style, finish, TEAL, AMBER, NAVY

apply_style()

# --- knobs --------------------------------------------------------------- #
LEVELS = ["Component", "Subsystem", "System"]
STAT_UNCERTAINTY = [0.92, 0.46, 0.22]    # teal, circles
DESIGN_RESOLUTION = [1.00, 0.55, 0.18]   # amber, squares
# ------------------------------------------------------------------------- #

x = np.arange(len(LEVELS))
fig, ax = plt.subplots(figsize=(9.4, 6.0))

ax.plot(x, STAT_UNCERTAINTY, color=TEAL, lw=3.4, marker="o", ms=13,
        markeredgecolor="white", markeredgewidth=1.5,
        label="Statistical uncertainty", zorder=3)
ax.plot(x, DESIGN_RESOLUTION, color=AMBER, lw=3.4, marker="s", ms=13,
        markeredgecolor="white", markeredgewidth=1.5,
        label="Design resolution", zorder=3)

ax.text(1.52, 0.52, "more data,\ntighter estimates",
        color=TEAL, fontsize=14, fontweight="bold", ha="center")
ax.text(1.52, 0.12, "but you lose the\nlocation of the problem",
        color="#B07A0E", fontsize=14, fontweight="bold", ha="center")

ax.set_title("Where is inference most informative?")
ax.set_ylabel("Relative scale")
ax.set_xticks(x)
ax.set_xticklabels(LEVELS, fontsize=14)
ax.set_ylim(0, 1.08)
ax.set_xlim(-0.25, 2.25)
ax.legend(loc="upper right")

finish(fig, "slide11_scale_tradeoff.png")
