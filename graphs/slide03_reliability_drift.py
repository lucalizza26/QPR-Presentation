"""Slide 3 - "Predictions are built to be conservative".

The systematic gap between the conservative handbook prediction (ECSS / MIL-HDBK)
and what the hardware actually does in orbit (REX). Two exponential reliability
curves; the area between them is the gap.

Tweak the two failure rates below to open / close the gap.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import apply_style, reliability, finish, RED, TEAL, NAVY, FILL

apply_style()

# --- knobs --------------------------------------------------------------- #
YEARS          = 7                 # mission length on the x-axis
LAMBDA_PRED    = 1718              # FIT  -> ECSS/MIL prediction (R(7yr) ~ 0.90)
LAMBDA_OBSERVED = 666             # FIT  -> observed in-orbit    (R(7yr) ~ 0.96)
ARROW_YEAR     = 4.9              # where the "systematic gap" arrow points
# ------------------------------------------------------------------------- #

t = np.linspace(0, YEARS, 400)
R_pred = reliability(LAMBDA_PRED, t)
R_obs  = reliability(LAMBDA_OBSERVED, t)

fig, ax = plt.subplots(figsize=(8.6, 6.0))

ax.fill_between(t, R_pred, R_obs, color=FILL, zorder=1)
ax.plot(t, R_pred, color=RED,  lw=4.0, ls=(0, (6, 4)),
        label="ECSS / MIL-HDBK prediction", zorder=3)
ax.plot(t, R_obs,  color=TEAL, lw=4.0,
        label="Observed in-orbit (REX)", zorder=3)

# "systematic gap" annotation pointing into the shaded band
ymid = 0.5 * (reliability(LAMBDA_PRED, ARROW_YEAR) +
              reliability(LAMBDA_OBSERVED, ARROW_YEAR))
ax.annotate("systematic\ngap",
            xy=(ARROW_YEAR, ymid), xytext=(3.0, 0.80),
            color=NAVY, fontsize=16, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.2))

ax.set_title("Predictions are built to be conservative")
ax.set_xlabel("Mission elapsed time (years)")
ax.set_ylabel("Reliability  R(t)")
ax.set_xlim(0, YEARS)
ax.set_ylim(0.58, 1.0)
ax.set_xticks(range(0, YEARS + 1))
ax.legend(loc="lower left")

finish(fig, "slide03_reliability_drift.png")
