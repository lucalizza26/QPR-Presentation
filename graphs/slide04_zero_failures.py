"""Slide 4 - "Zero failures != zero failure rate".

A unit with T operating hours and ZERO failures. The classical MLE collapses to
lambda = 0 (R = 100%, useless). The Bayesian posterior is a proper distribution
with a sensible mean (~364 FIT) and, crucially, a 95% upper bound (~1089 FIT)
you can actually design margins to.

Zero failures, weak prior -> posterior is an exponential (shape = 1) with
mean 364 FIT.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         TEAL, NAVY, RED, AMBER, FILL)

apply_style()

# --- knobs --------------------------------------------------------------- #
POST_MEAN   = 364.0     # FIT  -> posterior mean with zero failures
POST_SHAPE  = 1.0       # shape = 1 -> exponential (monotonic decreasing)
XMAX        = 2200      # FIT  -> x-axis extent
# ------------------------------------------------------------------------- #

# Gamma(shape, rate) with the requested mean, expressed over FIT.
post = gamma_fit(POST_SHAPE, POST_SHAPE * 1e9 / POST_MEAN)
ub95 = post.ppf(0.95)

x = np.linspace(0, XMAX, 600)
y = post.pdf(x)

fig, ax = plt.subplots(figsize=(8.8, 6.2))

ax.fill_between(x, y, color=FILL, zorder=1)
ax.plot(x, y, color=TEAL, lw=4.0, zorder=3)

# vertical markers
ax.axvline(0,        color=RED,   lw=5, zorder=4)                       # MLE = 0
ax.axvline(POST_MEAN, color=NAVY, lw=2.5, zorder=4)                     # mean
ax.axvline(ub95,     color=AMBER, lw=4, ls=(0, (6, 4)), zorder=4)      # 95% UB

# annotations
ax.text(30, y.max() * 0.93, "posterior mean\n≈ 364 FIT",
        color=NAVY, fontsize=13, fontweight="bold", va="top")
ax.text(640, y.max() * 0.93,
        "Frequentist MLE\n$\\hat{\\lambda} = 0/T = 0$\n→ R = 100%  (?)",
        color=RED, fontsize=13, fontweight="bold", va="top")
ax.annotate("95% upper bound\n≈ 1089 FIT\n(usable for margins)",
            xy=(ub95, post.pdf(ub95)), xytext=(1280, y.max() * 0.62),
            color=AMBER, fontsize=14, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2))

ax.set_title("Zero failures ≠ zero failure rate")
ax.set_xlabel("Failure rate  λ  (FIT)")
ax.set_ylabel("Posterior density")
ax.set_xlim(0, XMAX)
ax.set_ylim(0, y.max() * 1.08)
ax.set_yticks([])

finish(fig, "slide04_zero_failures.png")
