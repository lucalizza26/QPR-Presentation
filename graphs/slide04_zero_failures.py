"""Slide 4 - "Zero failures != zero failure rate".

Worked example: a unit with a 500 FIT handbook prior that has flown
100 000 operating hours with ZERO failures.

The classical MLE collapses to lambda = 0 (R = 100%, useless). The Bayesian
posterior is a proper distribution with a sensible mean and, crucially, a 95%
upper bound you can actually design margins to.

Conjugate Gamma-Poisson update:  Gamma(a0, b0)  ->  Gamma(a0 + k, b0 + T).
With a weak prior (a0 = 1) the posterior stays an exponential (shape = 1,
monotonic decreasing) - and note how little 100 000 h moves a prior whose mean
already implies failures only every ~2 million hours.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         TEAL, NAVY, RED, AMBER, FILL)

apply_style()

# --- knobs (the worked example) ------------------------------------------ #
PRIOR_FIT    = 500.0      # FIT  -> handbook / ECSS prior failure rate
PRIOR_ALPHA  = 1.0        # prior strength = "equivalent prior failures"
T_HOURS      = 100_000    # operating hours flown
K_FAILURES   = 0          # observed failures  (zero -> the whole point)
XMAX         = 2800       # FIT  -> x-axis extent
# ------------------------------------------------------------------------- #

# Conjugate update: rate beta0 anchors the prior mean to PRIOR_FIT.
beta0 = PRIOR_ALPHA * 1e9 / PRIOR_FIT
post  = gamma_fit(PRIOR_ALPHA + K_FAILURES, beta0 + T_HOURS)
mean  = post.mean()
ub95  = post.ppf(0.95)
print(f"prior {PRIOR_FIT:.0f} FIT, {K_FAILURES} failures in {T_HOURS:,} h"
      f"  ->  posterior mean {mean:.0f} FIT, 95% upper bound {ub95:.0f} FIT")

x = np.linspace(0, XMAX, 600)
y = post.pdf(x)

fig, ax = plt.subplots(figsize=(9.8, 6.2))

ax.fill_between(x, y, color=FILL, zorder=1)
ax.plot(x, y, color=TEAL, lw=4.0, zorder=3)

# vertical markers
ax.axvline(0,    color=RED,   lw=5, zorder=4)                      # MLE = 0
ax.axvline(mean, color=NAVY,  lw=2.5, zorder=4)                    # posterior mean
ax.axvline(ub95, color=AMBER, lw=4, ls=(0, (6, 4)), zorder=4)     # 95% UB

# annotations
ax.text(0.015 * XMAX, y.max() * 0.93, f"posterior mean\n≈ {mean:.0f} FIT",
        color=NAVY, fontsize=13, fontweight="bold", va="top")
ax.text(0.25 * XMAX, y.max() * 0.93,
        "Frequentist MLE\n$\\hat{\\lambda} = 0/T = 0$\n→ R = 100%  (?)",
        color=RED, fontsize=13, fontweight="bold", va="top")
ax.annotate(f"95% upper bound\n≈ {ub95:.0f} FIT\n(usable for margins)",
            xy=(ub95, post.pdf(ub95)),
            xytext=(ub95 + 0.07 * XMAX, y.max() * 0.60),
            color=AMBER, fontsize=14, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=2.2))

ax.set_title("Zero failures ≠ zero failure rate")
ax.set_xlabel("Failure rate  λ  (FIT)")
ax.set_ylabel("Posterior density")
ax.set_xlim(0, XMAX)
ax.set_ylim(0, y.max() * 1.08)
ax.set_yticks([])

finish(fig, "slide04_zero_failures.png")
