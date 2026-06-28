"""Slide 5 - "Prior x Data -> Posterior".

The three pieces of Bayes, shown as densities over the failure rate:
    prior      (grey)        - the ECSS prediction, written as a distribution
    likelihood (blue dashed) - what the REX data says
    posterior  (teal)        - prior x likelihood, renormalised; narrower than either

Curves are normalised to a common peak height ("relative density") so the
shapes are comparable, exactly like the slide.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         GREY, BLUE, TEAL, FILL)

apply_style()

# --- knobs (all illustrative) -------------------------------------------- #
# prior: weak, mean ~1100 FIT (shape 1 -> monotonic decreasing, peak at 0)
PRIOR_ALPHA, PRIOR_MEAN = 1.0, 1100.0
# data: k failures in T hours -> likelihood is Gamma(k+1, T)
K, T = 2, 3.33e6
XMAX = 2600
# ------------------------------------------------------------------------- #

prior_beta = PRIOR_ALPHA * 1e9 / PRIOR_MEAN
prior = gamma_fit(PRIOR_ALPHA, prior_beta)
like  = gamma_fit(K + 1, T)                              # likelihood as a density
post  = gamma_fit(PRIOR_ALPHA + K, prior_beta + T)       # conjugate update

x = np.linspace(0, XMAX, 700)
norm = lambda d: d.pdf(x) / d.pdf(x).max()               # scale to unit peak

fig, ax = plt.subplots(figsize=(9.6, 6.0))

yp = norm(post)
ax.fill_between(x, yp, color=FILL, zorder=1)
ax.plot(x, norm(prior), color=GREY, lw=3.4, label="Prior  (ECSS prediction)")
ax.plot(x, norm(like),  color=BLUE, lw=3.6, ls=(0, (7, 4)),
        label="Likelihood  (REX data)")
ax.plot(x, yp,          color=TEAL, lw=4.0, label="Posterior  (updated)", zorder=3)

ax.set_title("Prior  ×  Data  →  Posterior")
ax.set_xlabel("Failure rate  λ  (FIT)")
ax.set_ylabel("Relative density")
ax.set_xlim(0, XMAX)
ax.set_ylim(0, 1.12)
ax.set_yticks([])
ax.legend(loc="upper right")

finish(fig, "slide05_prior_data_posterior.png")
