"""Slide 8 - "One component, start to finish".

The full worked example:
    prior      Gamma(1, 1e6 h)   = 1000 FIT (handbook)
    data       k = 1 failure in T = 1.75M operating hours
    posterior  Gamma(2, 2.75e6)  -> mean 727 FIT

The shaded band is the central CI credible interval under the posterior
(set CI below; default 60%).
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         GREY, TEAL, NAVY, FILL)

apply_style()

# --- knobs --------------------------------------------------------------- #
PRIOR_ALPHA, PRIOR_BETA = 1.0, 1e6        # 1000 FIT handbook prior
K, T = 1, 1.75e6                          # REX data
CI = 0.60                                 # central credible-interval fraction
XMAX = 2600
# ------------------------------------------------------------------------- #

prior = gamma_fit(PRIOR_ALPHA, PRIOR_BETA)
post  = gamma_fit(PRIOR_ALPHA + K, PRIOR_BETA + T)

mean = post.mean()
tail = (1 - CI) / 2                       # central interval -> equal tails
lo, hi = post.ppf(tail), post.ppf(1 - tail)
print(f"posterior mean {mean:.0f} FIT, {CI*100:.0f}% CI [{lo:.0f}, {hi:.0f}] FIT")

x = np.linspace(0, XMAX, 800)
yp = post.pdf(x)

fig, ax = plt.subplots(figsize=(9.6, 6.4))

# shade the credible interval under the posterior
mask = (x >= lo) & (x <= hi)
ax.fill_between(x[mask], yp[mask], color=FILL, zorder=1)

ax.plot(x, prior.pdf(x), color=GREY, lw=3.6, ls=(0, (7, 5)),
        label="Prior  (1000 FIT)")
ax.plot(x, yp, color=TEAL, lw=4.2, label="Posterior", zorder=3)

# reference verticals: posterior mean and prior mean (1000)
ax.axvline(mean, color=NAVY, lw=2.0, ls=":", zorder=4)
ax.axvline(1000, color=GREY, lw=2.0, ls=":", zorder=4)

ax.text(lo + 40, yp.max() * 0.86,
        f"{CI*100:.0f}% credible interval\n[{lo:.0f}, {hi:.0f}] FIT",
        color=NAVY, fontsize=14, fontweight="bold", va="top")
ax.annotate(f"posterior mean\n{mean:.0f} FIT",
            xy=(mean, post.pdf(mean) * 0.45), xytext=(1080, yp.max() * 0.52),
            color=NAVY, fontsize=14, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.0))

ax.set_title("One component: prior updated by 1 failure in 1.75M h")
ax.set_xlabel("Failure rate  λ  (FIT)")
ax.set_ylabel("Density")
ax.set_xlim(0, XMAX)
ax.set_ylim(0, yp.max() * 1.1)
ax.set_yticks([])
ax.legend(loc="upper right")

finish(fig, "slide08_one_component.png")
