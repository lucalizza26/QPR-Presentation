"""Slide 7 - "More evidence -> tighter posterior".

Same weak prior, three increasing amounts of REX data. As operating time and
failures accumulate the posterior tightens and homes in on the true rate
(dotted amber line). Demonstrates the conjugate update Gamma(a0+k, b0+T).

Prior: Gamma(1, 1e6 h)  ->  mean 1000 FIT.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         TEAL_L, TEAL, NAVY, AMBER)

apply_style()

# --- knobs --------------------------------------------------------------- #
PRIOR_ALPHA, PRIOR_BETA = 1.0, 1e6        # weak prior, mean 1000 FIT
TRUE_RATE = 700                           # FIT, where the data is generated
XMAX = 2600
# (T in hours, k failures, colour, label)
CASES = [
    (0.30e6, 0, TEAL_L, "T = 0.30M h,  k = 0"),
    (1.50e6, 1, TEAL,   "T = 1.50M h,  k = 1"),
    (6.00e6, 4, NAVY,   "T = 6.00M h,  k = 4"),
]
# ------------------------------------------------------------------------- #

x = np.linspace(0, XMAX, 700)
fig, ax = plt.subplots(figsize=(9.2, 6.2))

for T, k, color, label in CASES:
    post = gamma_fit(PRIOR_ALPHA + k, PRIOR_BETA + T)
    ax.plot(x, post.pdf(x), color=color, lw=4.0, label=label)

ax.axvline(TRUE_RATE, color=AMBER, lw=3.0, ls=":", label="true rate")

ax.set_title("More evidence → tighter posterior")
ax.set_xlabel("Failure rate  λ  (FIT)")
ax.set_ylabel("Posterior density")
ax.set_xlim(0, XMAX)
ax.set_ylim(bottom=0)
ax.set_yticks([])
ax.legend(loc="upper right")

finish(fig, "slide07_more_evidence.png")
