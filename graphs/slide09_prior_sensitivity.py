"""Slide 9 - "The prior matters when data is sparse - and washes out as evidence grows".

Two panels, three defensible priors each:
    informative (ECSS)   - navy solid
    weakly informative   - blue dashed
    less-conservative    - amber dash-dot

Left  : sparse data (k = 0,  T = 0.3M h)  -> the three posteriors disagree.
Right : abundant data (k = 30, T = 30M h) -> they collapse onto one answer.

Each curve is normalised to unit peak ("relative density"), as on the slide.
"""
import numpy as np
import matplotlib.pyplot as plt

from bayes_style import (apply_style, gamma_fit, finish,
                         NAVY, BLUE, AMBER)

apply_style()

# --- knobs --------------------------------------------------------------- #
# prior: (alpha, beta_per_hour, colour, linestyle, label)
PRIORS = [
    (2.5, 2.5e6, NAVY,  "-",            "Informative (ECSS)"),
    (1.0, 1.0e6, BLUE,  (0, (7, 4)),    "Weakly informative"),
    (2.2, 3.67e6, AMBER, (0, (8, 3, 2, 3)), "Less-conservative"),
]
SPARSE   = dict(k=0,  T=0.3e6, title="Sparse data  (k = 0,  T = 0.3M h)")
ABUNDANT = dict(k=30, T=30e6,  title="Abundant data  (k = 30,  T = 30M h)")
XMAX = 2600
# ------------------------------------------------------------------------- #

x = np.linspace(0, XMAX, 800)

def draw(ax, case, legend):
    for alpha, beta, color, ls, label in PRIORS:
        post = gamma_fit(alpha + case["k"], beta + case["T"])
        y = post.pdf(x)
        ax.plot(x, y / y.max(), color=color, lw=3.6, ls=ls,
                label=label if legend else None)
    ax.set_title(case["title"], fontsize=16)
    ax.set_xlabel("Failure rate  λ  (FIT)")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.6), sharey=True)

draw(axL, SPARSE,   legend=True)
draw(axR, ABUNDANT, legend=False)
axL.set_ylabel("Relative density")
axL.legend(loc="upper right", fontsize=12)

fig.suptitle("The prior matters when data is sparse — and washes out as evidence grows",
             fontsize=19, fontweight="bold", color=NAVY, y=1.0)
fig.tight_layout()

finish(fig, "slide09_prior_sensitivity.png")
