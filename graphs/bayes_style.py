"""Shared styling + helpers for the QPR Bayesian-reliability figures.

Every figure script imports from here so the whole deck stays visually
consistent. Tweak a colour or a font size in ONE place and it propagates to
all eight graphs. Per-graph data / layout lives in the individual scripts.

Palette was sampled directly from the original slides:
    navy   #003249   titles, dark text, mean lines, "abundant data" curve
    teal   #00AE9C   primary curve  (posterior / observed in-orbit)
    teal-L #86CFC6   light teal      (sparse-data curve)
    fill   #C7EDE9   light teal area fill under curves
    blue   #009BDA   likelihood / weakly-informative prior
    amber  #FBAB18   upper bound / less-conservative / credible-interval lines
    red    #EF2E42   ECSS / MIL-HDBK prediction
    grey   #728A98   prior line
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gamma as _gamma

# --------------------------------------------------------------------------- #
#  Colour palette                                                             #
# --------------------------------------------------------------------------- #
NAVY    = "#003249"
TEAL    = "#00AE9C"
TEAL_L  = "#86CFC6"
FILL    = "#C7EDE9"
BLUE    = "#009BDA"
AMBER   = "#FBAB18"
RED     = "#EF2E42"
GREY    = "#728A98"

# --------------------------------------------------------------------------- #
#  Global look & feel                                                         #
# --------------------------------------------------------------------------- #
def apply_style():
    """Apply the shared rcParams. Call once at the top of each script."""
    mpl.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       220,
        "font.family":       "DejaVu Sans",   # swap for "Arial" if you prefer
        "font.size":         14,

        "axes.titlesize":    19,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   NAVY,
        "axes.titlepad":     14,

        "axes.labelsize":    15,
        "axes.labelcolor":   NAVY,
        "axes.labelweight":  "regular",

        "axes.edgecolor":    "#9AA7AE",
        "axes.linewidth":    1.3,
        "axes.spines.top":   False,
        "axes.spines.right": False,

        "xtick.color":       NAVY,
        "ytick.color":       NAVY,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,

        "legend.fontsize":   12.5,
        "legend.frameon":    False,

        "lines.linewidth":   3.2,
        "lines.solid_capstyle": "round",

        "grid.color":        "#E7ECEE",
        "grid.linewidth":    1.0,
    })


# --------------------------------------------------------------------------- #
#  Distribution helper                                                        #
# --------------------------------------------------------------------------- #
def gamma_fit(alpha, beta_per_hour):
    """Gamma distribution for a failure rate, expressed in FIT (1 FIT = 1e-9/h).

    alpha          : shape  (= "equivalent prior failures" for a prior)
    beta_per_hour  : rate   (units of 1/hour)

    Returns a frozen scipy gamma object whose support is the rate in FIT,
    so .pdf(x), .mean(), .ppf(q) all work in FIT directly.
    """
    return _gamma(a=alpha, scale=1e9 / beta_per_hour)


def reliability(lambda_fit, years):
    """Exponential reliability R(t) = exp(-lambda * t) for a rate given in FIT."""
    hours = np.asarray(years) * 8760.0
    return np.exp(-np.asarray(lambda_fit) * 1e-9 * hours)


# --------------------------------------------------------------------------- #
#  Save / show                                                                #
# --------------------------------------------------------------------------- #
def finish(fig, name, show=True):
    """Save a transparent high-res PNG (drop straight into PowerPoint) + show."""
    fig.savefig(name, dpi=220, bbox_inches="tight", transparent=True,
                pad_inches=0.12)
    print(f"saved -> {name}")
    if show:
        plt.show()
