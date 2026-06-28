import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func


def gamma_pdf(x, alpha, beta):
    """Gamma distribution PDF with shape alpha and rate beta."""
    x = np.asarray(x)
    pdf = (beta**alpha) / gamma_func(alpha) * x**(alpha - 1) * np.exp(-beta * x)
    pdf[x <= 0] = 0.0
    return pdf


def plot_gamma(alpha=2.0, beta=1.0, xmax=None, points=1000):
    if xmax is None:
        # set xmax to a few scale lengths: mean = alpha/beta
        xmax = max(5 * alpha / beta, 10.0)
    x = np.linspace(0, xmax, points)
    y = gamma_pdf(x, alpha, beta)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, label=f"alpha={alpha}, beta={beta}")
    plt.title("Gamma distribution PDF")
    plt.xlabel("x")
    plt.ylabel("pdf")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage: change alpha and beta as needed
    plot_gamma(alpha=1, beta=3)
