# bayesian_optimization.py
import numpy as np


def jit_plus_server(parameters):
    x = np.array(parameters)
    d = len(x)
    x1 = x - 0.15 * np.ones(shape=(d,))
    x2 = x - 0.85 * np.ones(shape=(d,))
    cpu_time = 2 - np.exp(-10 * x1**2) - 0.5 * np.exp(-10 * x2**2)
    return cpu_time.mean() + 0.005 * np.random.normal()


def plot_example_gpr(GPR, ax, x, y, err_bars=False, bottom_trace=False):
    x = np.array(x)
    y = np.array(y)
    gpr = GPR(x, y, sigma=0.15)
    x_hats = np.linspace(0, 1, 100)
    y_hats = []
    sigma_y_hats = []
    for x_hat in x_hats:
        ret = gpr.estimate(x_hat)
        try:
            y_hat, sigma_y_hat = ret
        except Exception:
            y_hat = ret
            sigma_y_hat = 0
        y_hats.append(y_hat)
        sigma_y_hats.append(sigma_y_hat)

    y_hats = np.array(y_hats)
    sigma_y_hats = np.array(sigma_y_hats)

    if err_bars:
        ax.fill_between(
            x_hats.flatten(),
            y_hats - sigma_y_hats,
            y_hats + sigma_y_hats,
            linewidth=1,
            label="Uncertainty in estimate",
            alpha=0.2,
        )

    ax.plot(x, y, "o", markersize=5, color="orange")
    ax.plot(x_hats, y_hats, ":", label="Expected CPU time", color="orange")
    ax.legend()

    if bottom_trace:
        y_bots = y_hats - sigma_y_hats
        ax.plot(x_hats, y_bots, linewidth=2)

    # i = np.where(y_hat == y_hat.min())[0]

    # ax.axis([-.05, 1.05, -.1, 2])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
