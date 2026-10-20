"""Gaussian process emulator for any n-dimensional performance function."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from Performance_Functions import wave

g = wave              # performance function to emulate (see Performance_Functions.py)
n_dims = 1            # number of inputs
lower, upper = -5, 5  # range of every input
n_train = 25          # runs of the performance function to train the emulator
n_test = 12           # extra runs only to check the emulator
noise = 0             # random noise added to training samples
SEED = None           # random seed


def fit_gpe(g, n_dims, lower=-5, upper=5, n_train=25, noise=0, seed=None):
    """Train a GP on n_train uniform runs of g. Returns the fitted GaussianProcessRegressor."""
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(lower, upper, (n_train, n_dims))
    y_train = g(x_train) + noise * rng.standard_normal(n_train)

    kernel = ConstantKernel() * RBF(length_scale=np.ones(n_dims))
    if noise > 0:
        kernel = kernel + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10,
                                  random_state=rng.integers(2 ** 31))
    return gp.fit(x_train, y_train)


def emulator(gp):
    """The emulator as a performance function, usable anywhere g is (SuS, NSuS, EC)."""
    return gp.predict


# Plot 1 (n-dimensional):
def plot_validation(gp, g, n_test=12, lower=-5, upper=5, seed=None):
    x_test = np.random.default_rng(seed).uniform(lower, upper, (n_test, gp.n_features_in_))
    y_test = g(x_test)
    y_pred = gp.predict(x_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

    plt.figure(figsize=(6, 6))
    plt.plot(y_test, y_pred, "o")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect (y = x)")
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"Emulator validation (RMSE = {rmse:.3f})")
    plt.legend()
    return rmse


# Plot 2 (2D):
def plot_1d(gp, g, lower=-5, upper=5):
    x = np.linspace(lower, upper, 500)[:, None]
    mean, std = gp.predict(x, return_std=True)

    plt.figure(figsize=(10, 4))
    plt.fill_between(x[:, 0], mean - 1.96 * std, mean + 1.96 * std, color="0.85", label="95% confidence")
    plt.plot(x, g(x), label="True")
    plt.plot(x, mean, "--", label="Predicted")
    plt.plot(gp.X_train_, g(gp.X_train_), "ko", label="Training points")
    plt.legend()


# Plot 2 (3D):
def plot_2d(gp, g, lower=-5, upper=5):
    grid_1d = np.linspace(lower, upper, 50)
    X1, X2 = np.meshgrid(grid_1d, grid_1d)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    z_pred = gp.predict(grid).reshape(X1.shape)
    error = np.abs(z_pred - g(grid).reshape(X1.shape))

    colours = plt.Normalize(0, error.max())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, z_pred, facecolors=plt.cm.RdYlGn_r(colours(error)))
    fig.colorbar(plt.cm.ScalarMappable(colours, "RdYlGn_r"), ax=ax, shrink=0.6, pad=0.1, label="Absolute error")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Predicted value")
    ax.set_title("Emulator prediction, coloured by error")


if __name__ == "__main__":
    gp = fit_gpe(g, n_dims, lower, upper, n_train, noise, SEED)
    print("kernel:", gp.kernel_)
    print("RMSE  :", plot_validation(gp, g, n_test, lower, upper))
    if n_dims == 1:
        plot_1d(gp, g, lower, upper)
    if n_dims == 2:
        plot_2d(gp, g, lower, upper)
    plt.show()
