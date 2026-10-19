"""Simple Gaussian process emulator demo with validation plots (scikit-learn).

1. 1D input:  prediction with a 95% confidence band
2. 2D input:  heat maps of true function, prediction, and absolute error
3. nD input:  predicted vs actual on held-out points (closer to y = x is better)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

np.random.seed(42)


def fit_gp(x_train, y_train):
    """Fit a GP with one length scale per input dimension."""
    n_dims = x_train.shape[1]
    kernel = ConstantKernel() * RBF(length_scale=np.ones(n_dims)) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
    gp.fit(x_train, y_train)
    return gp


# ---------------------------------------------------------------- 1D example
def f1(x):
    return np.exp(-0.7 * x) * np.sin(2 * np.pi * x / 0.9)

x = np.linspace(0, 2, 2000)
x_train = np.random.uniform(0, 2, (10, 1))
y_train = f1(x_train[:, 0]) + np.random.randn(10) * 0.02

gp = fit_gp(x_train, y_train)
y_pred, y_std = gp.predict(x[:, None], return_std=True)

plt.figure(figsize=(12, 4))
plt.fill_between(x, y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, color="0.8", label="95% confidence")
plt.plot(x, f1(x), "-", label="True")
plt.plot(x, y_pred, "--", lw=2, label="Predicted")
plt.plot(x_train, y_train, "ko", label="Training points")
plt.legend(loc="best")
plt.title("1D Gaussian process emulator")


# ---------------------------------------------------------------- 2D example
def f2(x1, x2):
    return np.sin(3 * x1) * np.cos(3 * x2)

x_train = np.random.uniform(-1, 1, (30, 2))
y_train = f2(x_train[:, 0], x_train[:, 1])
gp = fit_gp(x_train, y_train)

g = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(g, g)
grid = np.column_stack([X1.ravel(), X2.ravel()])
y_pred = gp.predict(grid).reshape(X1.shape)
y_true = f2(X1, X2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [(y_true, "True"), (y_pred, "Predicted"), (np.abs(y_pred - y_true), "Absolute error")]
for ax, (z, title) in zip(axes, panels):
    im = ax.contourf(X1, X2, z, levels=30, cmap="viridis")
    ax.plot(x_train[:, 0], x_train[:, 1], "r.", label="Training points")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(im, ax=ax)
axes[0].legend(loc="upper right")
fig.suptitle("2D Gaussian process emulator")
fig.tight_layout()


# ---------------------------------------------------------------- nD example
def f5(x):
    return np.sin(x[:, 0]) + x[:, 1] ** 2 - x[:, 2] * x[:, 3] + 0.5 * np.cos(2 * x[:, 4])

n_dims = 5
x_train = np.random.uniform(-1, 1, (40, n_dims))
x_test = np.random.uniform(-1, 1, (50, n_dims))
gp = fit_gp(x_train, f5(x_train))

y_true = f5(x_test)
y_pred, y_std = gp.predict(x_test, return_std=True)
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

plt.figure(figsize=(6, 6))
plt.errorbar(y_true, y_pred, yerr=1.96 * y_std, fmt="o", capsize=3, label="Test points (95% CI)")
lims = [y_true.min(), y_true.max()]
plt.plot(lims, lims, "k--", label="y = x (perfect)")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title(f"{n_dims}D emulator validation (RMSE = {rmse:.3f})")
plt.legend(loc="best")

plt.show()
