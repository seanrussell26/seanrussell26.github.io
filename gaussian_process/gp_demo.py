"""Gaussian process emulator for any n-dimensional performance function."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# ---- Your settings ----------------------------------------------------------
def performance_function(x):
    """x has shape (n_points, n_dims). Return one value per point."""
    return np.sin(3 * x[:, 0]) * np.cos(3 * x[:, 1])

n_dims = 2            # number of inputs
lower, upper = -1, 1  # range of every input
n_train = 30          # runs of the performance function used to train the emulator
n_test = 50           # extra runs used only to check the emulator
# -----------------------------------------------------------------------------

np.random.seed(42)
x_train = np.random.uniform(lower, upper, (n_train, n_dims))
x_test = np.random.uniform(lower, upper, (n_test, n_dims))
y_train = performance_function(x_train)
y_test = performance_function(x_test)

kernel = ConstantKernel() * RBF(length_scale=np.ones(n_dims))
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)
gp.fit(x_train, y_train)

# Plot 1 (any number of inputs): predicted vs actual on the test points
y_pred = gp.predict(x_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

plt.figure(figsize=(6, 6))
plt.plot(y_test, y_pred, "o")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", label="Perfect (y = x)")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.title(f"Emulator validation (RMSE = {rmse:.3f})")
plt.legend()

# Plot 2 (1 input only): prediction with 95% confidence band
if n_dims == 1:
    x = np.linspace(lower, upper, 500)[:, None]
    mean, std = gp.predict(x, return_std=True)

    plt.figure(figsize=(10, 4))
    plt.fill_between(x[:, 0], mean - 1.96 * std, mean + 1.96 * std, color="0.85", label="95% confidence")
    plt.plot(x, performance_function(x), label="True")
    plt.plot(x, mean, "--", label="Predicted")
    plt.plot(x_train, y_train, "ko", label="Training points")
    plt.legend()

# Plot 2 (2 inputs only): 3D surface of the prediction, coloured by its error
if n_dims == 2:
    g = np.linspace(lower, upper, 50)
    X1, X2 = np.meshgrid(g, g)
    grid = np.column_stack([X1.ravel(), X2.ravel()])
    z_pred = gp.predict(grid).reshape(X1.shape)
    error = np.abs(z_pred - performance_function(grid).reshape(X1.shape))

    colours = plt.Normalize(0, error.max())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X1, X2, z_pred, facecolors=plt.cm.RdYlGn_r(colours(error)))
    fig.colorbar(plt.cm.ScalarMappable(colours, "RdYlGn_r"), ax=ax, shrink=0.6, pad=0.1, label="Absolute error")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Predicted value")
    ax.set_title("Emulator prediction, coloured by error")

plt.show()
