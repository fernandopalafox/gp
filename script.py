import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0
mu = 0.5
scale = 1.0
s_0 = [0.0, 1.0]
noise = 1e-3
t_range = (0, 10)
ts = np.linspace(t_range[0], t_range[1], 10000)
train_indices = np.sort(np.random.choice(len(ts), 50, replace=False))
ts_test = np.sort(np.random.uniform(t_range[0], t_range[1] + 2, 100))

# Dynamical system
def dynamics(x: np.ndarray, xd: np.ndarray) -> np.ndarray:
    # xdd = -alpha * xd**2  # drag
    xdd = mu * (1 - x**2) * xd - x # VDP oscillator
    return xdd

# Kernel
def rbf(xi: np.ndarray, xj: np.ndarray, scale: float) -> np.ndarray:
    xi = xi.reshape(-1, 1)
    xj = xj.reshape(-1, 1)
    return np.exp(-np.abs(xi - xj.T)**2 / (2 * scale**2))

# Training data
# Numerically integrate the dynamics
xs = np.zeros_like(ts)
xds = np.zeros_like(ts)
xdds = np.zeros_like(ts)
xs[0] = s_0[0]
xds[0] = s_0[1]
for i in range(1, len(ts)):
    dt = ts[i] - ts[i - 1]
    xdds[i] = dynamics(xs[i - 1], xds[i - 1])
    xds[i] = xds[i - 1] + xdds[i] * dt
    xs[i] = xs[i - 1] + xds[i] * dt

# xs = np.sin(ts)
# xds = np.cos(ts)

# Plot phase
plt.figure()
plt.plot(xs, xds)
plt.xlabel("x")
plt.ylabel("xd")
plt.savefig("phase.png")

# Gaussian Process
# Function that predicts position at a given time using a Gaussian Process
def gp_predict(
    xs_train: np.ndarray, ys_train: np.ndarray, xs_test, scale: float
) -> np.ndarray:
    K = rbf(xs_train, xs_train, scale) + noise * np.eye(len(xs_train))
    K_inv = np.linalg.inv(K)
    K_star = rbf(xs_train, xs_test, scale)
    K_star_star = rbf(xs_test, xs_test, scale) + noise * np.eye(len(xs_test))
    mu_star = K_star.T @ K_inv @ ys_train
    cov_star = K_star_star - K_star.T @ K_inv @ K_star
    return mu_star, cov_star

# Test the GP
ts_train = ts[train_indices]
xs_train = xs[train_indices]
mu_predict, cov_predict = gp_predict(ts_train, xs_train, ts_test, scale)

# Plot the results
plt.figure()
plt.plot(ts, xs, label="True")
plt.plot(ts_train, xs_train, "o", label="Train")
plt.plot(ts_test, mu_predict, "x", label="Predict")
plt.fill_between(
    ts_test,
    mu_predict - np.sqrt(np.diag(cov_predict)),
    mu_predict + np.sqrt(np.diag(cov_predict)),
    alpha=0.5,
)
plt.legend()

plt.savefig("gp.png")