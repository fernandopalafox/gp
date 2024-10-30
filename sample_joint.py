import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Parameters
alpha = 1.0
scale = 1.0
noise = 1e-6
t_range = (-5, 5)
ts_test = np.linspace(t_range[0], t_range[1], 1000)

# Kernel
def rbf(xi: np.ndarray, xj: np.ndarray, scale: float) -> np.ndarray:
    xi = xi.reshape(-1, 1)
    xj = xj.reshape(-1, 1)
    return np.exp(-np.abs(xi - xj.T) ** 2 / (2 * scale**2))


# Training data
# Random points
ts_train = np.random.uniform(-4.5, 4.5, 5)
xs_train = np.random.normal(0, 1, len(ts_train))

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
mu_predict, cov_predict = gp_predict(ts_train, xs_train, ts_test, scale)

# Sample from the joint distribution
n_samples = 5
f_star = np.random.multivariate_normal(mu_predict, cov_predict, n_samples)

# Plot the results
plt.figure()
for i in range(n_samples):
    plt.plot(ts_test, f_star[i])
plt.fill_between(
    ts_test,
    mu_predict - 2*np.sqrt(np.diag(cov_predict)),
    mu_predict + 2*np.sqrt(np.diag(cov_predict)),
    alpha=0.5,
)
plt.plot(ts_train, xs_train, "x", color='black', markersize=10, markeredgewidth=2)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("figures/posterior_samples.png")