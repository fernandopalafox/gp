import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend
import matplotlib.pyplot as plt

np.random.seed(0)

xs_sample = np.linspace(-5, 5, 100)

# Kernel
def rbf(xi: np.ndarray, xj: np.ndarray, scale: float) -> np.ndarray:
    xi = xi.reshape(-1, 1)
    xj = xj.reshape(-1, 1)
    return np.exp(-(xi - xj.T) ** 2 / (2 * scale**2))

# Sample from the prior
def sample_prior(xs: np.ndarray, scale: float, num_samples: int) -> np.ndarray:
    K = rbf(xs, xs, scale) + 1e-15 * np.eye(len(xs))
    L = np.linalg.cholesky(K)
    samples = L @ np.random.normal(size=(len(xs), num_samples))
    return samples.T

# Plot the samples
num_samples = 5
samples = sample_prior(xs_sample, 1.0, num_samples)
for i in range(num_samples):
    plt.plot(xs_sample, samples[i], label=f"Sample {i+1}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("figures/prior_samples.png")