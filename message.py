import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

def pollards_kangaroo(g, h, p, a, b):
    """
    Pollard's Kangaroo Algorithm to solve g^x ≡ h (mod p) for x in [a, b].
    Assumes x_target (with h = g^(x_target) mod p) lies uniformly in [a, b].
    This implementation uses roughly O(sqrt(b-a)) steps.
    """
    d = b - a  # interval length
    # Deterministic jump table (you can adjust these values)
    jump_table = [1, 2, 4, 8, 16, 32]
    m = len(jump_table)
    
    def f(y):
        # Deterministic jump function: choose a jump based on y modulo table size.
        return jump_table[y % m]
    
    # --- Tame kangaroo: start at x = b ---
    T = int(2 * np.sqrt(d)) + 10   # number of tame steps
    tame_dict = {}
    x_tame = b
    y_tame = pow(g, x_tame, p)
    tame_dict[y_tame] = x_tame

    for _ in range(T):
        step = f(y_tame)
        x_tame += step
        y_tame = (y_tame * pow(g, step, p)) % p
        if y_tame not in tame_dict:
            tame_dict[y_tame] = x_tame

    # --- Wild kangaroo: start at h with x_wild = 0 ---
    T_wild = int(4 * np.sqrt(d)) + 10  # number of wild steps
    x_wild = 0
    y_wild = h
    for _ in range(T_wild):
        step = f(y_wild)
        x_wild += step
        y_wild = (y_wild * pow(g, step, p)) % p
        if y_wild in tame_dict:
            candidate = tame_dict[y_wild] - x_wild
            if pow(g, candidate, p) == h:
                return candidate
    return None

# ------------------------------
# Experiment Setup and Plotting
# ------------------------------

# Parameters for the discrete log problem
g = 3
p = 104729

# For each interval [a, b] with b = N (and a = 1), choose a random target x in [1, N].
# (On average, this gives a gap of ~N/2 for the wild kangaroo to cover.)
interval_sizes = [2**i for i in range(8, 20)]  # using larger N to see asymptotic behaviour
num_trials = 10  # number of trials per interval size

runtime_medians = {}

for N in interval_sizes:
    trial_runtimes = []
    a, b = 1, N
    # Choose x_target uniformly in [a, b]
    x_target = random.randint(a, b)
    h = pow(g, x_target, p)
    for _ in range(num_trials):
        start_time = time.perf_counter_ns()
        result = pollards_kangaroo(g, h, p, a, b)
        runtime = time.perf_counter_ns() - start_time
        trial_runtimes.append(runtime)
    runtime_medians[N] = np.median(trial_runtimes)
    print(f"Interval size: {N:7d}, Median Runtime: {runtime_medians[N]:.10f} ns, "
          f"Target x: {x_target}, Found x: {result}")

# Convert median runtime dictionary to numpy arrays
x_data = np.array(list(runtime_medians.keys()), dtype=float)
y_data = np.array(list(runtime_medians.values()), dtype=float)

# Define the power-law model: y = a * x^b
def model_func(x, a, b):
    return a * np.power(x, b)

# Fit the model to the data
params, _ = curve_fit(model_func, x_data, y_data)
a_fit, b_fit = params
equation = f"y = {a_fit:.2e} * x^{b_fit:.2f}"
print(f"Fitted equation: {equation}")

# Generate fitted curve data
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = model_func(x_fit, a_fit, b_fit)

# Plot the scatter data and fitted curve on a log–log scale (as before)
plt.figure()
plt.scatter(x_data, y_data, color='blue', label="Measured Runtime")
plt.plot(x_fit, y_fit, color='red', linestyle='--', label=equation)
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel("Interval size (N)")
plt.ylabel("Runtime (ns)")
plt.title("Pollard's Kangaroo Runtime vs. Interval Size (Log-Log Scale)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

# Plot the scatter data and fitted curve on a linear scale
plt.figure()
plt.scatter(x_data, y_data, color='blue', label="Measured Runtime")
plt.plot(x_fit, y_fit, color='red', linestyle='--', label=equation)
plt.xlabel("Interval size (N)")
plt.ylabel("Runtime (ns)")
plt.title("Pollard's Kangaroo Runtime vs. Interval Size (Linear Scale)")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
