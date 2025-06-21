import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.optimize import minimize

# Function to define energy
def energy_functional(theta, p, h, c, penalty_coeff1, penalty_coeff2):
    n = len(theta) - 1
    diffs = np.diff(theta)
    v = 0.5 * np.sum(diffs ** 2) + p / (n + 1) ** 2 * np.sum(np.cos(theta))
    dy = h * np.sin(theta[:n])
    y = np.concatenate(([0], np.cumsum(dy)))
    penalty = penalty_coeff1 * np.sum(np.maximum(0, y ** 2 - c ** 2)) + penalty_coeff2 * y[-1] ** 2
    return v + penalty

# Optimization wrapper
def run_optimization(n, p, c, penalty_coeff1, penalty_coeff2):
    s, h = np.linspace(0, 1, n + 1, retstep=True)
    theta0 = c / 0.35 * np.linspace(-1, 1, n + 1)
    res = minimize(
        fun=energy_functional,
        x0=theta0,
        args=(p, h, c, penalty_coeff1, penalty_coeff2),
        method="BFGS",
        options={"disp": False}
    )
    return res.x, theta0, h

# Reconstruct (x, y) from theta
def compute_shape(theta, h):
    x = [0]
    y = [0]
    for i in range(len(theta) - 1):
        x.append(x[-1] + h * np.cos(theta[i]))
        y.append(y[-1] + h * np.sin(theta[i]))
    return np.array(x), np.array(y)

# Initial parameter values
init_vals = {
    "n": 100,
    "p": 100.0,
    "c": 0.01,
    "penalty_coeff1": 1e2,
    "penalty_coeff2": 1e2
}

# Initial run
res_theta, theta0, h = run_optimization(**init_vals)
x_opt, y_opt = compute_shape(res_theta, h)
x_init, y_init = compute_shape(theta0, h)

# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.4)

line_opt, = axs[0].plot(x_opt, y_opt)
line_init, = axs[1].plot(x_init, y_init)
for ax in axs:
    ax.axhline(init_vals["c"], color='r', linestyle="dotted")
    ax.axhline(-init_vals["c"], color='r', linestyle="dotted")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
axs[0].set_title("Optimized Shape")
axs[1].set_title("Initial Guess")

# Input boxes
box_width = 0.1
textboxes = {}
params = ['n', 'p', 'c', 'penalty_coeff1', 'penalty_coeff2']
for i, param in enumerate(params):
    axbox = plt.axes([0.15, 0.23 - i*0.05, box_width, 0.04])
    textboxes[param] = TextBox(axbox, f"{param}:", initial=str(init_vals[param]))

# Button
ax_button = plt.axes([0.4, 0.18, 0.2, 0.05])
button = Button(ax_button, "Update")

# Update function
def update(event):
    try:
        n = int(float(textboxes["n"].text))
        p = float(textboxes["p"].text)
        c = float(textboxes["c"].text)
        penalty_coeff1 = float(textboxes["penalty_coeff1"].text)
        penalty_coeff2 = float(textboxes["penalty_coeff2"].text)
        if n <= 1:
            print("n must be greater than 1.")
            return
    except ValueError:
        print("Invalid input.")
        return

    res_theta, theta0, h = run_optimization(n, p, c, penalty_coeff1, penalty_coeff2)
    x_opt, y_opt = compute_shape(res_theta, h)
    x_init, y_init = compute_shape(theta0, h)

    line_opt.set_xdata(x_opt)
    line_opt.set_ydata(y_opt)
    line_init.set_xdata(x_init)
    line_init.set_ydata(y_init)

    for ax in axs:
        # Update constraint lines
        if len(ax.lines) >= 3:
            ax.lines[1].set_ydata([c] * 2)
            ax.lines[2].set_ydata([-c] * 2)
        ax.relim()
        ax.autoscale_view()
    fig.canvas.draw_idle()

# Bind button
button.on_clicked(update)
plt.show()