import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import minimize

# Base parameters
n = 100  # number of segments
s, h = np.linspace(0, 1, n + 1, retstep=True)

def energy_functional(theta, p, h, c, penalty_coeff1, penalty_coeff2):
    n = len(theta) - 1
    diffs = np.diff(theta)
    v = 0.5 * np.sum(diffs ** 2) + p / (n + 1) ** 2 * np.sum(np.cos(theta))
    dy = h * np.sin(theta[:n])
    y = np.concatenate(([0], np.cumsum(dy)))
    penalty = penalty_coeff1 * np.sum(np.maximum(0, y ** 2 - c ** 2)) + penalty_coeff2 * y[-1] ** 2
    return v + penalty

def run_optimization(p, c, penalty_coeff1, penalty_coeff2):
    theta0 = c / 0.35 * np.linspace(-1, 1, n + 1)
    res = minimize(
        fun=energy_functional,
        x0=theta0,
        args=(p, h, c, penalty_coeff1, penalty_coeff2),
        method="BFGS",
        options={"disp": False}
    )
    return res.x, theta0

def compute_shape(theta):
    x = [0]
    y = [0]
    for i in range(n):
        x.append(x[-1] + h * np.cos(theta[i]))
        y.append(y[-1] + h * np.sin(theta[i]))
    return np.array(x), np.array(y)

# Initial values
init_vals = {
    "p": 100.0,
    "c": 0.01,
    "penalty_coeff1": 1e2,
    "penalty_coeff2": 1e2
}

# Create plot layout
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.35)

# Initial computation
res_theta, theta0 = run_optimization(**init_vals)
x_opt, y_opt = compute_shape(res_theta)
x_init, y_init = compute_shape(theta0)

# Plots
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

# Slider Axes
ax_p = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_c = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_pen1 = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_pen2 = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_button = plt.axes([0.4, 0.025, 0.2, 0.05])

# Sliders
slider_p = Slider(ax_p, "p", 1, 500, valinit=init_vals["p"])
slider_c = Slider(ax_c, "c", 0.001, 0.1, valinit=init_vals["c"])
slider_pen1 = Slider(ax_pen1, "Penalty 1", 1, 1e4, valinit=init_vals["penalty_coeff1"])
slider_pen2 = Slider(ax_pen2, "Penalty 2", 1, 1e4, valinit=init_vals["penalty_coeff2"])

# Button
button = Button(ax_button, "Update")

# Callback
def update(val=None):
    p = slider_p.val
    c = slider_c.val
    pen1 = slider_pen1.val
    pen2 = slider_pen2.val
    res_theta, theta0 = run_optimization(p, c, pen1, pen2)
    x_opt, y_opt = compute_shape(res_theta)
    x_init, y_init = compute_shape(theta0)

    line_opt.set_xdata(x_opt)
    line_opt.set_ydata(y_opt)
    line_init.set_xdata(x_init)
    line_init.set_ydata(y_init)
    for ax in axs:
        ax.lines[1].set_ydata([c] * 2)
        ax.lines[2].set_ydata([-c] * 2)
        ax.relim()
        ax.autoscale_view()
    fig.canvas.draw_idle()

# Bind update to button
button.on_clicked(update)

plt.show()

