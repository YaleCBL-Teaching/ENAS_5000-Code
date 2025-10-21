#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# ---- Parameters you can tweak ----
t_max = 8 * np.pi    # animate from t=0 to t=t_max
n_frames = 1000      # number of frames in the animation
interval_ms = 5      # delay between frames (milliseconds)
# ----------------------------------

# Parameter and trajectory
t = np.linspace(0.0, t_max, n_frames)
x = np.cos(t)
y = np.sin(t)
z = t**2

# Figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d', facecolor='black')
fig.patch.set_facecolor('black')

# Pre-create artists: the growing trajectory and the current point
(line,) = ax.plot([], [], [], color='white', linewidth=2)     # trajectory so far
(point,) = ax.plot([], [], [], marker='o', ms=8, color='red')  # current point

# Axis labels and title (white text)
ax.set_xlabel('x = cos(t)', color='white')
ax.set_ylabel('y = sin(t)', color='white')
ax.set_zlabel('z = t²', color='white')
# ax.set_title('Animating (cos(t), sin(t), t²) from t=0 → current t', color='white')

# Fix axis limits
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(0.0, (t_max**2) * 1.05)

# Restrict tick labels on x and y to -1 and 1
ax.set_xticks([-1, 1])
ax.set_yticks([-1, 1])

# Set view
ax.view_init(elev=25, azim=45)

# White tick labels and turn off panes/grid
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.grid(False)

# Remove background panes (transparent)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.fill = False
    axis.line.set_color('white')  # axis line visible in white

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(i):
    xi, yi, zi = x[: i + 1], y[: i + 1], z[: i + 1]
    line.set_data(xi, yi)
    line.set_3d_properties(zi)
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])
    return line, point

ani = FuncAnimation(
    fig, update, frames=n_frames, init_func=init, interval=interval_ms, blit=True
)

# Save to video (requires ffmpeg installed)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.tight_layout(pad=0)
plt.margins(0)
ani.save('trajectory.mp4', writer='ffmpeg', fps=1000//interval_ms)
