#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# -------------------------
# Global animation settings
# -------------------------
colors = {
    "trajectory": "white",
    "tangent": "cyan",
    "normal": "yellow",
    "velocity": "magenta",
    "acceleration": "orange",
}

t_max = 20 * np.pi  # animate from t=0 to t=t_max
n_frames = 4000  # number of frames in the animation
interval_ms = 5  # delay between frames (milliseconds)
dpi_out = 300  # export DPI

# Visual scales
vec_len_T = 1.0  # fixed length for unit tangent
vec_len_N = 1.0  # fixed length for unit normal
vec_plot_scale_mag_per_unit = 1.0  # scale factor for velocity/acceleration vectors
max_vec_length = 10.0

# Axis and background colors
background_color = "black"
# axis_color = "#7082E9"  # light gray (change here as desired)
axis_color = "white"

# -------------------------
# Trajectory and kinematics
# -------------------------
tilt = 100
t = np.linspace(0.0, t_max, n_frames)
x = np.cos(t)
y = np.sin(t)
z = t**2 / tilt

# Velocity r'(t) and speed
vx = -np.sin(t)
vy = np.cos(t)
vz = 2.0 * t / tilt
speed = np.sqrt(vx**2 + vy**2 + vz**2)

# Unit tangent
Tx, Ty, Tz = vx / speed, vy / speed, vz / speed

# Acceleration r''(t)
ax_v = -np.cos(t)
ay_v = -np.sin(t)
az_v = 2.0 * np.ones_like(t) / tilt
acc_mag = np.sqrt(ax_v**2 + ay_v**2 + az_v**2)

# Unit normal (principal normal)
dTdt_x = np.gradient(Tx, t)
dTdt_y = np.gradient(Ty, t)
dTdt_z = np.gradient(Tz, t)
dTdt_norm = np.sqrt(dTdt_x**2 + dTdt_y**2 + dTdt_z**2)
dTdt_norm[dTdt_norm == 0] = 1.0
Nx, Ny, Nz = dTdt_x / dTdt_norm, dTdt_y / dTdt_norm, dTdt_z / dTdt_norm


# -------------------------
# Helper to add a vector segment
# -------------------------
def add_vector_segment(ax_obj, color):
    (seg,) = ax_obj.plot([], [], [], linewidth=2, color=color)
    return seg


# -------------------------
# Animation builder
# -------------------------
def build_and_save_animation(
    filename,
    show_tangent=False,
    show_normal=False,
    show_velocity=False,
    show_acceleration=False,
):
    fig = plt.figure(figsize=(8, 6))
    ax3d = fig.add_subplot(111, projection="3d", facecolor=background_color)
    fig.patch.set_facecolor(background_color)
    ax3d.set_proj_type("ortho")

    # Trajectory and current point
    (line,) = ax3d.plot([], [], [], color="white", linewidth=2)
    (point,) = ax3d.plot([], [], [], marker="o", ms=8, color="red")

    # Optional vectors
    tangent_line = add_vector_segment(ax3d, "cyan") if show_tangent else None
    normal_line = add_vector_segment(ax3d, "yellow") if show_normal else None
    vel_line = add_vector_segment(ax3d, "magenta") if show_velocity else None
    acc_line = add_vector_segment(ax3d, "orange") if show_acceleration else None

    # Labels, ticks, limits (unchanged)
    ax3d.set_xlabel("x = cos(t)", color=axis_color)
    ax3d.set_ylabel("y = sin(t)", color=axis_color)
    ax3d.set_zlabel("z = t²", color=axis_color)
    legend_items = [("Trajectory", colors["trajectory"])]
    if show_tangent:
        legend_items.append(("Tangent", colors["tangent"]))
    if show_normal:
        legend_items.append(("Normal", colors["normal"]))
    if show_velocity:
        legend_items.append(("Velocity", colors["velocity"]))
    if show_acceleration:
        legend_items.append(("Acceleration", colors["acceleration"]))
    add_color_legend(fig, legend_items, y=1.001)

    ax3d.set_xlim(-1.1, 1.1)
    ax3d.set_ylim(-1.1, 1.1)
    ax3d.set_zlim(0.0, np.max(z) * 1.05)
    ax3d.set_xticks([-1, 1])
    ax3d.set_yticks([-1, 1])
    ax3d.tick_params(colors=axis_color)
    ax3d.grid(False)
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.pane.fill = False
        axis.line.set_color(axis_color)
    ax3d.view_init(elev=25, azim=45)
    ax3d.set_box_aspect([1, 1, 1])
    # Set rear bounding box and grid colors

    def init():
        line.set_data_3d([], [], [])
        point.set_data_3d([], [], [])
        for ln in [tangent_line, normal_line, vel_line, acc_line]:
            if ln is not None:
                ln.set_data_3d([], [], [])
        artists = [line, point]
        for ln in [tangent_line, normal_line, vel_line, acc_line]:
            if ln is not None:
                artists.append(ln)
        return tuple(artists)

    def update(i):
        # Curve so far
        xi, yi, zi = x[: i + 1], y[: i + 1], z[: i + 1]
        line.set_data_3d(xi, yi, zi)

        # Current point
        px, py, pz = x[i], y[i], z[i]
        point.set_data_3d([px], [py], [pz])

        # Tangent (fixed length)
        if tangent_line is not None:
            qx, qy, qz = (
                px + vec_len_T * Tx[i],
                py + vec_len_T * Ty[i],
                pz + vec_len_T * Tz[i],
            )
            tangent_line.set_data_3d([px, qx], [py, qy], [pz, qz])

        # Normal (fixed length)
        if normal_line is not None:
            rx, ry, rz = (
                px + vec_len_N * Nx[i],
                py + vec_len_N * Ny[i],
                pz + vec_len_N * Nz[i],
            )
            normal_line.set_data_3d([px, rx], [py, ry], [pz, rz])

        # Velocity (scaled by magnitude + clamp)
        if vel_line is not None:
            vnorm = speed[i]
            if vnorm > 0:
                L = min(max_vec_length, vec_plot_scale_mag_per_unit * vnorm)
                vdx, vdy, vdz = vx[i] / vnorm, vy[i] / vnorm, vz[i] / vnorm
            else:
                L = 0
                vdx = vdy = vdz = 0
            pvx, pvy, pvz = px + L * vdx, py + L * vdy, pz + L * vdz
            vel_line.set_data_3d([px, pvx], [py, pvy], [pz, pvz])

        # Acceleration (scaled by magnitude + clamp)
        if acc_line is not None:
            anorm = acc_mag[i]
            if anorm > 0:
                L = min(max_vec_length, vec_plot_scale_mag_per_unit * anorm)
                adx, ady, adz = ax_v[i] / anorm, ay_v[i] / anorm, az_v[i] / anorm
            else:
                L = 0
                adx = ady = adz = 0
            pax, pay, paz = px + L * adx, py + L * ady, pz + L * adz
            acc_line.set_data_3d([px, pax], [py, pay], [pz, paz])

        artists = [line, point]
        for ln in [tangent_line, normal_line, vel_line, acc_line]:
            if ln is not None:
                artists.append(ln)
        return tuple(artists)

    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=interval_ms,
        blit=False,  # <- IMPORTANT: disable blitting in 3D
    )

    # Tight framing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.tight_layout(pad=0)
    plt.margins(0)

    ani.save(
        filename,
        writer="ffmpeg",
        fps=max(1, 1000 // interval_ms),
        dpi=dpi_out,
        savefig_kwargs={
            "bbox_inches": "tight",
            "pad_inches": 0,
            "facecolor": background_color,
        },
    )
    plt.close(fig)


def add_color_legend(fig, items, y=0.99):
    """
    items: list of (label, color)
    y: vertical anchor in figure coords (0..1); use >1 to push further above
    """
    handles = [Line2D([0], [0], lw=6, color=c) for _, c in items]
    labels = [lab for lab, _ in items]
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(labels),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )
    # color each label text
    for txt, (_, c) in zip(leg.get_texts(), items):
        txt.set_color(c)
        txt.set_fontweight("bold")
    return leg


# -------------------------
# Build all five variants
# -------------------------
configs = [
    ("trajectory.mp4", dict()),
    ("trajectory_tangent.mp4", dict(show_tangent=True)),
    ("trajectory_tangent_normal.mp4", dict(show_tangent=True, show_normal=True)),
    ("trajectory_velocity.mp4", dict(show_velocity=True)),
    (
        "trajectory_velocity_acceleration.mp4",
        dict(show_velocity=True, show_acceleration=True),
    ),
]

for fname, kwargs in configs:
    build_and_save_animation(fname, **kwargs)

print("Done: saved", ", ".join(fname for fname, _ in configs))
