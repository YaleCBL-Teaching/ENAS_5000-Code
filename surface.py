#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- Parameters ----------
# Domain: [-2, 2] x [-2, 2]
x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0
n = 300                   # surface resolution
qstep = 32                # stride for 3D gradient vectors (even fewer arrows)
qstep2d = 10              # stride for 2D/top-down quiver (even fewer arrows)
elev, azim = 30, -45      # 3D camera for the first three plots
cmap = plt.get_cmap("viridis")
phi_level = 0.03          # contour level

# Arrow scaling:
# We keep vector components in data units so length == |∇φ|.
# If you want globally longer/shorter arrows while preserving "length ∝ |∇φ|",
# tweak these multipliers (1.0 = true magnitude).
arrow_scale_3d = 1.0
arrow_scale_2d = 0.05

# ---------- Define φ(x,y) and ∇φ ----------
def phi(x, y):
    r2 = x**2 + y**2
    return np.sin(r2) + 0.5*np.cos(2*x - 2)*np.sin(y) + 0.2*np.cos(y + 1)

def grad_phi(x, y):
    r2 = x**2 + y**2
    dphidx = 2*x*np.cos(r2) - np.sin(2*x - 2)*np.sin(y)
    dphidy = 2*y*np.cos(r2) + 0.5*np.cos(2*x - 2)*np.cos(y) - 0.2*np.sin(y + 1)
    return dphidx, dphidy

# ---------- Grid ----------
x = np.linspace(x_min, x_max, n)
y = np.linspace(y_min, y_max, n)
X, Y = np.meshgrid(x, y, indexing="xy")
Z = phi(X, Y)
vmin, vmax = np.min(Z), np.max(Z)
Zrange = vmax - vmin

# ---------- Styling helpers ----------
def style_axes_black_3d(ax):
    fig = ax.figure
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
    ax.grid(False)
    # Try to set axis lines to white (version-dependent)
    for line in getattr(ax, "w_xaxis", getattr(ax, "xaxis", None)), \
                getattr(ax, "w_yaxis", getattr(ax, "yaxis", None)), \
                getattr(ax, "w_zaxis", getattr(ax, "zaxis", None)):
        try:
            line.line.set_color("white")
        except Exception:
            pass
    try:
        ax.xaxis.pane.set_facecolor((0,0,0,1))
        ax.yaxis.pane.set_facecolor((0,0,0,1))
        ax.zaxis.pane.set_facecolor((0,0,0,1))
        ax.xaxis.pane.set_edgecolor("white")
        ax.yaxis.pane.set_edgecolor("white")
        ax.zaxis.pane.set_edgecolor("white")
    except Exception:
        pass
    fig.tight_layout()

def style_axes_black_2d(ax):
    fig = ax.figure
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(False)
    fig.tight_layout()

def colorbar_on_black(fig, mappable, label="φ(x, y)"):
    cb = fig.colorbar(mappable, shrink=0.75, pad=0.05)
    cb.outline.set_edgecolor("white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")
    cb.set_label(label, color="white")
    return cb

# ---------- Figure 1: surface ----------
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection="3d")
style_axes_black_3d(ax1)
surf = ax1.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
colorbar_on_black(fig1, surf)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("φ(x, y)")
ax1.view_init(elev=elev, azim=azim)
fig1.savefig("surface.pdf", dpi=300, facecolor=fig1.get_facecolor(), edgecolor="none")
plt.close(fig1)

# ---------- Figure 4: TOP-DOWN φ(x,y) (2D) with red contour on top ----------
fig4 = plt.figure(figsize=(7, 7))
ax4 = fig4.add_subplot(111)
style_axes_black_2d(ax4)

im = ax4.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
colorbar_on_black(fig4, im, label="φ(x, y)")

# Red level-set overlaid with high zorder so it's not hidden
ax4.contour(X, Y, Z, levels=[phi_level], colors="red", linewidths=2.2, zorder=5)

ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_aspect("equal", "box")
fig4.savefig("surface_topdown.pdf", dpi=300, facecolor=fig4.get_facecolor(), edgecolor="none")
plt.close(fig4)

# ---------- Figure 5: TOP-DOWN gradient field (2D; fewer points; length = |∇φ|) ----------
fig5 = plt.figure(figsize=(7, 7))
ax5 = fig5.add_subplot(111)
style_axes_black_2d(ax5)

# Background: same φ(x,y) map for context
im2 = ax5.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
colorbar_on_black(fig5, im2, label="φ(x, y)")

Xs2 = X[::qstep2d, ::qstep2d]
Ys2 = Y[::qstep2d, ::qstep2d]
Ux2, Uy2 = grad_phi(Xs2, Ys2)

# Keep 2D arrows in data units: scale=1.0, scale_units='xy' -> length equals ||∇φ|| * arrow_scale_2d
ax5.quiver(
    Xs2, Ys2,
    arrow_scale_2d * Ux2, arrow_scale_2d * Uy2,
    color="red",
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.004
)

ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.set_aspect("equal", "box")
fig5.savefig("surface_topdown_gradient.pdf", dpi=300, facecolor=fig5.get_facecolor(), edgecolor="none")
plt.close(fig5)
