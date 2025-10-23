#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

# ---------- Field definition ----------
def phi(x, y, z):
    # φ(x,y,z) = x^2 + y^2 – z^2
    return x**2 + y**2 - z**2

def rotate_z(x, y, theta_deg):
    """Rotate coordinates (x,y) by +theta_deg around z (CCW)."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr

def phi_rotated_z(x, y, z, theta_deg):
    """Evaluate φ at coordinates rotated by +theta around z."""
    xr, yr = rotate_z(x, y, theta_deg)
    return phi(xr, yr, z)

# ---------- Plot helpers ----------
def style_3d_axes(ax, lim=4):
    # Black background & white ticks/labels/axes
    ax.set_facecolor("black")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    # Dark panes + faint white grid
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1.0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")

    # Draw axes lines in white
    L = lim
    ax.plot([-L, L], [0, 0], [0, 0], lw=1, color="white")
    ax.plot([0, 0], [-L, L], [0, 0], lw=1, color="white")
    ax.plot([0, 0], [0, 0], [-L, L], lw=1, color="white")

def add_colored_face(
    ax, plane, value, xr, yr, lim, cmap, norm, *,
    clip_x_min=-np.inf, clip_y_min=-np.inf, field_func=phi
):
    """
    plane: 'x', 'y', or 'z' indicating constant plane
    value: coordinate value of the plane (e.g., x=+lim)
    xr, yr: tuples defining parameter ranges for the mesh
    lim: axes limit for masking
    cmap, norm: colormap and normalizer
    clip_x_min, clip_y_min: optional half-box clipping
    field_func: function f(x,y,z) to color with (phi or rotated variant)
    """
    u = np.linspace(xr[0], xr[1], 120)
    v = np.linspace(yr[0], yr[1], 120)
    U, V = np.meshgrid(u, v)

    if plane == "x":
        X, Y, Z = np.full_like(U, value), U, V
    elif plane == "y":
        X, Y, Z = U, np.full_like(U, value), V
    elif plane == "z":
        X, Y, Z = U, V, np.full_like(U, value)
    else:
        raise ValueError("plane must be 'x', 'y', or 'z'")

    mask = (
        (np.abs(X) <= lim) &
        (np.abs(Y) <= lim) &
        (np.abs(Z) <= lim) &
        (X >= clip_x_min) &
        (Y >= clip_y_min)
    )

    if not np.any(mask):
        return None

    Xmasked = np.ma.array(X, mask=~mask)
    Ymasked = np.ma.array(Y, mask=~mask)
    Zmasked = np.ma.array(Z, mask=~mask)

    P = field_func(Xmasked, Ymasked, Zmasked)
    facecolors = get_cmap(cmap)(norm(P))

    surf = ax.plot_surface(
        Xmasked, Ymasked, Zmasked,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False
    )
    return surf

def add_phi_minus_one_isosurface(
    ax, lim, *, clip_x_min=-np.inf, clip_y_min=-np.inf,
    edge_color="white", alpha=0.9
):
    """
    φ = -1 => x^2 + y^2 - z^2 = -1  =>  z = ±sqrt(x^2 + y^2 + 1)
    Clipped to the box and optional half-box conditions.
    """
    n = 200
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y)
    Zabs = np.sqrt(X**2 + Y**2 + 1.0)

    for sign in (+1, -1):
        Z = sign * Zabs
        mask = (np.abs(Z) <= lim) & (X >= clip_x_min) & (Y >= clip_y_min)
        Xp = np.ma.array(X, mask=~mask)
        Yp = np.ma.array(Y, mask=~mask)
        Zp = np.ma.array(Z, mask=~mask)
        ax.plot_surface(
            Xp, Yp, Zp,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            facecolors=None, edgecolor=edge_color, alpha=alpha
        )

# ---------- Globals ----------
lim = 4
cmap = "viridis"

# Build a global normalizer from samples on the box (covers both plots)
samp = np.linspace(-lim, lim, 5)
XX, YY, ZZ = np.meshgrid(samp, samp, samp)
P_all = phi(XX, YY, ZZ)
norm = Normalize(vmin=P_all.min(), vmax=P_all.max())
sm = ScalarMappable(norm=norm, cmap=get_cmap(cmap))

# ============================================================
# Figure 1: Full box colored by φ, saved to full_box.pdf
# ============================================================
fig1 = plt.figure(figsize=(7.2, 6), facecolor="black")
ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
style_3d_axes(ax1, lim=lim)

# Six faces of the cube
add_colored_face(ax1, "x", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)
add_colored_face(ax1, "x", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)
add_colored_face(ax1, "y", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)
add_colored_face(ax1, "y", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)
add_colored_face(ax1, "z", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)
add_colored_face(ax1, "z", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi)

cbar1 = fig1.colorbar(sm, ax=ax1, shrink=0.75, pad=0.08)
cbar1.ax.tick_params(colors="white")
cbar1.set_label("$\\varphi(x,y,z)$", color="white")

fig1.tight_layout()
fig1.savefig("volume_full.pdf", facecolor=fig1.get_facecolor(), bbox_inches="tight", dpi=300)

# =======================================================================
# Figure 2: Half box after 90° z-rotation, with φ=-1 isosurface, to PDF
# =======================================================================
# Interpret "rotate by 90° along z" as rotating the cut: x>=0 -> y>=0,
# and color faces using the field evaluated at coordinates rotated by +90°.
theta_deg = 90.0

fig2 = plt.figure(figsize=(7.2, 6), facecolor="black")
ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
style_3d_axes(ax2, lim=lim)

# Half-box boundary faces for y ∈ [0, 4], plus x=±lim and z=±lim
# Use rotated field for coloring
field_rot = lambda X, Y, Z: phi_rotated_z(X, Y, Z, theta_deg)

add_colored_face(ax2, "x", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)
add_colored_face(ax2, "x", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)
add_colored_face(ax2, "y", 0.0, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)  # the cut face
add_colored_face(ax2, "y", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)
add_colored_face(ax2, "z", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)
add_colored_face(ax2, "z", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot)

# Add φ = -1 isosurface, clipped to y >= 0
add_phi_minus_one_isosurface(ax2, lim=lim, clip_y_min=0, edge_color="white", alpha=0.9)

cbar2 = fig2.colorbar(sm, ax=ax2, shrink=0.75, pad=0.08)
cbar2.ax.tick_params(colors="white")
cbar2.set_label("$\\varphi(x,y,z)$ (colored on faces)", color="white")

fig2.tight_layout()
fig2.savefig("volume_half.pdf", facecolor=fig2.get_facecolor(), bbox_inches="tight", dpi=300)
