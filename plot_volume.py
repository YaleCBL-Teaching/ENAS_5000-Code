#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable

# -----------------------------
# Field definition and rotation
# -----------------------------
def phi(x, y, z):
    """φ(x,y,z) = x^2 + y^2 – z^2"""
    return x**2 + y**2 - z**2

def rotate_z(x, y, theta_deg):
    """Rotate (x,y) by +theta_deg (CCW) around the z-axis."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr

def phi_rotated_z(x, y, z, theta_deg):
    """Evaluate φ at coordinates rotated by +theta_deg around z."""
    xr, yr = rotate_z(x, y, theta_deg)
    return phi(xr, yr, z)

# -----------------------------
# Plot helpers
# -----------------------------
def style_3d_axes(ax, lim=4):
    """Black background, white axes/ticks/labels, cube limits."""
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
    clip_x_min=-np.inf, clip_y_min=-np.inf, field_func=phi, alpha=0.9
):
    """
    Draw a colored plane (x=value, y=value, or z=value) within the cube.
    - xr, yr: parameter ranges for the mesh
    - clip_x_min, clip_y_min: half-box clipping (e.g., x>=0 or y>=0)
    - field_func: scalar field used to color faces
    - alpha: transparency for faces
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

    # Build masked arrays for geometry (plot_surface supports masked X/Y/Z)
    Xmasked = np.ma.array(X, mask=~mask)
    Ymasked = np.ma.array(Y, mask=~mask)
    Zmasked = np.ma.array(Z, mask=~mask)

    # Compute field and map to colors
    P = field_func(Xmasked, Ymasked, Zmasked)
    rgba = get_cmap(cmap)(norm(P.filled(np.nan)))  # shape (m,n,4)

    # IMPORTANT: facecolors must be a regular array (not masked).
    # Set alpha=0 where mask is False (i.e., outside the valid region).
    rgba[~mask] = (0.0, 0.0, 0.0, 0.0)
    rgba[..., 3] *= alpha  # scale alpha for the whole face

    surf = ax.plot_surface(
        Xmasked, Ymasked, Zmasked,
        facecolors=rgba,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False,
    )
    return surf

def add_phi_minus_one_isosurface(
    ax, lim, *, clip_x_min=-np.inf, clip_y_min=-np.inf,
    edge_color="white", face_alpha=0.20, edge_lw=0.6
):
    """
    Draw the isosurface φ = -1:
        x^2 + y^2 - z^2 = -1  =>  z = ±sqrt(x^2 + y^2 + 1)
    Clipped to the cube and optional half-box constraints.
    Render with a translucent white face and thin white edges.
    """
    n = 220
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y)
    Zabs = np.sqrt(X**2 + Y**2 + 1.0)

    for sign in (+1, -1):
        Z = sign * Zabs
        mask = (np.abs(Z) <= lim) & (X >= clip_x_min) & (Y >= clip_y_min)

        # Geometry as masked arrays (supported by plot_surface)
        Xp = np.ma.array(X, mask=~mask)
        Yp = np.ma.array(Y, mask=~mask)
        Zp = np.ma.array(Z, mask=~mask)

        # Facecolors as a regular ndarray; alpha=0 where invalid
        Fp = np.zeros(X.shape + (4,), dtype=float)
        Fp[..., 0] = 1.0  # R
        Fp[..., 1] = 1.0  # G
        Fp[..., 2] = 1.0  # B
        Fp[..., 3] = 0.0  # alpha default 0 (invisible)
        Fp[mask, 3] = face_alpha  # visible only where valid

        ax.plot_surface(
            Xp, Yp, Zp,
            rstride=1, cstride=1,
            facecolors=Fp,
            linewidth=edge_lw,
            edgecolor=edge_color,
            antialiased=False,
            shade=False,
        )

# -----------------------------
# Globals / normalization
# -----------------------------
lim = 4
cmap = "viridis"

# Build a global normalizer from samples on the box (covers both plots)
samp = np.linspace(-lim, lim, 5)
XX, YY, ZZ = np.meshgrid(samp, samp, samp)
P_all = phi(XX, YY, ZZ)
norm = Normalize(vmin=float(P_all.min()), vmax=float(P_all.max()))
sm = ScalarMappable(norm=norm, cmap=get_cmap(cmap))

# ============================================================
# Figure 1: Full box colored by φ -> full_box.pdf
# ============================================================
fig1 = plt.figure(figsize=(7.2, 6), facecolor="black")
ax1 = fig1.add_subplot(1, 1, 1, projection="3d")
style_3d_axes(ax1, lim=lim)
ax1.set_title("Full box colored by $\\varphi$", color="white", pad=12)

# Six faces of the cube
add_colored_face(ax1, "x", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)
add_colored_face(ax1, "x", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)
add_colored_face(ax1, "y", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)
add_colored_face(ax1, "y", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)
add_colored_face(ax1, "z", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)
add_colored_face(ax1, "z", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm, field_func=phi, alpha=1.0)

cbar1 = fig1.colorbar(sm, ax=ax1, shrink=0.75, pad=0.08)
cbar1.ax.tick_params(colors="white")
cbar1.set_label("$\\varphi(x,y,z)$", color="white")

# Robust margins for 3D
fig1.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
fig1.savefig("full_box.pdf", facecolor=fig1.get_facecolor(), bbox_inches="tight", dpi=300)

# =======================================================================
# Figure 2: Half box after 90° z-rotation (cut y>=0),
#           faces colored by rotated field; add φ=-1 isosurface
#           -> half_box_rotZ90.pdf
# =======================================================================
theta_deg = 90.0  # rotate field/cut by +90° about z
field_rot = lambda X, Y, Z: phi_rotated_z(X, Y, Z, theta_deg)

fig2 = plt.figure(figsize=(7.2, 6), facecolor="black")
ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
style_3d_axes(ax2, lim=lim)

# Half-box boundary faces for y ∈ [0, 4], plus x=±lim and z=±lim
# Use a bit of transparency so the isosurface remains visible
add_colored_face(ax2, "x", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)
add_colored_face(ax2, "x", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)
add_colored_face(ax2, "y", 0.0, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)  # cut face
add_colored_face(ax2, "y", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)
add_colored_face(ax2, "z", +lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)
add_colored_face(ax2, "z", -lim, (-lim, lim), (-lim, lim), lim, cmap, norm,
                 clip_y_min=0, field_func=field_rot, alpha=0.75)

# Draw the φ=-1 isosurface last so it appears on top where visible
add_phi_minus_one_isosurface(ax2, lim=lim, clip_y_min=0,
                             edge_color="white", face_alpha=0.20, edge_lw=0.6)

# Optional: tweak view to make the isosurface pop
ax2.view_init(elev=20, azim=-60)

cbar2 = fig2.colorbar(sm, ax=ax2, shrink=0.75, pad=0.08)
cbar2.ax.tick_params(colors="white")
cbar2.set_label("$\\varphi(x,y,z)$ (colored on faces)", color="white")

fig2.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
fig2.savefig("half_box_rotZ90.pdf", facecolor=fig2.get_facecolor(), bbox_inches="tight", dpi=300)

print("Saved: full_box.pdf, half_box_rotZ90.pdf")
