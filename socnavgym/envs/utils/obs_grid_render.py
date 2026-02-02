# socnavgym/envs/utils/obs_grid_render.py

import numpy as np
import cv2


def _edges_for_render(max_abs: float, bins: int):
    """
    Build edges for digitize/rendering as BIN BOUNDARIES (length bins+1):
      [-max_abs ... +max_abs]
    Your discretizer uses internal 'theta_edges' and '_bin_scalar' boundaries of length (bins-1),
    but for rendering a grid we want full cell boundaries.
    """
    bins = int(bins)
    if bins < 2:
        bins = 2
    return np.linspace(-max_abs, max_abs, num=bins + 1, dtype=np.float32)


def _bin_to_cell_center(ix: int, edges_full: np.ndarray) -> float:
    """Return center coordinate of bin ix given full boundaries (len = bins+1)."""
    ix = int(ix)
    ix = max(0, min(ix, len(edges_full) - 2))
    return float(0.5 * (edges_full[ix] + edges_full[ix + 1]))


def _draw_robot_glyph_centered(img, env, extent_m: float):
    """
    Reuse Robot.draw to draw the exact same robot glyph as in world render,
    but centered in a local map where world coordinates are in [-extent, +extent].
    """
    H, W = img.shape[:2]
    map_size_x = 2.0 * float(extent_m)
    map_size_y = 2.0 * float(extent_m)

    # Robot.draw expects px-per-meter (despite naming in codebase)
    px_per_m_x = W / map_size_x
    px_per_m_y = H / map_size_y

    # Save current pose
    rx, ry = float(env.robot.x), float(env.robot.y)
    rori = float(env.robot.orientation)

    try:
        env.robot.x = 0.0
        env.robot.y = 0.0
        env.robot.orientation = rori
        env.robot.draw(img, px_per_m_x, px_per_m_y, map_size_x, map_size_y)
    finally:
        env.robot.x, env.robot.y, env.robot.orientation = rx, ry, rori


def render_state_discretizer_grid(
    obs: dict,
    env,
    discretizer,
    show_grid_lines: bool = True,
    draw_robot_glyph: bool = True,
    label: str = "StateDiscretizer grid",
):
    """
    Render EXACTLY what your StateDiscretizer encodes:
      (gx_bin, gy_bin, theta_bin, hx_bin, hy_bin)

    Visual:
      - Goal bin as cyan
      - Closest human bin as blue (per your request)
      - Robot glyph centered (same as world render)
    """
    # Bins/ranges from discretizer
    gx_bins = int(getattr(discretizer, "xy_bins", 30))
    gx_max = float(getattr(discretizer, "xy_max_abs", 10.0))
    hx_bins = int(getattr(discretizer, "human_xy_bins", 30))
    hx_max = float(getattr(discretizer, "human_xy_max_abs", 10.0))

    # We want a single grid; use the human grid resolution (usually same as xy_bins anyway).
    # If they differ, we still render a grid with max(bins) and map both onto it.
    bins = int(max(gx_bins, hx_bins))
    extent = float(max(gx_max, hx_max))

    # State from discretizer
    gx_bin, gy_bin, theta_bin, hx_bin, hy_bin = discretizer.encode(obs)

    # Grid image in cell-resolution then upscale to env resolution
    cell_img = np.zeros((bins, bins, 3), dtype=np.uint8)

    # Map bins from their own ranges into the unified grid resolution if needed
    def _remap_bin(b: int, src_bins: int, dst_bins: int):
        if b is None or b < 0:
            return -1
        if src_bins <= 1:
            return -1
        # scale bin index proportionally
        return int(np.clip(round(b * (dst_bins - 1) / (src_bins - 1)), 0, dst_bins - 1))

    gx_ix = _remap_bin(gx_bin, gx_bins, bins)
    gy_iy = _remap_bin(gy_bin, gx_bins, bins)
    hx_ix = _remap_bin(hx_bin, hx_bins, bins)
    hy_iy = _remap_bin(hy_bin, hx_bins, bins)

    # Draw goal (cyan = BGR(255,255,0))
    if gx_ix >= 0 and gy_iy >= 0:
        cell_img[gy_iy, gx_ix] = (255, 255, 0)

    # Draw closest human (blue = BGR(255,0,0))
    if hx_ix >= 0 and hy_iy >= 0:
        # if overlap, make it white for visibility
        if gx_ix == hx_ix and gy_iy == hy_iy:
            cell_img[hy_iy, hx_ix] = (255, 255, 255)
        else:
            cell_img[hy_iy, hx_ix] = (255, 0, 0)

    # Optional grid lines (thin)
    if show_grid_lines:
        # draw lines on cell_img then upscale
        for i in range(bins):
            cell_img[i, :] = np.maximum(cell_img[i, :], (10, 10, 10))
            cell_img[:, i] = np.maximum(cell_img[:, i], (10, 10, 10))

    # Upscale to env resolution
    img = cv2.resize(cell_img, (env.RESOLUTION_X, env.RESOLUTION_Y), interpolation=cv2.INTER_NEAREST)

    # Label + state text
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"state=(g:{gx_bin},{gy_bin} th:{theta_bin} h:{hx_bin},{hy_bin})",
        (10, env.RESOLUTION_Y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Overlay same robot glyph, centered and oriented
    if draw_robot_glyph:
        _draw_robot_glyph_centered(img, env, extent_m=extent)

    return img
