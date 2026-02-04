# socnavgym/envs/utils/continuous_obs_render.py

import numpy as np
import cv2


class ContinuousObsRenderer:
    """
    Render raw (continuous) dict observations into a local robot-frame panel.

    - Left panel (optional): env.render_without_showing(...) world map
    - Right panel: local robot-frame view in meters
        * robot at (0,0), same robot glyph as world render
        * humans/tables/laptops/plants plotted at their robot-frame coords
        * robot goal plotted from obs["robot"] (goal_dx, goal_dy)
        * optional range rings + axes

    Designed for DQN using raw obs (not discretized).
    """

    def __init__(
        self,
        env,
        extent_m: float = 5.0,
        out_size=None,  # (W, H). default: (env.RESOLUTION_X, env.RESOLUTION_Y)
        entity_dim: int | None = None,
        draw_rings: bool = True,
        draw_axes: bool = True,
        draw_robot_glyph: bool = True,
        show_world: bool = True,
        world_kwargs=None,  # dict passed to render_without_showing
    ):
        self.env = env
        self.extent_m = float(extent_m)
        self.entity_dim = int(entity_dim) if entity_dim is not None else int(getattr(env, "entity_obs_dim", 14))
        self.draw_rings = bool(draw_rings)
        self.draw_axes = bool(draw_axes)
        self.draw_robot_glyph = bool(draw_robot_glyph)
        self.show_world = bool(show_world)
        self.world_kwargs = world_kwargs or dict(draw_human_gaze=False, draw_human_goal=True)

        W = int(out_size[0]) if out_size is not None else int(env.RESOLUTION_X)
        H = int(out_size[1]) if out_size is not None else int(env.RESOLUTION_Y)
        self.W, self.H = W, H

    # -------- coordinate helpers --------

    def _meters_to_px(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        """
        Local robot-frame meters -> image pixels.
        x axis: forward (right on image)
        y axis: left/right (up on image, so we flip y).
        """
        e = self.extent_m
        if x_m < -e or x_m > e or y_m < -e or y_m > e:
            return None

        # map x: [-e,e] -> [0,W-1]
        u = int((x_m + e) / (2.0 * e) * (self.W - 1))
        # map y: [-e,e] -> [H-1,0] (flip so +y is up)
        v = int((e - y_m) / (2.0 * e) * (self.H - 1))
        return u, v

    def _draw_robot_glyph_centered(self, img):
        """
        Reuse env.robot.draw to draw the exact same robot glyph used on the world map,
        but centered at (0,0) in a local square map of size 2*extent.
        """
        # Robot.draw expects px-per-meter and map sizes (meters)
        map_size_x = 2.0 * self.extent_m
        map_size_y = 2.0 * self.extent_m
        px_per_m_x = self.W / map_size_x
        px_per_m_y = self.H / map_size_y

        rx, ry = float(self.env.robot.x), float(self.env.robot.y)
        rori = float(self.env.robot.orientation)

        try:
            self.env.robot.x = 0.0
            self.env.robot.y = 0.0
            self.env.robot.orientation = rori
            self.env.robot.draw(img, px_per_m_x, px_per_m_y, map_size_x, map_size_y)
        finally:
            self.env.robot.x, self.env.robot.y, self.env.robot.orientation = rx, ry, rori

    def _entities_from_flat(self, flat: np.ndarray) -> np.ndarray:
        """
        Split flat -> (K, entity_dim) and filter padded rows.
        Assumes first 6 dims are one-hot; works with your obs encoding.
        """
        a = np.asarray(flat, dtype=np.float32).flatten()
        if a.size == 0:
            return np.zeros((0, self.entity_dim), dtype=np.float32)
        k = a.size // self.entity_dim
        ents = a[: k * self.entity_dim].reshape(k, self.entity_dim)

        # filter padded (one-hot all zeros)
        onehot = ents[:, :6]
        keep = np.any(onehot != 0.0, axis=1)
        return ents[keep]

    # -------- drawing primitives --------

    def _draw_point(self, img, x_m, y_m, bgr, radius_px=3, label=None):
        p = self._meters_to_px(float(x_m), float(y_m))
        if p is None:
            return
        cv2.circle(img, p, int(radius_px), tuple(int(c) for c in bgr), -1)
        if label is not None:
            cv2.putText(img, label, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    def _draw_goal_from_robot_obs(self, img, obs):
        """
        Robot obs layout: [one_hot(D), goal_dx, goal_dy, robot_radius]
        so dx is at index D, dy at D+1 with D = len(robot)-3.
        Draw goal as a point only (no arrow).
        """
        robot = obs.get("robot", None)
        if robot is None:
            return None

        r = np.asarray(robot, dtype=np.float32).flatten()
        if r.size < 3:
            return None

        D = max(0, r.size - 3)
        dx = float(r[D])
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        # goal point (cyan)
        self._draw_point(img, dx, dy, bgr=(255, 255, 0), radius_px=5, label="goal")
        return dx, dy

    # -------- public API --------

    def render_local_panel(self, obs: dict) -> np.ndarray:
        """
        Returns a local-panel BGR image (H,W,3) uint8.
        """
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        img[:] = 15  # dark background

        # plot entities using robot-frame coords at indices 6,7 in your entity vector
        def plot_group(key, bgr, label_prefix=None):
            arr = obs.get(key, None)
            if arr is None:
                return 0
            ents = self._entities_from_flat(arr)
            for i, ent in enumerate(ents):
                x_rel = float(ent[6])
                y_rel = float(ent[7])
                lab = None
                if label_prefix is not None and i < 3:  # keep labels minimal
                    lab = f"{label_prefix}{i}"
                self._draw_point(img, x_rel, y_rel, bgr=bgr, radius_px=3, label=lab)
            return len(ents)

        # humans requested blue
        human_pts = self._extract_human_xy_list(obs.get("humans", None))
        for i, (hx, hy) in enumerate(human_pts):
            self._draw_point(img, hx, hy, bgr=(255, 0, 0), radius_px=4)
        plot_group("tables", bgr=(0, 255, 0), label_prefix=None)
        plot_group("laptops", bgr=(0, 0, 255), label_prefix=None)
        plot_group("plants", bgr=(0, 255, 255), label_prefix=None)

        # goal from robot obs
        goal_xy = self._draw_goal_from_robot_obs(img, obs)

        # robot glyph on top
        if self.draw_robot_glyph:
            self._draw_robot_glyph_centered(img)

        # title + quick stats
        cv2.putText(img, "raw obs (continuous, robot frame)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        if goal_xy is not None:
            cv2.putText(img, f"goal_dxdy=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

        return img

    def render(self, obs: dict, show: bool = True, window_name="SocNavGym RawObs", delay_ms: int = 30):
        """
        Returns a panel image. If show_world=True, concatenates world||local.
        """
        local = self.render_local_panel(obs)

        if self.show_world:
            world = self.env.render_without_showing(**self.world_kwargs)
            panel = cv2.hconcat([world, local])
        else:
            panel = local

        if show:
            cv2.imshow(window_name, panel)
            cv2.waitKey(int(delay_ms))

        return panel

    def _extract_human_xy_list(self, humans) -> list[tuple[float, float]]:
        """
        Robustly extract a list of (x,y) in ROBOT FRAME from obs["humans"].
        Supports common formats:
        - shape (N,2) or (N,>=2): uses columns 0,1
        - shape (2,): single (x,y)
        - flat with blocks of 14 (or inferred block): uses indices (6,7) per block
        - flat fallback: if len>=8 uses (6,7), else if len>=2 uses (0,1)
        """
        if humans is None:
            return []

        h = np.asarray(humans, dtype=np.float32)

        # (N,2) or (N,>=2)
        if h.ndim == 2 and h.shape[1] >= 2:
            return [(float(h[i, 0]), float(h[i, 1])) for i in range(h.shape[0])]

        # (2,) -> single
        h = h.flatten()
        if h.size == 2:
            return [(float(h[0]), float(h[1]))]

        if h.size == 0:
            return []

        # Try entity-block style: default (one_hot_len=6, block=14) => x,y at (6,7)
        one_hot_len = 6
        block = 14

        # infer if needed (same style as your discretizer)
        if h.size % block != 0:
            inferred = False
            for k in range(3, 16):
                bs = k + 8
                if bs > 0 and h.size % bs == 0:
                    one_hot_len = k
                    block = bs
                    inferred = True
                    break
            # if not inferred, fall back to best-guess indices
            if not inferred:
                if h.size >= 8:
                    return [(float(h[6]), float(h[7]))]
                if h.size >= 2:
                    return [(float(h[0]), float(h[1]))]
                return []

        # decode blocks
        if h.size % block == 0:
            pts = []
            for i in range(0, h.size, block):
                base = i + one_hot_len
                if base + 1 < h.size:
                    pts.append((float(h[base]), float(h[base + 1])))
            return pts

        return []
