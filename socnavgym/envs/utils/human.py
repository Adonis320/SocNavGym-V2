import cv2
import numpy as np
from socnavgym.envs.utils.object import Object
from socnavgym.envs.utils.utils import w2px, w2py
from math import atan2
import time

MAX_TIME_TO_REACH_GOAL = 15

class Human(Object):
    """
    Class for humans
    """

    def __init__(
        self, 
        id=None, 
        x=None, 
        y=None, 
        theta=None, 
        width=None, 
        speed=None, 
        goal_x=None, 
        goal_y=None, 
        goal_radius=None, 
        policy=None,
        prob_to_avoid_robot=0.05,
        type="dynamic",
        fov=2*np.pi,
        pos_noise_std=None,
        angle_noise_std=None,
        draw_waypoints=False,
        waypoints=None
    ) -> None:
        super().__init__(id, "human")
        self.width = None  # diameter of the human
        self.is_static = False  # humans can move, so is_static is False
        self.speed = 0  # linear speed
        self.collided_object = None  # name of the object with which collision has happened
        self.goal_x = None  # x coordinate of the goal
        self.goal_y = None  # y coordinate of the goal
        self.goal_radius = None # goal radius
        self.policy = None  # policy is sfm or orca
        self.prob_to_avoid_robot = prob_to_avoid_robot
        self.fov = fov  # field of view
        self.type = type  # whether human is static or dynamic
        self.pos_noise_std = pos_noise_std if pos_noise_std!=None else 0
        self.angle_noise_std = angle_noise_std if angle_noise_std!=None else 0
        self.draw_waypoints = bool(draw_waypoints)
        
        assert(self.type == "static" or self.type == "dynamic"), "type can be \"static\" or \"dynamic\" only."
        self.set(id, x, y, theta, width, speed, goal_x, goal_y, goal_radius, policy)

        self.initial_time = time.time()
        # PRM waypoint following
        self.waypoints = waypoints  # list of (x, y) tuples or None
        self.cur_wp_idx = 0
        # waypoint switching threshold (defaults to goal_radius if available, else 0.5m)
        self.wp_thresh = self.goal_radius if getattr(self, "goal_radius", None) is not None else 0.5

    def set_goal(self, goal_x, goal_y):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.initial_time = time.time()

    def set(self, id, x, y, theta, width, speed, goal_x, goal_y, goal_radius, policy):
        super().set(id, x, y, theta)
        self.width = width
        if self.width is not None:
            self.length = width * 0.2  # thickness of the shoulder (for visualization)
            self.radius = width / 5  # radius of head (for visualization)
        if speed is not None:
            self.speed = speed  # speed
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_radius = goal_radius
        self.policy = policy
        self.initial_x = x
        self.initial_y = y
        self.initial_orientation = theta


    def has_reached_goal(self, offset=None):
        if offset is None: offset = self.width/2
        if self.type == "static": return False  # static humans do not have goals, so they would not reach their goal
        # if self.width == None or self.goal_radius == None or self.goal_x==None or self.goal_y == None: return False
        distance_to_goal = np.sqrt((self.x-self.goal_x)**2 + (self.y-self.goal_y)**2)
        if distance_to_goal < (offset + self.goal_radius) or time.time()-self.initial_time>MAX_TIME_TO_REACH_GOAL:
            return True
        else:
            return False
    @property
    def avoids_robot(self):
        n = np.random.random()
        if n <= self.prob_to_avoid_robot:
            return True
        else:
            return False
    
    def set_new_orientation_with_limits(self, orientation, max_rotation_speed, time):
        diffO = atan2(np.sin(orientation-self.orientation), np.cos(orientation-self.orientation))
        if abs(diffO)/time>max_rotation_speed:
            if diffO>0:
                diffO = max_rotation_speed*time
            else:
                diffO = -max_rotation_speed*time
            self.orientation = atan2(np.sin(self.orientation+diffO), np.cos(self.orientation+diffO))
            return False

        self.orientation = orientation
        return True


    def update_orientation(self, theta):
        if self.type == "static": return  # static humans do not change their orientation
        self.orientation = theta

    def current_target(self):
        """
        Returns the current waypoint target. Advances to the next waypoint if within threshold.
        If no waypoints are set, falls back to the final goal.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return (self.goal_x, self.goal_y)
        # clamp index
        if self.cur_wp_idx < 0:
            self.cur_wp_idx = 0
        if self.cur_wp_idx >= len(self.waypoints):
            self.cur_wp_idx = len(self.waypoints) - 1
        tx, ty = self.waypoints[self.cur_wp_idx]
        dx = tx - self.x
        dy = ty - self.y
        thresh = self.wp_thresh if self.wp_thresh is not None else 0.5
        if dx*dx + dy*dy < thresh**2:
            if self.cur_wp_idx < len(self.waypoints) - 1:
                self.cur_wp_idx += 1
                tx, ty = self.waypoints[self.cur_wp_idx]
        return (tx, ty)

    def set_waypoints(self, waypoints, wp_thresh=None):
        """
        Assign a new list of waypoints for this human to follow.
        waypoints: iterable of (x, y) tuples starting at (approximately) the human's current position and ending at the final goal.
        wp_thresh: optional float distance to switch to the next waypoint.
        """
        self.waypoints = list(waypoints) if waypoints is not None else None
        self.cur_wp_idx = 0
        if wp_thresh is not None:
            self.wp_thresh = wp_thresh

    def update(self, time):
        """
        For updating the coordinates of the human for a single time step
        """
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"
        # if self.type == "static": return  # static humans do not change their position
        if self.type == "static":
            self.x = self.initial_x
            self.y = self.initial_y
            self.orientation = self.initial_orientation
        r_moved = np.random.normal(0, self.pos_noise_std)
        moved = time * self.speed  + r_moved# distance moved = speed x time
        r_angle = np.random.normal(0, self.angle_noise_std)
        self.initial_x = self.x
        self.initial_y = self.y
        self.initial_orientation = self.orientation

        self.orientation = self.orientation + r_angle
        self.x += moved * np.cos(self.orientation)  # updating x position
        self.y += moved * np.sin(self.orientation)  # updating y position

    def draw(self, img, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        if self.color == None:
            color = (240, 114, 66)  # blue
        else:
            color = self.color
        assert self.width != None, "Width is None type."
        assert (
            self.x != None and self.y != None and self.orientation != None
        ), "Coordinates or orientation are None type"

        # p1, p2, p3, p4 are the coordinates of the corners of the rectangle. calculation is done so as to orient the rectangle at an angle.

        p1 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p2 = [
            w2px(
                (
                    self.x
                    + self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    + self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p3 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    + self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    - self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]

        p4 = [
            w2px(
                (
                    self.x
                    - self.length / 2 * np.cos(self.orientation)
                    - self.width / 2 * np.sin(self.orientation)
                ),
                PIXEL_TO_WORLD_X,
                MAP_SIZE_X,
            ),
            w2py(
                (
                    self.y
                    - self.length / 2 * np.sin(self.orientation)
                    + self.width / 2 * np.cos(self.orientation)
                ),
                PIXEL_TO_WORLD_Y,
                MAP_SIZE_Y,
            ),
        ]
        points = np.array([p1, p2, p3, p4])
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(
            img, [np.int32(points)], color
        )  # filling the rectangle made from the points with the specified color
        cv2.polylines(
            img, [np.int32(points)], True, (0, 0, 0), 2
        )  # bordering the rectangle

        black = (0, 0, 0)  # color for the head
        assert self.radius != None, "Radius is None type."
        assert self.x != None and self.y != None, "Coordinates are None type"

        radius = w2px(self.x + self.radius, PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius

        cv2.circle(
            img,
            (
                w2px(
                    self.x + (self.width / 10) * np.cos(self.orientation),
                    PIXEL_TO_WORLD_X,
                    MAP_SIZE_X,
                ),
                w2py(
                    self.y + (self.width / 10) * np.sin(self.orientation),
                    PIXEL_TO_WORLD_Y,
                    MAP_SIZE_Y,
                ),
            ),
            radius,
            black,
            -1,
        )  # drawing a circle for the head of the human

        # draw waypoints and path if available
        if self.draw_waypoints and self.waypoints is not None and len(self.waypoints) > 0:
            try:
                # colors and sizes
                path_color = (255, 0, 255)  # magenta
                node_color = (200, 0, 200)  # slightly different magenta
                target_color = (0, 0, 255)  # red
                node_radius_px = 4
                target_radius_px = 6
                thickness = 1

                # clamp current waypoint index without modifying it
                idx = max(0, min(self.cur_wp_idx, len(self.waypoints) - 1))

                # convert waypoints to pixel coordinates
                wp_pts = []
                for (wx, wy) in self.waypoints:
                    px = w2px(wx, PIXEL_TO_WORLD_X, MAP_SIZE_X)
                    py = w2py(wy, PIXEL_TO_WORLD_Y, MAP_SIZE_Y)
                    wp_pts.append([int(px), int(py)])

                # draw path polyline
                if len(wp_pts) >= 2:
                    poly = np.array(wp_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [poly], False, path_color, thickness)

                # draw waypoint nodes
                for j, (px, py) in enumerate(wp_pts):
                    cv2.circle(img, (px, py), node_radius_px, node_color, -1)

                # highlight current target waypoint
                if 0 <= idx < len(wp_pts):
                    cx, cy = wp_pts[idx]
                    cv2.circle(img, (cx, cy), target_radius_px, target_color, 2)
            except Exception:
                # avoid breaking rendering if anything goes wrong
                pass

    def draw_gaze_range(self, img, gaze_angle, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_SIZE_X, MAP_SIZE_Y):
        center = (w2px(self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X), w2py(self.y, PIXEL_TO_WORLD_Y, MAP_SIZE_Y))
        radius = w2px(self.x + np.sqrt(MAP_SIZE_X**2 + MAP_SIZE_Y**2), PIXEL_TO_WORLD_X, MAP_SIZE_X) - w2px(
            self.x, PIXEL_TO_WORLD_X, MAP_SIZE_X
        )  # calculating no. of pixels corresponding to the radius
       
        axesLength = (radius, radius)
        gaze_angle = gaze_angle * 180 / np.pi
        orientation = self.orientation * 180 / np.pi

        cv2.ellipse(
            img,
            center,
            axesLength,
            angle=-orientation,
            startAngle=(-gaze_angle/2),
            endAngle=(gaze_angle/2),
            color=(218, 252, 81), 
            thickness=-1
        )
