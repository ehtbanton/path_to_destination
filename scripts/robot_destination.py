"""
Robot Destination script - A robot (blue circle) moves towards a destination (green circle).

Features:
- 32m x 18m simulated space with 16:9 aspect ratio
- Blue circle (robot) that faces movement direction with 70-degree FOV vision
- Green circle (destination) as the target
- Red rectangle obstacle (draggable, resizable, rotatable)
- Path avoidance when FOV detects obstacle
- Start/Stop button and speed slider
"""

import pygame
import math
from engine import Game, Sprite, Colors


# World coordinate system: 32m x 18m space
WORLD_WIDTH_METERS = 32.0
WORLD_HEIGHT_METERS = 18.0
ASPECT_RATIO = 16 / 9

# Screen dimensions (16:9 aspect ratio)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# Pixels per meter
PPM = SCREEN_WIDTH / WORLD_WIDTH_METERS  # 40 pixels per meter


def meters_to_pixels(meters):
    """Convert meters to pixels."""
    return meters * PPM


def pixels_to_meters(pixels):
    """Convert pixels to meters."""
    return pixels / PPM


class Obstacle(Sprite):
    """
    A red rectangle obstacle that can be dragged, resized, and rotated.

    Controls:
    - Left-click drag center: Move the obstacle
    - Left-click drag corners: Resize the obstacle
    - Mouse wheel: Rotate the obstacle
    """

    HANDLE_RADIUS = 8
    MIN_SIZE = meters_to_pixels(0.5)  # Minimum 0.5m

    def __init__(self, x, y, width=None, height=None):
        # Default size: 3m x 1.5m
        if width is None:
            width = meters_to_pixels(3.0)
        if height is None:
            height = meters_to_pixels(1.5)
        super().__init__(x, y, width, height, Colors.RED)
        self.tag = "obstacle"
        self.angle = 0  # Rotation angle in degrees
        self.color = Colors.RED

        # Interaction states
        self.dragging = False
        self.resizing = False
        self.resize_corner = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self._rebuild_rotated_image()

    def _rebuild_rotated_image(self):
        """Rebuild the sprite image with current rotation."""
        base_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(base_surface, self.color, (0, 0, self.width, self.height))
        self._image = pygame.transform.rotate(base_surface, -self.angle)
        self._custom_image = True

    @property
    def center(self):
        """Get the center of the obstacle."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @center.setter
    def center(self, pos):
        """Set position by center."""
        self.x = pos[0] - self.width / 2
        self.y = pos[1] - self.height / 2

    def get_corners(self):
        """Get the four corners of the rectangle in world space (accounting for rotation)."""
        cx, cy = self.center
        hw, hh = self.width / 2, self.height / 2

        local_corners = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh),
        ]

        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        world_corners = []
        for lx, ly in local_corners:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            world_corners.append((cx + rx, cy + ry))

        return world_corners

    def contains_point(self, pos):
        """Check if a point is inside the rotated rectangle."""
        cx, cy = self.center
        px, py = pos[0] - cx, pos[1] - cy

        angle_rad = math.radians(-self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        local_x = px * cos_a - py * sin_a
        local_y = px * sin_a + py * cos_a

        hw, hh = self.width / 2, self.height / 2
        return -hw <= local_x <= hw and -hh <= local_y <= hh

    def get_corner_at_point(self, pos):
        """Check if point is near a corner handle."""
        corners = self.get_corners()
        for i, corner in enumerate(corners):
            dx = pos[0] - corner[0]
            dy = pos[1] - corner[1]
            if dx * dx + dy * dy <= self.HANDLE_RADIUS * self.HANDLE_RADIUS * 4:
                return i
        return None

    def start_drag(self, mouse_pos):
        """Start dragging the obstacle."""
        corner = self.get_corner_at_point(mouse_pos)
        if corner is not None:
            self.resizing = True
            self.resize_corner = corner
        else:
            self.dragging = True
            self.drag_offset_x = self.x - mouse_pos[0]
            self.drag_offset_y = self.y - mouse_pos[1]

    def stop_drag(self):
        """Stop dragging/resizing."""
        self.dragging = False
        self.resizing = False
        self.resize_corner = None

    def update_drag(self, mouse_pos):
        """Update position or size while being dragged."""
        if self.dragging:
            self.x = mouse_pos[0] + self.drag_offset_x
            self.y = mouse_pos[1] + self.drag_offset_y
        elif self.resizing and self.resize_corner is not None:
            self._handle_resize(mouse_pos)

    def _handle_resize(self, mouse_pos):
        """Handle resizing from a corner."""
        cx, cy = self.center
        px, py = mouse_pos[0] - cx, mouse_pos[1] - cy
        angle_rad = math.radians(-self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        local_x = px * cos_a - py * sin_a
        local_y = px * sin_a + py * cos_a

        new_width = max(self.MIN_SIZE, abs(local_x) * 2)
        new_height = max(self.MIN_SIZE, abs(local_y) * 2)

        old_center = self.center
        self.width = new_width
        self.height = new_height
        self.center = old_center
        self._rebuild_rotated_image()

    def rotate(self, delta_angle):
        """Rotate the obstacle by delta degrees."""
        self.angle = (self.angle + delta_angle) % 360
        self._rebuild_rotated_image()

    def draw(self, screen):
        """Draw the obstacle with rotation and corner handles."""
        corners = self.get_corners()
        pygame.draw.polygon(screen, self.color, corners)
        pygame.draw.polygon(screen, Colors.WHITE, corners, 2)

        for corner in corners:
            pygame.draw.circle(screen, Colors.WHITE, (int(corner[0]), int(corner[1])), self.HANDLE_RADIUS)
            pygame.draw.circle(screen, Colors.RED, (int(corner[0]), int(corner[1])), self.HANDLE_RADIUS - 2)

    def _draw(self, screen):
        """Override default draw to use custom rotation drawing."""
        if self.visible:
            self.draw(screen)


class Path:
    """
    Computes and manages a path from start to end point with obstacle avoidance.
    """

    def __init__(self):
        self.waypoints = []
        self.dot_spacing = meters_to_pixels(0.5)  # 0.5m between dots
        self.dot_radius = 4
        self.dot_color = Colors.YELLOW
        self.obstacle = None
        self.avoidance_point = None  # The intermediate point to go around obstacle

    def compute(self, start_pos, end_pos, obstacle=None, fov_sees_obstacle=False):
        """
        Compute the path from start to end position, avoiding obstacle if detected.
        """
        self.obstacle = obstacle

        if fov_sees_obstacle and obstacle is not None:
            # Compute 2-segment avoidance path
            self.waypoints = self._compute_avoidance_path(start_pos, end_pos, obstacle)
        else:
            # Direct path
            self.waypoints = self._compute_straight_line(start_pos, end_pos)
            self.avoidance_point = None

    def _compute_avoidance_path(self, start_pos, end_pos, obstacle):
        """Compute a 2-segment path that goes around the obstacle."""
        # Get obstacle corners and center
        obstacle_center = obstacle.center
        corners = obstacle.get_corners()

        # Calculate direction from start to end
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]

        # Calculate perpendicular direction (for going around)
        perp_x = -dy
        perp_y = dx
        perp_len = math.sqrt(perp_x * perp_x + perp_y * perp_y)
        if perp_len > 0:
            perp_x /= perp_len
            perp_y /= perp_len

        # Find which side to go around (left or right of the obstacle)
        # Check which side has more clearance
        margin = meters_to_pixels(1.0)  # 1m margin around obstacle

        # Calculate bounding box extents in the perpendicular direction
        max_extent = 0
        for corner in corners:
            cx, cy = corner[0] - obstacle_center[0], corner[1] - obstacle_center[1]
            extent = abs(cx * perp_x + cy * perp_y)
            max_extent = max(max_extent, extent)

        # Determine which side to go (based on cross product to see which is shorter)
        # Use cross product of (start->obstacle) and (start->end) to determine side
        to_obstacle_x = obstacle_center[0] - start_pos[0]
        to_obstacle_y = obstacle_center[1] - start_pos[1]
        cross = dx * to_obstacle_y - dy * to_obstacle_x

        # Choose side based on cross product sign
        side = 1 if cross > 0 else -1

        # Calculate avoidance point: beside the obstacle
        avoidance_x = obstacle_center[0] + side * perp_x * (max_extent + margin)
        avoidance_y = obstacle_center[1] + side * perp_y * (max_extent + margin)
        self.avoidance_point = (avoidance_x, avoidance_y)

        # Generate waypoints for both segments
        segment1 = self._compute_straight_line(start_pos, self.avoidance_point)
        segment2 = self._compute_straight_line(self.avoidance_point, end_pos)

        # Combine segments (remove duplicate point at junction)
        return segment1 + segment2[1:]

    def _compute_straight_line(self, start_pos, end_pos):
        """Compute waypoints along a straight line."""
        waypoints = []

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:
            return [end_pos]

        num_points = max(2, int(distance / self.dot_spacing))
        for i in range(num_points + 1):
            t = i / num_points
            x = start_pos[0] + dx * t
            y = start_pos[1] + dy * t
            waypoints.append((x, y))

        return waypoints

    def get_next_waypoint(self, current_pos, threshold=None):
        """Get the next waypoint to move towards."""
        if threshold is None:
            threshold = meters_to_pixels(0.25)  # 0.25m threshold

        if not self.waypoints:
            return None

        for waypoint in self.waypoints:
            dx = waypoint[0] - current_pos[0]
            dy = waypoint[1] - current_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > threshold:
                return waypoint

        return self.waypoints[-1] if self.waypoints else None

    def draw(self, screen):
        """Draw the path as a dotted line."""
        for waypoint in self.waypoints:
            pygame.draw.circle(
                screen,
                self.dot_color,
                (int(waypoint[0]), int(waypoint[1])),
                self.dot_radius
            )

    def is_complete(self, current_pos, threshold=None):
        """Check if the path has been completed."""
        if threshold is None:
            threshold = meters_to_pixels(0.25)

        if not self.waypoints:
            return True

        final = self.waypoints[-1]
        dx = final[0] - current_pos[0]
        dy = final[1] - current_pos[1]
        return math.sqrt(dx * dx + dy * dy) <= threshold


class CircleSprite(Sprite):
    """A circular sprite that can be dragged."""

    def __init__(self, x, y, radius, color):
        super().__init__(x, y, width=radius * 2, height=radius * 2)
        self.radius = radius
        self.color = color
        self.dragging = False
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self._create_circle_image()

    def _create_circle_image(self):
        """Create a circular image with transparency."""
        self._image = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(
            self._image,
            self.color,
            (self.radius, self.radius),
            self.radius
        )
        self._custom_image = True

    def contains_point(self, pos):
        """Check if a point is within the circle."""
        cx, cy = self.center
        dx = pos[0] - cx
        dy = pos[1] - cy
        return (dx * dx + dy * dy) <= (self.radius * self.radius)

    def start_drag(self, mouse_pos):
        """Start dragging the sprite."""
        self.dragging = True
        self.drag_offset_x = self.x - mouse_pos[0]
        self.drag_offset_y = self.y - mouse_pos[1]

    def stop_drag(self):
        """Stop dragging the sprite."""
        self.dragging = False

    def update_drag(self, mouse_pos):
        """Update position while being dragged."""
        if self.dragging:
            self.x = mouse_pos[0] + self.drag_offset_x
            self.y = mouse_pos[1] + self.drag_offset_y


class Robot(CircleSprite):
    """The blue robot circle with FOV vision that follows a computed path."""

    FOV_ANGLE = 70  # Degrees
    FOV_RANGE_METERS = 4.0  # 4 meters ahead

    def __init__(self, x, y, radius=None):
        if radius is None:
            radius = meters_to_pixels(0.5)  # 0.5m radius robot
        super().__init__(x, y, radius, Colors.BLUE)
        self.tag = "robot"
        self.moving = False
        self.speed = meters_to_pixels(2.0)  # 2 m/s default
        self.destination = None
        self.obstacle = None
        self.path = Path()
        self.facing_angle = 0  # Degrees, 0 = right, 90 = down
        self.fov_sees_obstacle = False
        self._last_robot_pos = None
        self._last_dest_pos = None
        self._last_obstacle_state = None

    def get_fov_triangle(self):
        """Get the three points of the FOV vision triangle."""
        cx, cy = self.center
        fov_range = meters_to_pixels(self.FOV_RANGE_METERS)
        half_angle = self.FOV_ANGLE / 2

        # Robot center is the apex
        apex = (cx, cy)

        # Calculate the two far corners of the FOV
        angle_rad = math.radians(self.facing_angle)
        left_angle = angle_rad - math.radians(half_angle)
        right_angle = angle_rad + math.radians(half_angle)

        left_point = (
            cx + fov_range * math.cos(left_angle),
            cy + fov_range * math.sin(left_angle)
        )
        right_point = (
            cx + fov_range * math.cos(right_angle),
            cy + fov_range * math.sin(right_angle)
        )

        return [apex, left_point, right_point]

    def check_fov_obstacle_collision(self, obstacle):
        """Check if the FOV triangle intersects with the obstacle."""
        if obstacle is None:
            return False

        fov_triangle = self.get_fov_triangle()
        obstacle_corners = obstacle.get_corners()

        # Check if any obstacle corner is inside FOV triangle
        for corner in obstacle_corners:
            if self._point_in_triangle(corner, fov_triangle):
                return True

        # Check if any FOV triangle point is inside obstacle
        for point in fov_triangle:
            if obstacle.contains_point(point):
                return True

        # Check if edges intersect
        fov_edges = [
            (fov_triangle[0], fov_triangle[1]),
            (fov_triangle[1], fov_triangle[2]),
            (fov_triangle[2], fov_triangle[0])
        ]
        obstacle_edges = [
            (obstacle_corners[0], obstacle_corners[1]),
            (obstacle_corners[1], obstacle_corners[2]),
            (obstacle_corners[2], obstacle_corners[3]),
            (obstacle_corners[3], obstacle_corners[0])
        ]

        for fov_edge in fov_edges:
            for obs_edge in obstacle_edges:
                if self._segments_intersect(fov_edge[0], fov_edge[1], obs_edge[0], obs_edge[1]):
                    return True

        return False

    def _point_in_triangle(self, point, triangle):
        """Check if a point is inside a triangle using barycentric coordinates."""
        p0, p1, p2 = triangle
        px, py = point

        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(point, p0, p1)
        d2 = sign(point, p1, p2)
        d3 = sign(point, p2, p0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _segments_intersect(self, p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects with p3-p4."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def update_path(self):
        """Recompute the path if needed."""
        if self.destination is None:
            return

        current_robot_pos = self.center
        current_dest_pos = self.destination.center

        # Check obstacle state
        obstacle_state = None
        if self.obstacle:
            obstacle_state = (
                self.obstacle.center,
                self.obstacle.width,
                self.obstacle.height,
                self.obstacle.angle
            )

        # Check if anything has changed
        needs_update = False

        if self._last_robot_pos is None or self._last_dest_pos is None:
            needs_update = True
        else:
            robot_moved = (
                abs(current_robot_pos[0] - self._last_robot_pos[0]) > 5 or
                abs(current_robot_pos[1] - self._last_robot_pos[1]) > 5
            )
            dest_moved = (
                abs(current_dest_pos[0] - self._last_dest_pos[0]) > 5 or
                abs(current_dest_pos[1] - self._last_dest_pos[1]) > 5
            )
            obstacle_changed = obstacle_state != self._last_obstacle_state
            needs_update = robot_moved or dest_moved or obstacle_changed

        if needs_update:
            # Check FOV collision
            self.fov_sees_obstacle = self.check_fov_obstacle_collision(self.obstacle)

            # Compute path with avoidance if needed
            self.path.compute(
                current_robot_pos,
                current_dest_pos,
                self.obstacle,
                self.fov_sees_obstacle
            )

            self._last_robot_pos = current_robot_pos
            self._last_dest_pos = current_dest_pos
            self._last_obstacle_state = obstacle_state

    def on_update(self, dt):
        # Always update path when positions change
        self.update_path()

        if self.moving and self.destination and not self.dragging:
            next_waypoint = self.path.get_next_waypoint(self.center)

            if next_waypoint and not self.path.is_complete(self.center):
                dx = next_waypoint[0] - self.center[0]
                dy = next_waypoint[1] - self.center[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > meters_to_pixels(0.1):
                    # Update facing angle based on movement direction
                    self.facing_angle = math.degrees(math.atan2(dy, dx))

                    # Move towards waypoint
                    self.x += (dx / distance) * self.speed * dt
                    self.y += (dy / distance) * self.speed * dt
        elif not self.moving and self.destination:
            # When stopped, face towards destination
            dx = self.destination.center[0] - self.center[0]
            dy = self.destination.center[1] - self.center[1]
            if dx != 0 or dy != 0:
                self.facing_angle = math.degrees(math.atan2(dy, dx))

    def draw_fov(self, screen):
        """Draw the FOV vision triangle."""
        fov_triangle = self.get_fov_triangle()

        # Create translucent surface for FOV
        fov_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        # Choose color based on whether obstacle is detected
        if self.fov_sees_obstacle:
            fov_color = (255, 100, 100, 80)  # Red tint when seeing obstacle
        else:
            fov_color = (100, 200, 255, 80)  # Blue tint normally

        # Draw filled triangle
        int_triangle = [(int(p[0]), int(p[1])) for p in fov_triangle]
        pygame.draw.polygon(fov_surface, fov_color, int_triangle)

        # Draw outline
        outline_color = (255, 255, 255, 150)
        pygame.draw.polygon(fov_surface, outline_color, int_triangle, 2)

        screen.blit(fov_surface, (0, 0))

    def draw_robot(self, screen):
        """Draw the robot with facing direction indicator."""
        cx, cy = self.center
        int_cx, int_cy = int(cx), int(cy)
        int_radius = int(self.radius)

        # Draw robot body
        pygame.draw.circle(screen, self.color, (int_cx, int_cy), int_radius)
        pygame.draw.circle(screen, Colors.WHITE, (int_cx, int_cy), int_radius, 2)

        # Draw direction indicator (small triangle pointing forward)
        angle_rad = math.radians(self.facing_angle)
        indicator_length = self.radius * 0.8
        tip_x = cx + indicator_length * math.cos(angle_rad)
        tip_y = cy + indicator_length * math.sin(angle_rad)

        # Draw a small arrow/triangle at the front
        back_angle1 = angle_rad + math.radians(140)
        back_angle2 = angle_rad - math.radians(140)
        back_len = self.radius * 0.4
        back1 = (cx + back_len * math.cos(back_angle1), cy + back_len * math.sin(back_angle1))
        back2 = (cx + back_len * math.cos(back_angle2), cy + back_len * math.sin(back_angle2))

        indicator_points = [(int(tip_x), int(tip_y)), (int(back1[0]), int(back1[1])), (int(back2[0]), int(back2[1]))]
        pygame.draw.polygon(screen, Colors.WHITE, indicator_points)

    def draw_path(self, screen):
        """Draw the path visualization."""
        self.path.draw(screen)

    def _draw(self, screen):
        """Override to use custom drawing."""
        if self.visible:
            self.draw_robot(screen)


class Destination(CircleSprite):
    """The green destination circle."""

    def __init__(self, x, y, radius=None):
        if radius is None:
            radius = meters_to_pixels(0.4)  # 0.4m radius
        super().__init__(x, y, radius, Colors.GREEN)
        self.tag = "destination"


class Button:
    """A simple clickable button."""

    def __init__(self, x, y, width, height, text, color, hover_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.font = None
        self.hovered = False

    def draw(self, screen):
        if self.font is None:
            self.font = pygame.font.Font(None, 28)

        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, Colors.WHITE, self.rect, 2, border_radius=5)

        text_surface = self.font.render(self.text, True, Colors.WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class Slider:
    """A horizontal slider for adjusting values."""

    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label, unit=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.unit = unit
        self.dragging = False
        self.font = None
        self.knob_radius = height // 2 + 2

    @property
    def knob_x(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + ratio * self.rect.width

    def draw(self, screen):
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

        label_text = self.font.render(f"{self.label}: {self.value:.1f}{self.unit}", True, Colors.WHITE)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

        track_rect = pygame.Rect(
            self.rect.x, self.rect.y + self.rect.height // 2 - 3,
            self.rect.width, 6
        )
        pygame.draw.rect(screen, Colors.GRAY, track_rect, border_radius=3)

        filled_width = self.knob_x - self.rect.x
        if filled_width > 0:
            filled_rect = pygame.Rect(
                self.rect.x, self.rect.y + self.rect.height // 2 - 3,
                filled_width, 6
            )
            pygame.draw.rect(screen, Colors.CYAN, filled_rect, border_radius=3)

        pygame.draw.circle(
            screen, Colors.WHITE,
            (int(self.knob_x), self.rect.y + self.rect.height // 2),
            self.knob_radius
        )

    def handle_mouse_down(self, mouse_pos):
        knob_center = (self.knob_x, self.rect.y + self.rect.height // 2)
        dx = mouse_pos[0] - knob_center[0]
        dy = mouse_pos[1] - knob_center[1]
        if dx * dx + dy * dy <= self.knob_radius * self.knob_radius:
            self.dragging = True
            return True
        elif self.rect.collidepoint(mouse_pos):
            self._update_value(mouse_pos[0])
            self.dragging = True
            return True
        return False

    def handle_mouse_up(self):
        self.dragging = False

    def handle_mouse_move(self, mouse_pos):
        if self.dragging:
            self._update_value(mouse_pos[0])

    def _update_value(self, mouse_x):
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)


class RobotDestinationGame(Game):
    """Custom game class with UI handling for robot destination."""

    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Robot to Destination (32m x 18m)", 60)
        self.robot = None
        self.destination = None
        self.obstacle = None
        self.button = None
        self.slider = None
        self.is_moving = False
        self.dragged_sprite = None

    def on_setup(self):
        self.background_color = Colors.DARK_GRAY

        # Create robot at 4m, 9m (left side, center height)
        robot_x = meters_to_pixels(4.0) - meters_to_pixels(0.5)
        robot_y = meters_to_pixels(9.0) - meters_to_pixels(0.5)
        self.robot = Robot(robot_x, robot_y)
        self.add(self.robot)

        # Create destination at 28m, 9m (right side, center height)
        dest_x = meters_to_pixels(28.0) - meters_to_pixels(0.4)
        dest_y = meters_to_pixels(9.0) - meters_to_pixels(0.4)
        self.destination = Destination(dest_x, dest_y)
        self.add(self.destination)

        # Create obstacle at center (16m, 9m)
        obs_x = meters_to_pixels(16.0) - meters_to_pixels(1.5)
        obs_y = meters_to_pixels(9.0) - meters_to_pixels(0.75)
        self.obstacle = Obstacle(obs_x, obs_y)
        self.add(self.obstacle)

        # Link robot to destination and obstacle
        self.robot.destination = self.destination
        self.robot.obstacle = self.obstacle

        # Create UI elements
        self.button = Button(
            self.width // 2 - 50, 20, 100, 40,
            "Start", Colors.GREEN, (0, 200, 0)
        )

        # Speed slider in m/s
        self.slider = Slider(
            self.width // 2 - 100, self.height - 50,
            200, 20, 0.5, 8.0, 2.0, "Speed", " m/s"
        )

        print("Robot Destination Game (32m x 18m space)")
        print("- Drag the blue robot or green destination anywhere")
        print("- Drag the red obstacle to move it, drag corners to resize")
        print("- Mouse wheel over obstacle to rotate it")
        print("- Robot has 70-degree FOV with 4m range")
        print("- Path avoids obstacle when detected in FOV")

    def on_update(self, dt):
        mouse_pos = pygame.mouse.get_pos()

        self.button.update(mouse_pos)
        self.slider.handle_mouse_move(mouse_pos)

        # Update robot speed from slider (convert m/s to pixels/s)
        self.robot.speed = meters_to_pixels(self.slider.value)

        if self.dragged_sprite:
            self.dragged_sprite.update_drag(mouse_pos)

    def on_draw(self, screen):
        # Draw FOV first (behind path)
        self.robot.draw_fov(screen)

        # Draw path
        self.robot.draw_path(screen)

        # Draw button
        self.button.draw(screen)

        # Draw slider
        self.slider.draw(screen)

        # Draw scale indicator
        font = pygame.font.Font(None, 20)
        scale_text = font.render("Scale: 1m = {:.0f}px".format(PPM), True, Colors.LIGHT_GRAY)
        screen.blit(scale_text, (10, self.height - 25))

        # Draw coordinate info
        coord_text = font.render("Space: 32m x 18m", True, Colors.LIGHT_GRAY)
        screen.blit(coord_text, (10, self.height - 45))

    def run(self):
        """Custom run loop with UI event handling."""
        self.running = True
        self.on_setup()

        while self.running:
            self.dt = self._clock.tick(self.fps) / 1000.0

            for sprite in self._sprites_to_add:
                self._sprites.append(sprite)
                sprite.on_create()
            self._sprites_to_add.clear()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._keys_pressed.add(event.key)
                    for sprite in self._sprites:
                        sprite.on_key_down(event.key)
                elif event.type == pygame.KEYUP:
                    self._keys_pressed.discard(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_mouse_wheel(event.y)

            for sprite in self._sprites:
                sprite.on_key_held(self._keys_pressed)
                sprite._internal_update(self.dt)

            for i, sprite in enumerate(self._sprites):
                for other in self._sprites[i + 1:]:
                    if sprite.collides_with(other):
                        sprite.on_collision(other)
                        other.on_collision(sprite)

            self.on_update(self.dt)

            for sprite in self._sprites:
                if sprite._destroyed:
                    sprite.on_destroy()
            self._sprites = [s for s in self._sprites if not s._destroyed]

            self._screen.fill(self.background_color)

            # Draw FOV and path before sprites
            self.robot.draw_fov(self._screen)
            self.robot.draw_path(self._screen)

            for sprite in self._sprites:
                sprite._draw(self._screen)
            self.on_draw(self._screen)
            pygame.display.flip()

        pygame.quit()

    def _handle_mouse_down(self, pos, button):
        """Handle mouse button press."""
        if button != 1:
            return

        if self.slider.handle_mouse_down(pos):
            return

        if self.button.is_clicked(pos):
            self.is_moving = not self.is_moving
            self.robot.moving = self.is_moving
            self.button.text = "Stop" if self.is_moving else "Start"
            self.button.color = Colors.RED if self.is_moving else Colors.GREEN
            self.button.hover_color = (200, 0, 0) if self.is_moving else (0, 200, 0)
            return

        if self.obstacle.get_corner_at_point(pos) is not None:
            self.obstacle.start_drag(pos)
            self.dragged_sprite = self.obstacle
            return

        for sprite in [self.robot, self.destination, self.obstacle]:
            if sprite.contains_point(pos):
                sprite.start_drag(pos)
                self.dragged_sprite = sprite
                break

    def _handle_mouse_up(self, pos, button):
        """Handle mouse button release."""
        if button != 1:
            return

        self.slider.handle_mouse_up()

        if self.dragged_sprite:
            self.dragged_sprite.stop_drag()
            self.dragged_sprite = None

    def _handle_mouse_wheel(self, scroll_y):
        """Handle mouse wheel for obstacle rotation."""
        mouse_pos = pygame.mouse.get_pos()

        if self.obstacle.contains_point(mouse_pos) or self.obstacle.get_corner_at_point(mouse_pos) is not None:
            rotation_speed = 5
            self.obstacle.rotate(scroll_y * rotation_speed)


def create_game():
    """Factory function to create the game instance."""
    return RobotDestinationGame()
