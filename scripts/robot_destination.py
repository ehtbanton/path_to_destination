"""
Robot Destination script - A robot (blue circle) moves towards a destination (green circle).

Features:
- 32m x 18m simulated space with 16:9 aspect ratio
- Blue circle (robot) that faces movement direction with 70-degree FOV vision
- Green circle (destination) as the target
- Red rectangle obstacle (draggable, resizable, rotatable)
- Proximity cloud in FOV with inverse distance values
- Diversion point that moves perpendicular based on obstacle overlap
- Start/Stop button and parameter sliders
"""

import pygame
import math
import random
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


class ProximityPoint:
    """A single point in the proximity cloud with an inverse distance value."""

    def __init__(self, relative_angle, relative_distance, is_left_side):
        # Store relative position (angle offset and distance from robot)
        self.relative_angle = relative_angle  # Angle offset from facing direction
        self.relative_distance = relative_distance  # Distance from robot center
        self.is_left_side = is_left_side

        # World position (updated each frame)
        self.x = 0
        self.y = 0

        # Proximity value (1/distance), negated if on left side
        self.base_proximity = 1.0 / max(relative_distance, 1)
        self.proximity = -self.base_proximity if is_left_side else self.base_proximity

        # Whether this point overlaps with obstacle
        self.overlapping = False

    @property
    def position(self):
        return (self.x, self.y)

    def update_world_position(self, robot_center, facing_angle):
        """Update world position based on robot position and facing."""
        angle_rad = math.radians(facing_angle) + math.radians(self.relative_angle)
        self.x = robot_center[0] + self.relative_distance * math.cos(angle_rad)
        self.y = robot_center[1] + self.relative_distance * math.sin(angle_rad)


class ProximityCloud:
    """
    A static cloud of proximity points within the FOV triangle.
    Each point has a fixed relative position and a value of 1/distance (inverse distance).
    Points on the left of the facing direction are negated.
    """

    def __init__(self, num_points=50):
        self.num_points = num_points
        self.points = []
        self.min_distance = meters_to_pixels(0.3)
        self._initialized = False
        self._last_fov_angle = None
        self._last_fov_range = None

    def _initialize_points(self, fov_angle, fov_range):
        """Initialize the static grid of points within the FOV."""
        self.points = []
        half_angle = fov_angle / 2

        # Create a grid pattern within the FOV triangle
        # Use rows at different distances, with points spread across the angle
        num_rows = max(3, int(math.sqrt(self.num_points)))

        for row in range(num_rows):
            # Distance increases with each row (from min to fov_range)
            t = (row + 1) / num_rows
            distance = self.min_distance + t * (fov_range - self.min_distance)

            # Number of points in this row (more points at greater distances)
            points_in_row = max(3, int(self.num_points / num_rows))

            for i in range(points_in_row):
                # Spread points across the FOV angle
                angle_t = (i + 0.5) / points_in_row  # 0 to 1
                angle_offset = -half_angle + angle_t * (2 * half_angle)

                is_left_side = angle_offset < 0

                point = ProximityPoint(angle_offset, distance, is_left_side)
                self.points.append(point)

        self._initialized = True
        self._last_fov_angle = fov_angle
        self._last_fov_range = fov_range

    def generate(self, robot_center, facing_angle, fov_angle, fov_range):
        """Update point positions based on robot state."""
        # Reinitialize if FOV parameters changed or first time
        if (not self._initialized or
            self._last_fov_angle != fov_angle or
            self._last_fov_range != fov_range):
            self._initialize_points(fov_angle, fov_range)

        # Update world positions of all points
        for point in self.points:
            point.update_world_position(robot_center, facing_angle)

    def get_obstacle_overlap_sum(self, obstacle):
        """
        Calculate the sum of proximity values for points that overlap with the obstacle.
        Returns a tuple: (signed_sum, absolute_sum)
        - signed_sum: positive means obstacle is on right, negative means left
        - absolute_sum: sum of all absolute proximity values (for threshold detection)
        """
        if obstacle is None:
            return 0.0, 0.0

        signed_total = 0.0
        abs_total = 0.0
        for point in self.points:
            point.overlapping = obstacle.contains_point(point.position)
            if point.overlapping:
                signed_total += point.proximity
                abs_total += abs(point.proximity)

        return signed_total, abs_total

    def draw(self, screen, show_values=True):
        """Draw the proximity cloud with static visual indicators."""
        # Create a surface for transparency
        cloud_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        for point in self.points:
            # Base alpha on proximity (closer = more opaque)
            base_alpha = min(200, int(point.base_proximity * 3000))

            # Color based on side (red for left, green for right)
            if point.is_left_side:
                color = (200, 60, 60, base_alpha)  # Red for left side
            else:
                color = (60, 200, 60, base_alpha)  # Green for right side

            # Highlight if overlapping with obstacle
            if point.overlapping:
                color = (255, 255, 0, 220)  # Bright yellow when overlapping

            # Size based on proximity (closer = larger)
            size = max(3, min(10, int(point.base_proximity * 500)))

            pygame.draw.circle(cloud_surface, color, (int(point.x), int(point.y)), size)

        screen.blit(cloud_surface, (0, 0))


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
    Computes and manages a path from start to end point via a diversion point.
    """

    def __init__(self):
        self.waypoints = []
        self.dot_spacing = meters_to_pixels(0.5)  # 0.5m between dots
        self.dot_radius = 4
        self.dot_color = Colors.YELLOW

    def compute(self, start_pos, end_pos, diversion_point=None):
        """
        Compute the path from start to end position, going through diversion point.
        """
        if diversion_point is not None:
            # Path goes: start -> diversion -> end
            diversion_pos = diversion_point.center
            segment1 = self._compute_straight_line(start_pos, diversion_pos)
            segment2 = self._compute_straight_line(diversion_pos, end_pos)
            # Combine segments (remove duplicate point at junction)
            self.waypoints = segment1 + segment2[1:]
        else:
            # Direct path
            self.waypoints = self._compute_straight_line(start_pos, end_pos)

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


class DiversionPoint(CircleSprite):
    """
    A sprite that represents the diversion point for obstacle avoidance.
    Stays still by default, moves perpendicular to robot facing based on proximity overlap.
    Positioned along the path ahead of the robot.
    """

    def __init__(self, x, y, radius=None):
        if radius is None:
            radius = meters_to_pixels(0.3)
        super().__init__(x, y, radius, Colors.ORANGE)
        self.tag = "diversion"
        self.base_angular_speed = 90.0  # Degrees per second
        self.overlap_sum = 0.0  # Current proximity overlap value
        self.robot_facing = 0  # Robot's facing angle
        self.sensitivity = 20.0  # Sensitivity multiplier for movement
        self.active = False  # Whether diversion is active (obstacle detected)
        self.damping = 0.85  # Angular damping when no overlap (lower = faster return)
        self.fixed_radius = meters_to_pixels(3.0)  # Fixed distance from robot
        self.angular_offset = 0  # Degrees offset from path direction (positive = right, negative = left)

        # Burst feature - one-time acceleration when overlap exceeds threshold
        self.burst_threshold = 1.0  # Threshold for absolute overlap sum
        self.burst_triggered = False  # Whether burst has been triggered
        self.burst_acceleration = 30.0  # Degrees to add on burst

    def update_from_overlap(self, overlap_sum, abs_overlap_sum, robot_facing, robot_center, destination_center, dt):
        """Update position based on proximity overlap sum. Stays at fixed radius from robot."""
        self.overlap_sum = overlap_sum
        self.robot_facing = robot_facing

        # Calculate base angle from robot to destination
        dx = destination_center[0] - robot_center[0]
        dy = destination_center[1] - robot_center[1]
        dist_to_dest = math.sqrt(dx * dx + dy * dy)

        if dist_to_dest > 1:
            # Base angle pointing towards destination
            base_angle = math.degrees(math.atan2(dy, dx))

            # Check for burst trigger - one-time acceleration when absolute overlap exceeds threshold
            if abs_overlap_sum >= self.burst_threshold and not self.burst_triggered:
                self.burst_triggered = True
                # Randomly pick left or right
                burst_direction = random.choice([-1, 1])
                self.angular_offset += burst_direction * self.burst_acceleration

            # Reset burst trigger when overlap drops below threshold
            if abs_overlap_sum < self.burst_threshold * 0.5:
                self.burst_triggered = False

            # Angular speed scales with overlap magnitude
            if abs(overlap_sum) > 0.001:
                self.active = True

                # Direction based on sign of overlap
                # Negative overlap (obstacle on left) -> rotate right (positive angle)
                # Positive overlap (obstacle on right) -> rotate left (negative angle)
                direction = -1 if overlap_sum > 0 else 1

                # Angular speed proportional to overlap magnitude
                angular_speed = self.base_angular_speed * abs(overlap_sum) * self.sensitivity
                self.angular_offset += direction * angular_speed * dt
            else:
                # Dampen angular offset back towards zero when no overlap
                # Lower damping value = faster return to center
                self.angular_offset *= self.damping

                if abs(self.angular_offset) < 0.5:
                    self.active = False
                    self.angular_offset = 0

            # Clamp angular_offset to keep final angle within +/-90 degrees of robot facing
            # Normalize the difference between base_angle and robot_facing to -180..180
            angle_diff = base_angle - robot_facing
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            # final_angle = base_angle + angular_offset
            # We want: robot_facing - 90 <= final_angle <= robot_facing + 90
            # So: -90 - angle_diff <= angular_offset <= 90 - angle_diff
            min_offset = -90 - angle_diff
            max_offset = 90 - angle_diff
            self.angular_offset = max(min_offset, min(max_offset, self.angular_offset))

            # Calculate final position at fixed radius with angular offset
            final_angle = math.radians(base_angle + self.angular_offset)
            final_x = robot_center[0] + self.fixed_radius * math.cos(final_angle)
            final_y = robot_center[1] + self.fixed_radius * math.sin(final_angle)

            # Set position (adjusting for sprite origin being top-left)
            self.x = final_x - self.radius
            self.y = final_y - self.radius

    def reset_to_path(self, robot_center, destination_center):
        """Reset position to be along the path ahead of the robot."""
        dx = destination_center[0] - robot_center[0]
        dy = destination_center[1] - robot_center[1]
        dist_to_dest = math.sqrt(dx * dx + dy * dy)

        if dist_to_dest > 1:
            angle = math.atan2(dy, dx)
            self.x = robot_center[0] + self.fixed_radius * math.cos(angle) - self.radius
            self.y = robot_center[1] + self.fixed_radius * math.sin(angle) - self.radius

        self.angular_offset = 0
        self.active = False

    def draw_diversion(self, screen):
        """Draw the diversion point with indicator."""
        cx, cy = self.center
        int_cx, int_cy = int(cx), int(cy)
        int_radius = int(self.radius)

        # Draw outer ring - color indicates state
        if self.active:
            ring_color = Colors.YELLOW
        else:
            ring_color = Colors.ORANGE
        pygame.draw.circle(screen, ring_color, (int_cx, int_cy), int_radius + 4, 3)

        # Draw inner circle
        pygame.draw.circle(screen, Colors.ORANGE, (int_cx, int_cy), int_radius)
        pygame.draw.circle(screen, Colors.WHITE, (int_cx, int_cy), int_radius, 2)

        # Draw offset indicator showing angular displacement
        if abs(self.angular_offset) > 2:
            # Draw inner color showing direction of offset
            bar_color = Colors.CYAN if self.angular_offset > 0 else Colors.MAGENTA
            inner_radius = max(3, int(min(abs(self.angular_offset) / 10, int_radius - 2)))
            pygame.draw.circle(screen, bar_color, (int_cx, int_cy), inner_radius)

    def _draw(self, screen):
        """Override to use custom drawing."""
        if self.visible:
            self.draw_diversion(screen)


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

        # Proximity cloud for obstacle detection
        self.proximity_cloud = ProximityCloud(num_points=100)
        self.diversion_point = None  # Will be set by game
        self.overlap_sum = 0.0  # Current overlap value for display
        self.show_proximity_cloud = True  # Toggle for visibility

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
        """Recompute the path through diversion point."""
        if self.destination is None:
            return

        # Path always goes through diversion point: robot -> diversion -> destination
        self.path.compute(
            self.center,
            self.destination.center,
            self.diversion_point
        )

        # Also check FOV collision for visual feedback
        self.fov_sees_obstacle = self.check_fov_obstacle_collision(self.obstacle)

    def on_update(self, dt):
        # Generate proximity cloud in FOV
        fov_range = meters_to_pixels(self.FOV_RANGE_METERS)
        self.proximity_cloud.generate(
            self.center,
            self.facing_angle,
            self.FOV_ANGLE,
            fov_range
        )

        # Calculate overlap sum with obstacle
        self.overlap_sum, self.abs_overlap_sum = self.proximity_cloud.get_obstacle_overlap_sum(self.obstacle)

        # Update diversion point based on overlap
        if self.diversion_point is not None and self.destination is not None:
            self.diversion_point.update_from_overlap(
                self.overlap_sum,
                self.abs_overlap_sum,
                self.facing_angle,
                self.center,
                self.destination.center,
                dt
            )

        # Update path to go through diversion point
        self.update_path()

        if self.moving and self.destination and not self.dragging:
            # Always follow the path (which goes through diversion point)
            target = self.path.get_next_waypoint(self.center)

            if target and not self.path.is_complete(self.center):
                dx = target[0] - self.center[0]
                dy = target[1] - self.center[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > meters_to_pixels(0.1):
                    # Update facing angle based on movement direction
                    self.facing_angle = math.degrees(math.atan2(dy, dx))

                    # Move towards target
                    self.x += (dx / distance) * self.speed * dt
                    self.y += (dy / distance) * self.speed * dt
        elif not self.moving and self.destination:
            # When stopped, face towards destination
            dx = self.destination.center[0] - self.center[0]
            dy = self.destination.center[1] - self.center[1]
            if dx != 0 or dy != 0:
                self.facing_angle = math.degrees(math.atan2(dy, dx))

    def draw_fov(self, screen):
        """Draw the proximity cloud."""
        # Draw proximity cloud
        if self.show_proximity_cloud:
            self.proximity_cloud.draw(screen)

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
        self.diversion_point = None
        self.button = None
        self.sliders = []  # List of all sliders
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

        # Create obstacle at center, slightly below middle
        obs_width = meters_to_pixels(4.0)
        obs_height = meters_to_pixels(2.0)
        obs_x = meters_to_pixels(16.0) - obs_width / 2
        obs_y = meters_to_pixels(10.5) - obs_height / 2
        self.obstacle = Obstacle(obs_x, obs_y, obs_width, obs_height)
        self.obstacle.angle = 15  # Tilt slightly
        self.add(self.obstacle)

        # Create diversion point (will be positioned along path to destination)
        self.diversion_point = DiversionPoint(0, 0)
        self.add(self.diversion_point)

        # Link robot to destination, obstacle, and diversion point
        self.robot.destination = self.destination
        self.robot.obstacle = self.obstacle
        self.robot.diversion_point = self.diversion_point

        # Create UI elements
        self.button = Button(
            self.width // 2 - 50, 20, 100, 40,
            "Start", Colors.GREEN, (0, 200, 0)
        )

        # Create sliders for adjustable parameters
        slider_width = 150
        slider_height = 16
        left_x = 15
        right_x = self.width - slider_width - 15

        # Left side sliders
        self.speed_slider = Slider(
            left_x, 80, slider_width, slider_height,
            0.5, 5.0, 2.0, "Speed", " m/s"
        )
        self.sliders.append(self.speed_slider)

        self.sensitivity_slider = Slider(
            left_x, 130, slider_width, slider_height,
            0.0, 100.0, 20.0, "Sensitivity", "x"
        )
        self.sliders.append(self.sensitivity_slider)

        self.damping_slider = Slider(
            left_x, 180, slider_width, slider_height,
            0.9, 0.999, 0.99, "Damping", ""
        )
        self.sliders.append(self.damping_slider)

        self.burst_threshold_slider = Slider(
            left_x, 230, slider_width, slider_height,
            0.1, 5.0, 1.0, "Burst Thresh", ""
        )
        self.sliders.append(self.burst_threshold_slider)

        # Right side sliders
        self.fov_angle_slider = Slider(
            right_x, 80, slider_width, slider_height,
            30, 120, 70, "FOV Angle", "°"
        )
        self.sliders.append(self.fov_angle_slider)

        self.fov_range_slider = Slider(
            right_x, 130, slider_width, slider_height,
            1.0, 10.0, 4.0, "FOV Range", " m"
        )
        self.sliders.append(self.fov_range_slider)

        self.num_points_slider = Slider(
            right_x, 180, slider_width, slider_height,
            20, 200, 100, "Points", ""
        )
        self.sliders.append(self.num_points_slider)

        self.radius_slider = Slider(
            right_x, 230, slider_width, slider_height,
            1.0, 8.0, 3.0, "Radius", " m"
        )
        self.sliders.append(self.radius_slider)

        # Initialize diversion point position
        self.diversion_point.reset_to_path(self.robot.center, self.destination.center)

        print("Robot Destination Game with Proximity Cloud Navigation")
        print("- Drag the blue robot or green destination anywhere")
        print("- Drag the red obstacle to move it, drag corners to resize")
        print("- Orange diversion point moves perpendicular to avoid obstacles")
        print("- Green dots = right side proximity, Red dots = left side")
        print("- Mouse wheel over obstacle to rotate it")

    def on_update(self, dt):
        mouse_pos = pygame.mouse.get_pos()

        self.button.update(mouse_pos)

        # Update all sliders
        for slider in self.sliders:
            slider.handle_mouse_move(mouse_pos)

        # Apply slider values to robot and diversion point
        self.robot.speed = meters_to_pixels(self.speed_slider.value)
        self.robot.FOV_ANGLE = self.fov_angle_slider.value
        self.robot.FOV_RANGE_METERS = self.fov_range_slider.value
        self.robot.proximity_cloud.num_points = int(self.num_points_slider.value)

        self.diversion_point.sensitivity = self.sensitivity_slider.value
        self.diversion_point.damping = self.damping_slider.value
        self.diversion_point.fixed_radius = meters_to_pixels(self.radius_slider.value)
        self.diversion_point.burst_threshold = self.burst_threshold_slider.value

        if self.dragged_sprite:
            self.dragged_sprite.update_drag(mouse_pos)

    def on_draw(self, screen):
        # Draw FOV first (behind path)
        self.robot.draw_fov(screen)

        # Draw path
        self.robot.draw_path(screen)

        # Draw button
        self.button.draw(screen)

        # Draw all sliders
        for slider in self.sliders:
            slider.draw(screen)

        # Draw overlap sum indicator (visual feedback)
        font = pygame.font.Font(None, 24)
        overlap_color = Colors.RED if self.robot.overlap_sum < 0 else Colors.GREEN
        if abs(self.robot.overlap_sum) < 0.001:
            overlap_color = Colors.WHITE
        overlap_text = font.render(f"Overlap: {self.robot.overlap_sum:.4f}", True, overlap_color)
        screen.blit(overlap_text, (self.width // 2 - 60, 70))

        # Draw diversion status
        status = "ACTIVE" if self.diversion_point.active else "IDLE"
        status_color = Colors.YELLOW if self.diversion_point.active else Colors.GRAY
        status_text = font.render(f"Diversion: {status}", True, status_color)
        screen.blit(status_text, (self.width // 2 - 60, 90))

        # Draw scale indicator
        font_small = pygame.font.Font(None, 20)
        scale_text = font_small.render("Scale: 1m = {:.0f}px".format(PPM), True, Colors.LIGHT_GRAY)
        screen.blit(scale_text, (10, self.height - 25))

        # Draw coordinate info
        coord_text = font_small.render("Space: 32m x 18m", True, Colors.LIGHT_GRAY)
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

        # Check all sliders
        for slider in self.sliders:
            if slider.handle_mouse_down(pos):
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

        # Allow dragging robot, destination, obstacle, and diversion point
        for sprite in [self.robot, self.destination, self.obstacle, self.diversion_point]:
            if sprite.contains_point(pos):
                sprite.start_drag(pos)
                self.dragged_sprite = sprite
                break

    def _handle_mouse_up(self, pos, button):
        """Handle mouse button release."""
        if button != 1:
            return

        # Release all sliders
        for slider in self.sliders:
            slider.handle_mouse_up()

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
