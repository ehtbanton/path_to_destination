"""
Robot Destination script - A robot (blue circle) moves towards a destination (green circle).

Features:
- Blue circle (robot) that moves towards destination when started
- Green circle (destination) as the target
- Both sprites are draggable at all times
- Start/Stop button to control robot movement
- Speed slider to adjust robot speed
- Path computation with dotted line visualization
"""

import pygame
import math
from engine import Game, Sprite, Colors


class Path:
    """
    Computes and manages a path from start to end point.

    The path is represented as a list of waypoints that can be followed.
    This architecture supports future enhancements like curved paths,
    obstacle avoidance, or dynamic path modifications.
    """

    def __init__(self):
        self.waypoints = []  # List of (x, y) tuples
        self.dot_spacing = 20  # Pixels between dots
        self.dot_radius = 3
        self.dot_color = Colors.YELLOW

    def compute(self, start_pos, end_pos):
        """
        Compute the path from start to end position.

        Currently computes a straight line, but this method can be
        overridden or modified to compute more complex paths.

        Args:
            start_pos: (x, y) tuple for starting position
            end_pos: (x, y) tuple for ending position
        """
        self.waypoints = self._compute_straight_line(start_pos, end_pos)

    def _compute_straight_line(self, start_pos, end_pos):
        """
        Compute waypoints along a straight line.

        Args:
            start_pos: (x, y) tuple for starting position
            end_pos: (x, y) tuple for ending position

        Returns:
            List of (x, y) waypoint tuples
        """
        waypoints = []

        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1:
            return [end_pos]

        # Normalize direction
        dir_x = dx / distance
        dir_y = dy / distance

        # Generate waypoints along the line
        num_points = max(2, int(distance / self.dot_spacing))
        for i in range(num_points + 1):
            t = i / num_points
            x = start_pos[0] + dx * t
            y = start_pos[1] + dy * t
            waypoints.append((x, y))

        return waypoints

    def get_next_waypoint(self, current_pos, threshold=10):
        """
        Get the next waypoint to move towards.

        Args:
            current_pos: Current (x, y) position
            threshold: Distance threshold to consider a waypoint reached

        Returns:
            Next waypoint (x, y) or None if path is complete
        """
        if not self.waypoints:
            return None

        # Find the first waypoint that hasn't been reached yet
        for waypoint in self.waypoints:
            dx = waypoint[0] - current_pos[0]
            dy = waypoint[1] - current_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > threshold:
                return waypoint

        # All waypoints reached, return the final one
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

    def is_complete(self, current_pos, threshold=10):
        """Check if the path has been completed."""
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
    """The blue robot circle that follows a computed path to the destination."""

    def __init__(self, x, y, radius=30):
        super().__init__(x, y, radius, Colors.BLUE)
        self.tag = "robot"
        self.moving = False
        self.speed = 100
        self.destination = None
        self.path = Path()
        self._last_robot_pos = None
        self._last_dest_pos = None

    def update_path(self):
        """Recompute the path if robot or destination has moved."""
        if self.destination is None:
            return

        current_robot_pos = self.center
        current_dest_pos = self.destination.center

        # Check if positions have changed significantly
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
            needs_update = robot_moved or dest_moved

        if needs_update:
            self.path.compute(current_robot_pos, current_dest_pos)
            self._last_robot_pos = current_robot_pos
            self._last_dest_pos = current_dest_pos

    def on_update(self, dt):
        # Always update path when positions change
        self.update_path()

        if self.moving and self.destination and not self.dragging:
            # Get next waypoint from path
            next_waypoint = self.path.get_next_waypoint(self.center)

            if next_waypoint and not self.path.is_complete(self.center):
                # Calculate direction to next waypoint
                dx = next_waypoint[0] - self.center[0]
                dy = next_waypoint[1] - self.center[1]
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > 5:
                    # Normalize and apply speed
                    self.x += (dx / distance) * self.speed * dt
                    self.y += (dy / distance) * self.speed * dt

    def draw_path(self, screen):
        """Draw the path visualization."""
        self.path.draw(screen)


class Destination(CircleSprite):
    """The green destination circle."""

    def __init__(self, x, y, radius=25):
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

    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
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

        # Draw label
        label_text = self.font.render(f"{self.label}: {int(self.value)}", True, Colors.WHITE)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

        # Draw track
        track_rect = pygame.Rect(
            self.rect.x,
            self.rect.y + self.rect.height // 2 - 3,
            self.rect.width,
            6
        )
        pygame.draw.rect(screen, Colors.GRAY, track_rect, border_radius=3)

        # Draw filled portion
        filled_width = self.knob_x - self.rect.x
        if filled_width > 0:
            filled_rect = pygame.Rect(
                self.rect.x,
                self.rect.y + self.rect.height // 2 - 3,
                filled_width,
                6
            )
            pygame.draw.rect(screen, Colors.CYAN, filled_rect, border_radius=3)

        # Draw knob
        pygame.draw.circle(
            screen,
            Colors.WHITE,
            (int(self.knob_x), self.rect.y + self.rect.height // 2),
            self.knob_radius
        )

    def handle_mouse_down(self, mouse_pos):
        # Check if clicking on knob or track
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

    def __init__(self, width=800, height=600, title="Robot to Destination", fps=60):
        super().__init__(width, height, title, fps)
        self.robot = None
        self.destination = None
        self.button = None
        self.slider = None
        self.is_moving = False
        self.dragged_sprite = None

    def on_setup(self):
        self.background_color = Colors.DARK_GRAY

        # Create robot (blue circle) in the left area
        self.robot = Robot(150, self.height // 2 - 30)
        self.add(self.robot)

        # Create destination (green circle) in the right area
        self.destination = Destination(self.width - 200, self.height // 2 - 25)
        self.add(self.destination)

        # Link robot to destination
        self.robot.destination = self.destination

        # Create UI elements
        self.button = Button(
            self.width // 2 - 50, 20, 100, 40,
            "Start", Colors.GREEN, (0, 200, 0)
        )

        self.slider = Slider(
            self.width // 2 - 100, self.height - 50,
            200, 20, 50, 400, 100, "Speed"
        )

        print("Robot Destination Game")
        print("- Drag the blue robot or green destination anywhere")
        print("- Click Start to make the robot follow the path")
        print("- Use the slider to adjust robot speed")
        print("- Path updates dynamically as you drag!")

    def on_update(self, dt):
        mouse_pos = pygame.mouse.get_pos()

        # Update button hover state
        self.button.update(mouse_pos)

        # Update slider drag
        self.slider.handle_mouse_move(mouse_pos)

        # Update robot speed from slider
        self.robot.speed = self.slider.value

        # Update dragged sprite position
        if self.dragged_sprite:
            self.dragged_sprite.update_drag(mouse_pos)

    def on_draw(self, screen):
        # Draw path first (behind everything)
        self.robot.draw_path(screen)

        # Draw button
        self.button.draw(screen)

        # Draw slider
        self.slider.draw(screen)

        # Draw instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "Drag the circles to reposition them",
            "Click Start/Stop to control the robot"
        ]
        for i, text in enumerate(instructions):
            text_surface = font.render(text, True, Colors.LIGHT_GRAY)
            screen.blit(text_surface, (10, 10 + i * 25))

    def run(self):
        """Custom run loop with UI event handling."""
        self.running = True
        self.on_setup()

        while self.running:
            self.dt = self._clock.tick(self.fps) / 1000.0

            # Process new sprites
            for sprite in self._sprites_to_add:
                self._sprites.append(sprite)
                sprite.on_create()
            self._sprites_to_add.clear()

            # Handle events
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

            # Update sprites
            for sprite in self._sprites:
                sprite.on_key_held(self._keys_pressed)
                sprite._internal_update(self.dt)

            # Check collisions
            for i, sprite in enumerate(self._sprites):
                for other in self._sprites[i + 1:]:
                    if sprite.collides_with(other):
                        sprite.on_collision(other)
                        other.on_collision(sprite)

            # Custom update
            self.on_update(self.dt)

            # Remove destroyed sprites
            for sprite in self._sprites:
                if sprite._destroyed:
                    sprite.on_destroy()
            self._sprites = [s for s in self._sprites if not s._destroyed]

            # Draw
            self._screen.fill(self.background_color)

            # Draw path before sprites
            self.robot.draw_path(self._screen)

            for sprite in self._sprites:
                sprite._draw(self._screen)
            self.on_draw(self._screen)
            pygame.display.flip()

        pygame.quit()

    def _handle_mouse_down(self, pos, button):
        """Handle mouse button press."""
        if button != 1:  # Only handle left click
            return

        # Check slider first
        if self.slider.handle_mouse_down(pos):
            return

        # Check button click
        if self.button.is_clicked(pos):
            self.is_moving = not self.is_moving
            self.robot.moving = self.is_moving
            self.button.text = "Stop" if self.is_moving else "Start"
            self.button.color = Colors.RED if self.is_moving else Colors.GREEN
            self.button.hover_color = (200, 0, 0) if self.is_moving else (0, 200, 0)
            return

        # Check sprite dragging (robot has priority if overlapping)
        for sprite in [self.robot, self.destination]:
            if sprite.contains_point(pos):
                sprite.start_drag(pos)
                self.dragged_sprite = sprite
                break

    def _handle_mouse_up(self, pos, button):
        """Handle mouse button release."""
        if button != 1:
            return

        self.slider.handle_mouse_up()

        # Stop dragging sprites
        if self.dragged_sprite:
            self.dragged_sprite.stop_drag()
            self.dragged_sprite = None


def create_game():
    """Factory function to create the game instance."""
    return RobotDestinationGame(
        width=800,
        height=600,
        title="Robot to Destination",
        fps=60
    )
