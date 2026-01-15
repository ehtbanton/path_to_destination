"""
Robot Destination script - A robot (blue circle) moves towards a destination (green circle).

Features:
- Blue circle (robot) that moves towards destination when started
- Green circle (destination) as the target
- Both sprites are draggable at all times
- Start/Stop button to control robot movement
- Speed slider to adjust robot speed
"""

import pygame
import math
from engine import Game, Sprite, Colors


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
    """The blue robot circle that moves towards the destination."""

    def __init__(self, x, y, radius=30):
        super().__init__(x, y, radius, Colors.BLUE)
        self.tag = "robot"
        self.moving = False
        self.speed = 100
        self.destination = None

    def on_update(self, dt):
        if self.moving and self.destination and not self.dragging:
            # Calculate direction to destination
            dx = self.destination.center[0] - self.center[0]
            dy = self.destination.center[1] - self.center[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > 5:  # Stop when close enough
                # Normalize and apply speed
                self.x += (dx / distance) * self.speed * dt
                self.y += (dy / distance) * self.speed * dt


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
        print("- Click Start to make the robot move towards the destination")
        print("- Use the slider to adjust robot speed")

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
