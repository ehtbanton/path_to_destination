"""
Sprite class - The core programmable sprite with a simple API.
"""

import pygame
from .colors import Colors


class Sprite:
    """
    A programmable sprite with position, velocity, and custom behavior.

    Simple API:
        sprite.x, sprite.y       - Position
        sprite.vx, sprite.vy     - Velocity
        sprite.width, sprite.height - Size
        sprite.color             - Fill color (RGB tuple)
        sprite.image             - Pygame surface (auto-created or load your own)
        sprite.visible           - Whether to draw the sprite
        sprite.tag               - Custom string tag for identification
        sprite.data              - Dict for storing custom data

    Override these methods for custom behavior:
        on_create()              - Called once when sprite is added to game
        on_update(dt)            - Called every frame (dt = delta time in seconds)
        on_collision(other)      - Called when colliding with another sprite
        on_key_down(key)         - Called when a key is pressed
        on_key_held(keys)        - Called every frame with currently held keys
        on_mouse_click(pos, button) - Called on mouse click
        on_destroy()             - Called when sprite is removed
    """

    def __init__(self, x=0, y=0, width=32, height=32, color=None):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = width
        self.height = height
        self.color = color or Colors.WHITE
        self.visible = True
        self.tag = ""
        self.data = {}
        self._game = None
        self._destroyed = False
        self._custom_image = False
        self._image = None
        self._rebuild_image()

    def _rebuild_image(self):
        """Rebuild the sprite's surface based on current size and color."""
        if not self._custom_image:
            self._image = pygame.Surface((self.width, self.height))
            self._image.fill(self.color)

    @property
    def image(self):
        """Get the sprite's image surface."""
        return self._image

    @image.setter
    def image(self, surface):
        """Set a custom image surface."""
        self._image = surface
        self._custom_image = True
        self.width = surface.get_width()
        self.height = surface.get_height()

    @property
    def rect(self):
        """Get the sprite's bounding rectangle."""
        return pygame.Rect(self.x, self.y, self.width, self.height)

    @property
    def center(self):
        """Get the center position of the sprite."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @center.setter
    def center(self, pos):
        """Set the sprite's position by its center."""
        self.x = pos[0] - self.width / 2
        self.y = pos[1] - self.height / 2

    def set_color(self, color):
        """Change the sprite's color and rebuild its image."""
        self.color = color
        self._custom_image = False
        self._rebuild_image()

    def set_size(self, width, height):
        """Change the sprite's size and rebuild its image."""
        self.width = width
        self.height = height
        if not self._custom_image:
            self._rebuild_image()

    def load_image(self, path):
        """Load an image from a file path."""
        self._image = pygame.image.load(path).convert_alpha()
        self._custom_image = True
        self.width = self._image.get_width()
        self.height = self._image.get_height()

    def move(self, dx, dy):
        """Move the sprite by a delta amount."""
        self.x += dx
        self.y += dy

    def move_to(self, x, y):
        """Move the sprite to an absolute position."""
        self.x = x
        self.y = y

    def collides_with(self, other):
        """Check if this sprite collides with another."""
        return self.rect.colliderect(other.rect)

    def distance_to(self, other):
        """Get the distance to another sprite's center."""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def destroy(self):
        """Mark this sprite for removal."""
        self._destroyed = True

    # --- Override these methods for custom behavior ---

    def on_create(self):
        """Called once when the sprite is added to the game."""
        pass

    def on_update(self, dt):
        """Called every frame. dt is delta time in seconds."""
        pass

    def on_collision(self, other):
        """Called when this sprite collides with another."""
        pass

    def on_key_down(self, key):
        """Called when a key is pressed. key is a pygame key constant."""
        pass

    def on_key_held(self, keys):
        """Called every frame with the set of currently held keys."""
        pass

    def on_mouse_click(self, pos, button):
        """Called when mouse is clicked. pos is (x, y), button is 1/2/3."""
        pass

    def on_destroy(self):
        """Called when the sprite is about to be removed."""
        pass

    def _internal_update(self, dt):
        """Internal update - applies velocity and calls on_update."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.on_update(dt)

    def _draw(self, screen):
        """Draw the sprite to the screen."""
        if self.visible and self._image:
            screen.blit(self._image, (int(self.x), int(self.y)))
