"""
Game class - Manages the game loop, sprites, and events.
"""

import pygame
from .colors import Colors


class Game:
    """
    The main game manager with a simple API.

    Simple API:
        game.width, game.height  - Screen dimensions
        game.background_color    - Background fill color
        game.fps                 - Target frames per second
        game.running             - Set to False to quit
        game.dt                  - Delta time of last frame (seconds)

    Methods:
        add(sprite)              - Add a sprite to the game
        remove(sprite)           - Remove a sprite from the game
        get_by_tag(tag)          - Get all sprites with a specific tag
        get_all()                - Get all sprites
        run()                    - Start the game loop
        quit()                   - Stop the game

    Override these for custom behavior:
        on_setup()               - Called once before the game starts
        on_update(dt)            - Called every frame
        on_draw(screen)          - Called after sprites are drawn
    """

    def __init__(self, width=800, height=600, title="Pygame App", fps=60):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.title = title
        self.background_color = Colors.BLACK
        self.running = False
        self.dt = 0

        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()
        self._sprites = []
        self._sprites_to_add = []
        self._keys_pressed = set()

    @property
    def screen(self):
        """Get the pygame screen surface for custom drawing."""
        return self._screen

    def add(self, sprite):
        """Add a sprite to the game."""
        self._sprites_to_add.append(sprite)
        sprite._game = self
        return sprite

    def remove(self, sprite):
        """Remove a sprite from the game."""
        sprite._destroyed = True

    def get_by_tag(self, tag):
        """Get all sprites with a specific tag."""
        return [s for s in self._sprites if s.tag == tag]

    def get_all(self):
        """Get all active sprites."""
        return list(self._sprites)

    def quit(self):
        """Stop the game loop."""
        self.running = False

    # --- Override these methods for custom behavior ---

    def on_setup(self):
        """Called once before the game loop starts. Add initial sprites here."""
        pass

    def on_update(self, dt):
        """Called every frame after sprite updates."""
        pass

    def on_draw(self, screen):
        """Called every frame after sprites are drawn. Add custom drawing here."""
        pass

    def run(self):
        """Start the game loop."""
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
                    for sprite in self._sprites:
                        sprite.on_mouse_click(event.pos, event.button)

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
