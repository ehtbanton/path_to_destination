"""
Main entry point for the pygame application.

Edit this file to customize your game!
The example script in /scripts shows how to create custom sprites.

Quick Start:
    1. Run: python startup.py
    2. Edit scripts/example.py to modify game behavior
    3. Or create your own sprites right here in main.py

Sprite API:
    - Inherit from Sprite and override on_update(dt), on_collision(other), etc.
    - Use game.add(sprite) to add sprites
    - Access game.width, game.height for screen dimensions
    - Use Colors.RED, Colors.BLUE, etc. for common colors
"""

from engine import Game, Sprite, Colors

# Import the example script's setup function
from scripts.example import setup as example_setup


class MyGame(Game):
    """
    Your custom game class.
    Override on_setup() to add your own sprites and logic.
    """

    def on_setup(self):
        """Called once when the game starts. Add your sprites here."""
        # Use the example setup - replace this with your own code!
        example_setup(self)

        # --- Add your own sprites below ---
        # Example:
        # my_sprite = Sprite(100, 100, width=50, height=50, color=Colors.GREEN)
        # self.add(my_sprite)

    def on_update(self, dt):
        """Called every frame. Add global game logic here."""
        pass

    def on_draw(self, screen):
        """Called after sprites are drawn. Add custom drawing here."""
        # Example: Draw text
        # font = pygame.font.Font(None, 36)
        # text = font.render("Hello!", True, Colors.WHITE)
        # screen.blit(text, (10, 10))
        pass


def main():
    """Create and run the game."""
    game = MyGame(
        width=800,
        height=600,
        title="Pygame Sprite Engine",
        fps=60
    )
    game.run()


if __name__ == "__main__":
    main()
