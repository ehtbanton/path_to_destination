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

from scripts.robot_destination import create_game


def main():
    """Create and run the game."""
    game = create_game()
    game.run()


if __name__ == "__main__":
    main()
