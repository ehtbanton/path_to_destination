"""
Example script - Demonstrates how to create custom sprites.

This shows:
- Creating a player sprite with keyboard controls
- Creating enemy sprites that move on their own
- Handling collisions
- Using the simple sprite API
"""

import pygame
from engine import Sprite, Colors


class Player(Sprite):
    """A player-controlled sprite that moves with arrow keys or WASD."""

    def __init__(self, x, y):
        super().__init__(x, y, width=40, height=40, color=Colors.CYAN)
        self.speed = 200
        self.tag = "player"

    def on_create(self):
        print("Player created! Use arrow keys or WASD to move.")

    def on_key_held(self, keys):
        # Reset velocity
        self.vx = 0
        self.vy = 0

        # Check movement keys
        if pygame.K_LEFT in keys or pygame.K_a in keys:
            self.vx = -self.speed
        if pygame.K_RIGHT in keys or pygame.K_d in keys:
            self.vx = self.speed
        if pygame.K_UP in keys or pygame.K_w in keys:
            self.vy = -self.speed
        if pygame.K_DOWN in keys or pygame.K_s in keys:
            self.vy = self.speed

    def on_update(self, dt):
        # Keep player on screen
        if self.x < 0:
            self.x = 0
        if self.x > self._game.width - self.width:
            self.x = self._game.width - self.width
        if self.y < 0:
            self.y = 0
        if self.y > self._game.height - self.height:
            self.y = self._game.height - self.height

    def on_collision(self, other):
        if other.tag == "enemy":
            print("Ouch! Hit an enemy!")


class Enemy(Sprite):
    """A simple enemy that bounces around the screen."""

    def __init__(self, x, y):
        super().__init__(x, y, width=30, height=30, color=Colors.RED)
        self.tag = "enemy"

    def on_create(self):
        # Start with random velocity
        import random
        self.vx = random.choice([-1, 1]) * random.randint(50, 150)
        self.vy = random.choice([-1, 1]) * random.randint(50, 150)

    def on_update(self, dt):
        # Bounce off screen edges
        if self.x <= 0 or self.x >= self._game.width - self.width:
            self.vx *= -1
            self.x = max(0, min(self.x, self._game.width - self.width))
        if self.y <= 0 or self.y >= self._game.height - self.height:
            self.vy *= -1
            self.y = max(0, min(self.y, self._game.height - self.height))


class Collectible(Sprite):
    """A collectible item that disappears when touched by the player."""

    score = 0  # Class variable to track score

    def __init__(self, x, y):
        super().__init__(x, y, width=20, height=20, color=Colors.YELLOW)
        self.tag = "collectible"

    def on_collision(self, other):
        if other.tag == "player":
            Collectible.score += 1
            print(f"Score: {Collectible.score}")
            self.destroy()


def setup(game):
    """
    Setup function called by main.py to initialize the game.
    Add your sprites here!
    """
    # Set background color
    game.background_color = Colors.DARK_GRAY

    # Create player in center
    player = Player(game.width // 2 - 20, game.height // 2 - 20)
    game.add(player)

    # Create some enemies
    game.add(Enemy(100, 100))
    game.add(Enemy(600, 100))
    game.add(Enemy(100, 400))
    game.add(Enemy(600, 400))

    # Create some collectibles
    import random
    for _ in range(5):
        x = random.randint(50, game.width - 70)
        y = random.randint(50, game.height - 70)
        game.add(Collectible(x, y))

    print("Game started! Collect the yellow squares, avoid the red ones!")
