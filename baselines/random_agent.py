import random

class RandomAgent:
    """Sélectionne une action uniformément au hasard."""
    def decide(self, obs):
        return random.randint(0, 3)