import sys, os
sys.path.insert(0, os.path.abspath("."))
from env.maze_env import MazeEnv

env = MazeEnv(size=9, n_keys=1, n_traps=1)
obs, _ = env.reset(seed=42)
env.render()

print("\nTest 10 actions aléatoires:")
import random
for i in range(10):
    action = random.randint(0, 3)
    obs, reward, term, trunc, _ = env.step(action)
    print(f"  action={action}, reward={reward:.2f}, term={term}, trunc={trunc}")
    if term or trunc:
        print("  Episode terminé!")
        break