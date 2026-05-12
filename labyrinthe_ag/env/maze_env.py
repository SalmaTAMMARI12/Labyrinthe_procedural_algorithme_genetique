import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .maze_generator import generate_maze

class MazeEnv(gym.Env):
    def __init__(self, size=15, n_keys=2, n_traps=3):
        super(MazeEnv, self).__init__()
        self.size = size
        self.n_keys = n_keys
        self.n_traps = n_traps
        # Actions: 0=Haut, 1=Bas, 2=Gauche, 3=Droite [cite: 73]
        self.action_space = spaces.Discrete(4)
        
        # Observations: Vue locale 5x5, clés portées, temps restant [cite: 68, 69, 70, 71]
        self.observation_space = spaces.Dict({
            "local_view": spaces.Box(low=0, high=5, shape=(5, 5), dtype=int),
            "keys_held": spaces.Discrete(n_keys + 1),
            "steps_left": spaces.Box(low=0, high=1, shape=(1,), dtype=float)
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = generate_maze(self.size, self.n_keys, self.n_traps, seed=seed)
        self.agent_pos = np.array([1, 1])
        self.keys_held = 0
        self.steps = 0
        self.max_steps = 500
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        # Logique de déplacement et calcul des récompenses selon le barème [cite: 75]
        # (Vérification murs, ramassage clés, ouverture portes, pièges)
        # ...
        return obs, reward, terminated, truncated, np.info({})