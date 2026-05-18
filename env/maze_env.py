import gymnasium as gym
from gymnasium import spaces
import numpy as np
from labyrinthe_ag.env.maze_generator import generate_maze

WALL = 0
PATH = 1
KEY  = 2
DOOR = 3
TRAP = 4
EXIT = 5

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, size=15, n_keys=2, n_traps=3, max_steps=500, render_mode=None):
        super().__init__()
        self.size      = size
        self.n_keys    = n_keys
        self.n_traps   = n_traps
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)  # 0=haut 1=bas 2=gauche 3=droite

        self.observation_space = spaces.Dict({
            "local_view": spaces.Box(low=0, high=5, shape=(5, 5), dtype=np.int32),
            "keys_held":  spaces.Discrete(n_keys + 1),
            "steps_left": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid      = generate_maze(self.size, self.n_keys, self.n_traps, seed=seed)
        self.agent_pos = np.array([1, 1], dtype=int)
        self.keys_held = 0
        self.steps     = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # haut bas gauche droite
        dr, dc = moves[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        reward     = 0.0
        terminated = False
        truncated  = False

        # Hors grille ou mur
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            reward = -0.05
        else:
            cell = self.grid[nr][nc]

            if cell == WALL:
                reward = -0.05  # coup invalide

            elif cell == KEY:
                self.agent_pos = np.array([nr, nc])
                self.keys_held += 1
                self.grid[nr][nc] = PATH
                reward = 1.0

            elif cell == DOOR:
                if self.keys_held > 0:
                    self.agent_pos = np.array([nr, nc])
                    self.keys_held -= 1
                    self.grid[nr][nc] = PATH
                    reward = 0.5
                else:
                    reward = -0.10  # porte fermée, pas de clé

            elif cell == TRAP:
                self.agent_pos = np.array([nr, nc])
                reward     = -2.0
                terminated = True  # fin d'épisode

            elif cell == EXIT:
                self.agent_pos = np.array([nr, nc])
                reward     = 10.0
                terminated = True

            else:  # PATH normal
                self.agent_pos = np.array([nr, nc])
                reward = -0.01

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        view = np.zeros((5, 5), dtype=np.int32)
        r, c = self.agent_pos
        for i in range(5):
            for j in range(5):
                gr = r + i - 2
                gc = c + j - 2
                if 0 <= gr < self.size and 0 <= gc < self.size:
                    view[i][j] = self.grid[gr][gc]
                else:
                    view[i][j] = WALL
        steps_left = np.array(
            [1.0 - self.steps / self.max_steps], dtype=np.float32
        )
        return {
            "local_view": view,
            "keys_held":  int(self.keys_held),
            "steps_left": steps_left,
        }

    # ------------------------------------------------------------------
    def render(self):
        symbols = {WALL: "█", PATH: " ", KEY: "K", DOOR: "D", TRAP: "X", EXIT: "E"}
        lines = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if np.array_equal(self.agent_pos, [r, c]):
                    row += "A"
                else:
                    row += symbols.get(self.grid[r][c], "?")
            lines.append(row)
        print("\n".join(lines))