import numpy as np
import random

WALL = 0
PATH = 1
KEY  = 2
DOOR = 3
TRAP = 4
EXIT = 5

def generate_maze(size=15, n_keys=2, n_traps=3, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.zeros((size, size), dtype=int)

    def carve(r, c):
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr][nc] == 0:
                grid[r + dr // 2][c + dc // 2] = PATH
                grid[nr][nc] = PATH
                carve(nr, nc)

    grid[1][1] = PATH
    carve(1, 1)

    # Collecte toutes les cases PATH disponibles (hors start et exit)
    path_cells = [(r, c) for r in range(size) for c in range(size)
                  if grid[r][c] == PATH and (r, c) != (1, 1) and (r, c) != (size - 2, size - 2)]
    random.shuffle(path_cells)

    # Placer la sortie
    grid[size - 2][size - 2] = EXIT

    # Placer clés et portes
    for _ in range(n_keys):
        if path_cells:
            grid[path_cells.pop()] = KEY
        if path_cells:
            grid[path_cells.pop()] = DOOR

    # Placer les pièges
    for _ in range(n_traps):
        if path_cells:
            grid[path_cells.pop()] = TRAP

    return grid