import numpy as np, random 

def generate_maze(size=15, n_keys=2, n_traps=3, seed=None): 
    if seed: random.seed(seed); np.random.seed(seed) 
    grid = np.zeros((size, size), dtype=int) 

    def carve(r, c): 
        dirs = [(0,2),(0,-2),(2,0),(-2,0)] 
        random.shuffle(dirs)
        for dr, dc in dirs: 
            nr, nc = r+dr, c+dc 
            if 0 < nr < size-1 and 0 < nc < size-1 and grid[nr][nc] == 0: 
                grid[r+dr//2][c+dc//2] = 1  # mur intermédiaire 
                grid[nr][nc] = 1 
                carve(nr, nc) 

    grid[1][1] = 1 
    carve(1, 1) 
    paths = random.sample(list(zip(*np.where(grid == 1))), size*size//2) 
    grid[size-2][size-2] = 5  # EXIT 
    for i in range(n_keys): 
        grid[paths.pop()] = 2  # KEY 
        grid[paths.pop()] = 3  # DOOR 
    for _ in range(n_traps): 
        grid[paths.pop()] = 4  # TRAP 
    return grid 