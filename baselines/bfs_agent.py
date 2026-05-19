from collections import deque


class BFSAgent:
    """Agent heuristique BFS — connaît toute la grille (borne supérieure)."""

    def __init__(self, env):
        self.env = env

    def decide(self, obs):
        if not hasattr(self.env, 'grid') or self.env.grid is None:
            return 0

        grid  = self.env.grid
        start = tuple(self.env.agent_pos)
        size  = self.env.size
        keys  = self.env.keys_held

        queue   = deque([(start, [], keys)])
        visited = {(start, keys)}
        moves   = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            (r, c), path, k = queue.popleft()

            if grid[r][c] == 5:  # EXIT
                return path[0] if path else 0

            for i, (dr, dc) in enumerate(moves):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                cell = grid[nr][nc]
                if cell == 0 or cell == 4:   # WALL ou TRAP
                    continue
                if cell == 3 and k <= 0:     # DOOR sans clé
                    continue

                # FIX : empêcher les clés négatives
                if cell == 3:
                    new_k = max(0, k - 1)
                elif cell == 2 and k == 0:
                    new_k = 1
                else:
                    new_k = k

                state = ((nr, nc), new_k)
                if state not in visited:
                    visited.add(state)
                    queue.append(((nr, nc), path + [i], new_k))

        return 0