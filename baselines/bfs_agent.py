from collections import deque 
class BFSAgent: 
    def __init__(self, env): self.env = env 
    def decide(self, obs): 
        grid = self.env.grid 
        start = tuple(self.env.agent_pos) 
        size = self.env.size 
        queue = deque([(start, [])]) 
        visited = {start} 
        moves = [(-1,0),(1,0),(0,-1),(0,1)] 
        while queue: 
            (r, c), path = queue.popleft() 
            if grid[r][c] == 5: return path[0] if path else 0  # EXIT 
            for i, (dr, dc) in enumerate(moves): 
                nr, nc = r+dr, c+dc 
                if 0<=nr<size and 0<=nc<size and (nr,nc) not in visited and grid[nr][nc]!=0: 
                    visited.add((nr,nc)) 
                    queue.append(((nr,nc), path+[i])) 
        return 0