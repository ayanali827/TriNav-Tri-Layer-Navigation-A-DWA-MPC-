import numpy as np
import heapq
from utils.geometry import perpendicular_distance, distance

class AStarPlanner:
    def __init__(self, grid_size=0.2, robot_radius=0.4):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        
    def create_grid(self, obstacles, xlim=(-10, 10), ylim=(-5, 15)):
        min_x, max_x = xlim
        min_y, max_y = ylim
        width = int((max_x - min_x) / self.grid_size) + 1
        height = int((max_y - min_y) / self.grid_size) + 1
        
        grid = np.zeros((width, height))
        
        for obs in obstacles:
            x, y, r = obs
            obs_min_x = int((x - r - min_x) / self.grid_size)
            obs_max_x = int((x + r - min_x) / self.grid_size + 1)
            obs_min_y = int((y - r - min_y) / self.grid_size)
            obs_max_y = int((y + r - min_y) / self.grid_size + 1)
            
            obs_min_x = max(0, min(width-1, obs_min_x))
            obs_max_x = max(0, min(width-1, obs_max_x))
            obs_min_y = max(0, min(height-1, obs_min_y))
            obs_max_y = max(0, min(height-1, obs_max_y))
            
            grid[obs_min_x:obs_max_x+1, obs_min_y:obs_max_y+1] = 1
        
        return grid, (min_x, min_y)
    
    def world_to_grid(self, pos, grid_origin):
        x, y = pos
        min_x, min_y = grid_origin
        grid_x = int((x - min_x) / self.grid_size)
        grid_y = int((y - min_y) / self.grid_size)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos, grid_origin):
        grid_x, grid_y = grid_pos
        min_x, min_y = grid_origin
        x = min_x + grid_x * self.grid_size
        y = min_y + grid_y * self.grid_size
        return (x, y)
    
    def heuristic(self, a, b):
        return distance(a, b)
    
    def a_star_search(self, start, goal, grid, grid_origin):
        start = self.world_to_grid(start, grid_origin)
        goal = self.world_to_grid(goal, grid_origin)
        
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return None
        
        neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                return [self.grid_to_world(p, grid_origin) for p in path]
            
            close_set.add(current)
            for i,j in neighbors:
                neighbor = current[0]+i, current[1]+j
                
                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    
                    if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None
    
    def simplify_path(self, path):
        if len(path) < 3:
            return path
            
        max_dist = 0
        index = 0
        end = len(path) - 1
        
        for i in range(1, end):
            dist = perpendicular_distance(np.array(path[i]), 
                                         np.array(path[0]), 
                                         np.array(path[end]))
            if dist > max_dist:
                index = i
                max_dist = dist
                
        if max_dist > self.grid_size * 0.9:
            rec_results1 = self.simplify_path(path[:index+1])
            rec_results2 = self.simplify_path(path[index:])
            return rec_results1[:-1] + rec_results2
        else:
            return [path[0], path[-1]]
    
    def plan_path(self, start, goal, obstacles):
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        
        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                xs.append(obs[0] - obs[2])
                xs.append(obs[0] + obs[2])
                ys.append(obs[1] - obs[2])
                ys.append(obs[1] + obs[2])
        
        padding = self.robot_radius + 0.5
        xlim = (min(xs) - padding, max(xs) + padding)
        ylim = (min(ys) - padding, max(ys) + padding)
        
        grid, grid_origin = self.create_grid(obstacles, xlim=xlim, ylim=ylim)
        path = self.a_star_search(start, goal, grid, grid_origin)
        if path:
            return self.simplify_path(path)
        return None