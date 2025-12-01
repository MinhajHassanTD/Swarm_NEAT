"""
Maze environment with BFS-based food placement.
"""
import copy
import random
from collections import deque

class Maze:
    _distance_cache = {}  # ⭐ CLASS-LEVEL CACHE
    
    def __init__(self, layout, cell_size=40, num_small_food=10, num_big_food=5):
        self.layout = layout
        self.cell_size = cell_size
        self.rows = len(layout)
        self.cols = max(len(row) for row in layout) if layout else 0
        
        # Find start position
        self.start_pos = None
        for y, row in enumerate(layout):
            for x, cell in enumerate(row):
                if cell == 'S':
                    self.start_pos = (x, y)
                    break
            if self.start_pos:
                break
        
        if not self.start_pos:
            raise ValueError("Maze must have start position 'S'")
        
        # Generate food
        self.food_items = []
        self._generate_food_by_distance(num_small_food, num_big_food)
    
    def _calculate_distance_map(self):
        """BFS to calculate distance from start (cached for performance)."""
        # ⭐ CACHE CHECK
        layout_key = tuple(tuple(row) for row in self.layout)
        cache_key = (layout_key, self.start_pos)
        
        if cache_key in Maze._distance_cache:
            return Maze._distance_cache[cache_key]
        
        # Calculate BFS
        distance_map = {self.start_pos: 0}
        queue = deque([self.start_pos])
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            current_dist = distance_map[(x, y)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= ny < self.rows and 
                    0 <= nx < len(self.layout[ny]) and 
                    self.layout[ny][nx] != '1' and 
                    (nx, ny) not in distance_map):
                    
                    distance_map[(nx, ny)] = current_dist + 1
                    queue.append((nx, ny))
        
        # ⭐ CACHE STORE
        Maze._distance_cache[cache_key] = distance_map
        return distance_map
    
    def _generate_food_by_distance(self, num_small, num_big):
        """Place food prioritizing distant locations."""
        distance_map = self._calculate_distance_map()
        
        # Filter valid cells (minimum distance 5, fallback to 3)
        valid_cells = [(d, x, y) for (x, y), d in distance_map.items() if d >= 5]
        if not valid_cells:
            valid_cells = [(d, x, y) for (x, y), d in distance_map.items() if d >= 3]
        if not valid_cells:
            valid_cells = [(d, x, y) for (x, y), d in distance_map.items()]
        
        # Sort by distance (farthest first)
        valid_cells.sort(key=lambda item: item[0], reverse=True)
        
        # Split zones: farthest 33% vs remaining
        farthest_count = max(1, len(valid_cells) // 3)
        farthest = valid_cells[:farthest_count]
        remaining = valid_cells[farthest_count:]
        
        # ⭐ HELPER METHOD FOR CONSISTENCY
        def add_food(x, y, is_big):
            self.food_items.append({
                'grid_x': x,
                'grid_y': y,
                'big': is_big,
                'eaten': False
            })
        
        # Place big food in farthest zone
        for _ in range(num_big):
            if farthest:
                dist, x, y = random.choice(farthest)
                add_food(x, y, True)
                farthest.remove((dist, x, y))
            elif remaining:
                dist, x, y = random.choice(remaining)
                add_food(x, y, True)
                remaining.remove((dist, x, y))
        
        # Place small food in remaining cells
        all_remaining = farthest + remaining
        for _ in range(num_small):
            if all_remaining:
                dist, x, y = random.choice(all_remaining)
                add_food(x, y, False)
                all_remaining.remove((dist, x, y))
    
    def copy_with_fresh_food(self):
        """Create copy with independent food state."""
        new_maze = Maze.__new__(Maze)
        new_maze.layout = self.layout
        new_maze.cell_size = self.cell_size
        new_maze.rows = self.rows
        new_maze.cols = self.cols
        new_maze.start_pos = self.start_pos
        new_maze.food_items = copy.deepcopy(self.food_items)
        return new_maze
    
    def is_wall(self, grid_x, grid_y):
        if grid_y < 0 or grid_y >= self.rows:
            return True
        if grid_x < 0 or grid_x >= len(self.layout[grid_y]):
            return True
        return self.layout[grid_y][grid_x] == '1'
    
    def get_food_at(self, grid_x, grid_y):
        for food in self.food_items:
            if food['grid_x'] == grid_x and food['grid_y'] == grid_y and not food['eaten']:
                return food
        return None
    
    def to_pixel(self, grid_x, grid_y):
        return (grid_x * self.cell_size + self.cell_size // 2,
                grid_y * self.cell_size + self.cell_size // 2)


DEFAULT_MAZE = [
    # "11111111111111111111111111",
    # "10010001111000000000000001",
    # "10010001000111111000000001",
    # "10000000001000001111100001",
    # "11110000111100000000000001",
    # "10000000000111100000000001",
    # "10001111100100000000000001",
    # "10000000100111111100000001",
    # "11100001000000000000011101",
    # "100000001111100S0000100001",
    # "10000000000001000000001001",
    # "10000011111000111100000001",
    # "10000010000000000000000001",
    # "10000010001111101000000001",
    # "10000000001000001000000001",
    # "10000111111000001111100001",
    # "10000000000000000000000001",
    # "11111111100001111111111001",
    # "10000000000000000000000001",
    # "11111111111111111111111111"
    "11111111111111111111111111",
    "10000000000001000000000001",
    "10111111111010111110111101",
    "10000000001000000010000001",
    "11111011111011111111101111",
    "10000000000000000000000001",
    "10111111111111111110111111",
    "10000000000000000000000001",
    "11111011111111111111111101",
    "1000000000000S000000000001",
    "10111111111111111110111111",
    "10000000000000000000000001",
    "11111111111011111111111101",
    "10000000000000000000000001",
    "10111111111111111111111111",
    "10000000000000000000000001",
    "10111111111011111111111011",
    "10000000000000000000000001",
    "10000000000000000000000001",
    "11111111111111111111111111"
]