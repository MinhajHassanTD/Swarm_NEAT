"""
Maze environment with walls and food items.
"""
import copy
import random
from collections import deque

class Maze:
    def __init__(self, layout, cell_size=40, num_small_food=43, num_big_food=12):
        """
        Initialize maze from a string layout.
        
        Args:
            layout: List of strings representing the maze grid:
                '1' = wall
                '0' = empty path
                'S' = start position
                'f' = small food
                'F' = big food
            cell_size: Pixel size of each grid cell for rendering
            num_small_food: Number of small food items to spawn
            num_big_food: Number of big food items to spawn
        """
        self.layout = layout
        self.cell_size = cell_size
        self.rows = len(layout)
        self.cols = len(layout[0]) if layout else 0
        self.num_small_food = num_small_food
        self.num_big_food = num_big_food
        
        # Parse maze layout
        self.start_pos = None
        self.food_items = []
        
        for y, row in enumerate(layout):
            for x, cell in enumerate(row):
                if cell == 'S':
                    self.start_pos = (x, y)
                elif cell == 'f':
                    self.food_items.append({
                        'grid_x': x,
                        'grid_y': y,
                        'big': False,
                        'eaten': False
                    })
                elif cell == 'F':
                    self.food_items.append({
                        'grid_x': x,
                        'grid_y': y,
                        'big': True,
                        'eaten': False
                    })
        
        if self.start_pos is None:
            raise ValueError("Maze must have a start position 'S'")
        
        # If no food in layout, generate random positions
        if len(self.food_items) == 0:
            self._randomize_food_positions()
    
    def _get_walkable_cells(self):
        """Get all walkable (non-wall) cells."""
        walkable = []
        for y in range(1, self.rows - 1):
            for x in range(1, self.cols - 1):
                if not self.is_wall(x, y) and (x, y) != self.start_pos:
                    walkable.append((x, y))
        return walkable
    
    def _bfs_spread_positions(self, walkable_cells, num_positions):
        """
        Use BFS to select well-spread positions.
        
        Args:
            walkable_cells: List of available (x, y) positions
            num_positions: Number of positions to select
        
        Returns:
            List of (x, y) positions
        """
        if num_positions >= len(walkable_cells):
            return walkable_cells
        
        selected = []
        remaining = set(walkable_cells)
        
        # Start with a random cell
        current = random.choice(list(remaining))
        selected.append(current)
        remaining.remove(current)
        
        while len(selected) < num_positions and remaining:
            # Find furthest cell from all selected cells using BFS
            max_min_dist = -1
            best_cell = None
            
            for candidate in remaining:
                # Calculate minimum distance to any selected cell
                min_dist = float('inf')
                for sel in selected:
                    dist = abs(candidate[0] - sel[0]) + abs(candidate[1] - sel[1])
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_cell = candidate
            
            if best_cell:
                selected.append(best_cell)
                remaining.remove(best_cell)
        
        return selected
    
    def _randomize_food_positions(self):
        """Generate random food positions using BFS spread."""
        walkable = self._get_walkable_cells()
        
        if len(walkable) < (self.num_small_food + self.num_big_food):
            print(f"⚠️  Not enough space for all food! Reducing amounts.")
            total = len(walkable)
            self.num_small_food = int(total * 0.8)
            self.num_big_food = total - self.num_small_food
        
        # Get well-spread positions
        all_positions = self._bfs_spread_positions(walkable, 
                                                   self.num_small_food + self.num_big_food)
        
        # Clear existing food
        self.food_items = []
        
        # Assign big food to first N positions, rest are small
        for i, (x, y) in enumerate(all_positions):
            is_big = i < self.num_big_food
            self.food_items.append({
                'grid_x': x,
                'grid_y': y,
                'big': is_big,
                'eaten': False
            })
    
    def randomize_food(self):
        """Public method to regenerate food positions."""
        self._randomize_food_positions()
    
    def copy_with_fresh_food(self):
        """Create a new maze instance with independent food state."""
        new_maze = Maze.__new__(Maze)
        new_maze.layout = self.layout  # Shared (immutable)
        new_maze.cell_size = self.cell_size
        new_maze.rows = self.rows
        new_maze.cols = self.cols
        new_maze.start_pos = self.start_pos
        
        # Deep copy food items so each agent has independent food
        new_maze.food_items = copy.deepcopy(self.food_items)
        
        return new_maze
    
    def is_wall(self, grid_x, grid_y):
        """Check if given grid position is a wall."""
        if grid_y < 1 or grid_y > self.rows - 1:
            return True
        if grid_x < 1 or grid_x > self.cols - 1:
            return True
        return self.layout[grid_y][grid_x] == '1'
    
    def get_food_at(self, grid_x, grid_y):
        """Return food item at position if it exists and hasn't been eaten."""
        for food in self.food_items:
            if (food['grid_x'] == grid_x and 
                food['grid_y'] == grid_y and 
                not food['eaten']):
                return food
        return None
    
    def to_pixel(self, grid_x, grid_y):
        """Convert grid coordinates to pixel coordinates (center of cell)."""
        pixel_x = grid_x * self.cell_size + self.cell_size // 2
        pixel_y = grid_y * self.cell_size + self.cell_size // 2
        return pixel_x, pixel_y
    
    def reset_food(self):
        """Reset all food items to uneaten state."""
        for food in self.food_items:
            food['eaten'] = False


# Default maze layout with balanced food distribution
DEFAULT_MAZE = [
    "11111111111111111111111111",
    "10000000010000010000000001",
    "10111110010000010111111101",
    "10000010010000000000000001",
    "11101010011110011101111111",
    "10001010000000000100000101",
    "11111010111110110101110101",
    "10000010000010010000000001",
    "10111110000010011110111111",
    "10000000000010000000000001",
    "11111101000011110000001101",
    "10000001000000000000000001",
    "10111111000000001111011111",
    "10100000000000000000000001",
    "100000000000S0000000000001",
    "10100000000000000000000001",
    "10111111000000001111011111",
    "10000001000000001000000001",
    "11111101000011111000001101",
    "10000000000010000000000001",
    "10111110000010011110111111",
    "10000010000010010000000001",
    "11100010111110010000011001",
    "10000000000000000000000001",
    "11111111111111111111111111"
]