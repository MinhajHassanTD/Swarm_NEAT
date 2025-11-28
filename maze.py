"""
Maze environment with walls and food items.
"""

class Maze:
    def __init__(self, layout, cell_size=40):
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
        """
        self.layout = layout
        self.cell_size = cell_size
        self.rows = len(layout)
        self.cols = len(layout[0]) if layout else 0
        
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
    
    def is_wall(self, grid_x, grid_y):
        """Check if given grid position is a wall."""
        if grid_y < 0 or grid_y >= self.rows:
            return True
        if grid_x < 0 or grid_x >= self.cols:
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
    "1S010001111000000000000001",
    "1001f0010001111110000f0001",
    "1f0000F0001000001111100001",
    "1111000f111100000F00000001",
    "10000000000111100000000001",
    "1000F1111F0100000000000F01",
    "10000000100111111100000001",
    "11100F001000000f0000011101",
    "10000000111110000000100001",
    "1F000000000001000000001001",
    "10000011111000111100000001",
    "10000010000000000000000f01",
    "10000010001111101000000001",
    "10000f00001000001000000001",
    "10000111111000001111100001",
    "10000000000F000000000f0001",
    "11111111100001111111111001",
    "100f000000000000000000F001",
    "11111111111111111111111111"
]