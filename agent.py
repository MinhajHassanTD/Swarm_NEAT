"""
Agent controlled by NEAT neural network navigating a maze.
This version uses minimal input features to allow NEAT to learn naturally.
"""
import math

class Agent:
    def __init__(self, maze, net, genome_id=None, max_steps=300):
        """
        Initialize an agent in the maze.
        
        Args:
            maze: Maze instance
            net: NEAT neural network
            genome_id: Optional ID for debugging
            max_steps: Maximum steps for normalization
        """
        self.maze = maze
        self.net = net
        self.genome_id = genome_id
        self.max_steps = max_steps
        
        # Position in grid coordinates
        self.gx, self.gy = maze.start_pos
        
        # Energy management system
        self.energy = 100.0
        self.max_energy = 100.0
        self.energy_per_step = 0.5
        self.energy_per_collision = 7.0
        
        # Performance tracking
        self.steps = 0
        self.collisions = 0
        self.collected_small = 0
        self.collected_big = 0
        self.alive = True
        
        # Distance tracking for fitness calculation
        self.initial_distance_to_food = None
        self.min_distance_to_food = float('inf')
        self.last_distance_to_food = float('inf')
        
        # Validate starting position
        if self.maze.is_wall(self.gx, self.gy):
            raise ValueError(f"Start position ({self.gx}, {self.gy}) is a wall!")
    
    def get_inputs(self):
        """
        Build input vector for the neural network.
        
        Returns 9 normalized inputs:
        - Wall sensors (4): Binary flags for walls in each direction
        - Food direction (2): Normalized dx, dy to nearest food
        - Food type (1): Binary flag indicating if nearest food is big
        - Energy level (1): Current energy as percentage of maximum
        - Bias (1): Constant input of 1.0
        """
        # Check for walls in each direction
        wall_up = 1.0 if self.maze.is_wall(self.gx, self.gy - 1) else 0.0
        wall_down = 1.0 if self.maze.is_wall(self.gx, self.gy + 1) else 0.0
        wall_left = 1.0 if self.maze.is_wall(self.gx - 1, self.gy) else 0.0
        wall_right = 1.0 if self.maze.is_wall(self.gx + 1, self.gy) else 0.0
        
        # Find nearest uneaten food
        nearest_food = None
        min_dist = float('inf')
        
        for food in self.maze.food_items:
            if not food['eaten']:
                dx = food['grid_x'] - self.gx
                dy = food['grid_y'] - self.gy
                dist = abs(dx) + abs(dy)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = (dx, dy, food['big'], dist)
        
        # Update distance tracking for fitness calculation
        if nearest_food:
            _, _, _, dist = nearest_food
            if self.initial_distance_to_food is None:
                self.initial_distance_to_food = dist
            self.min_distance_to_food = min(self.min_distance_to_food, dist)
            self.last_distance_to_food = dist
        
        # Normalize food direction using tanh for bounded output
        if nearest_food:
            dx, dy, is_big, dist = nearest_food
            max_dist = max(self.maze.cols, self.maze.rows, 10)
            
            # Scale by much higher value to make food direction overwhelmingly clear
            # This helps prevent oscillation near walls by making one direction clearly best
            dx_norm = math.tanh(dx * 15.0 / max_dist)
            dy_norm = math.tanh(dy * 15.0 / max_dist)
            is_big_norm = 1.0 if is_big else 0.0
        else:
            # No food available
            dx_norm = 0.0
            dy_norm = 0.0
            is_big_norm = 0.0
        
        # Normalize energy to [0, 1] range
        energy_norm = max(0.0, min(1.0, self.energy / self.max_energy))
        
        return [
            wall_up,
            wall_down,
            wall_left,
            wall_right,
            dx_norm,
            dy_norm,
            is_big_norm,
            energy_norm,
            1.0  # bias
        ]
    
    def step(self, direction_index):
        """
        Execute one movement step based on network output.
        
        Args:
            direction_index: 0=up, 1=down, 2=left, 3=right
        """
        if not self.alive:
            return
        
        # Check for energy depletion
        if self.energy <= 0:
            self.alive = False
            return
            
        self.steps += 1
        self.energy -= self.energy_per_step
        
        # Calculate new position
        new_gx, new_gy = self.gx, self.gy
        
        if direction_index == 0:
            new_gy -= 1
        elif direction_index == 1:
            new_gy += 1
        elif direction_index == 2:
            new_gx -= 1
        elif direction_index == 3:
            new_gx += 1
        
        # Handle wall collision
        if self.maze.is_wall(new_gx, new_gy):
            self.collisions += 1
            self.energy -= self.energy_per_collision
            if self.energy <= 0:
                self.alive = False
            return
        
        # Update position
        self.gx, self.gy = new_gx, new_gy
        
        # Check for food collection
        food = self.maze.get_food_at(self.gx, self.gy)
        if food:
            food['eaten'] = True
            if food['big']:
                self.collected_big += 1
                self.energy = min(self.max_energy, self.energy + 60.0)
            else:
                self.collected_small += 1
                self.energy = min(self.max_energy, self.energy + 30.0)
    
    def is_starving(self):
        """Check if agent has critically low energy."""
        return self.energy < 20.0
    
    def get_survival_time(self):
        """Return number of steps agent survived."""
        return self.steps
    
    def get_exploration_score(self):
        """
        Calculate how much progress was made toward food.
        Returns value between 0 and 1.
        """
        if self.initial_distance_to_food is None or self.initial_distance_to_food == 0:
            return 0.0
        
        improvement = self.initial_distance_to_food - self.min_distance_to_food
        max_possible = self.initial_distance_to_food
        
        return max(0.0, improvement / max_possible)