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
        self.max_steps = max(max_steps, 1)
        
        # Position in grid coordinates
        self.gx, self.gy = maze.start_pos
        
        # Energy management system
        self.energy = 150.0
        self.max_energy = 150.0
        self.energy_per_step = 0.5
        self.energy_per_collision = 5.0
        
        # Performance tracking
        self.steps = 0
        self.collisions = 0
        self.collected_small = 0
        self.collected_big = 0
        self.alive = True
        
        # Trajectory tracking for fitness
        self.trajectory = [(self.gx, self.gy)]
        self.collision_steps = []
        
        # Distance tracking for fitness calculation
        self.initial_distance_to_food = None
        self.min_distance_to_food = float('inf')
        self.last_distance_to_food = float('inf')
        
        # Track visited positions for revisit detection
        self.visited_positions = {(self.gx, self.gy): 1}
        
        # Validate starting position
        if self.maze.is_wall(self.gx, self.gy):
            raise ValueError(f"Start position ({self.gx}, {self.gy}) is a wall!")
    
    def get_distance_to_wall(self, direction):
        """
        Calculate distance to nearest wall in given direction.
        
        Args:
            direction: 0=up, 1=down, 2=left, 3=right
        
        Returns:
            int: Distance in cells to nearest wall
        """
        distance = 0
        x, y = self.gx, self.gy
        
        max_dist = max(self.maze.rows, self.maze.cols, 1)
        
        while distance < max_dist:
            if direction == 0:  # up
                y -= 1
            elif direction == 1:  # down
                y += 1
            elif direction == 2:  # left
                x -= 1
            elif direction == 3:  # right
                x += 1
            
            distance += 1
            
            if self.maze.is_wall(x, y):
                return distance
        
        return distance
    
    def get_unit_vector(self, target_x, target_y):
        """
        Calculate normalized direction vector to target.
        
        Returns:
            tuple: (dx, dy) normalized to unit length
        """
        dx = target_x - self.gx
        dy = target_y - self.gy
        magnitude = math.sqrt(dx**2 + dy**2)
        
        if magnitude < 0.001:
            return (0.0, 0.0)
        
        return (dx / magnitude, dy / magnitude)
    
    def get_manhattan_distance(self, target_x, target_y):
        """Calculate Manhattan distance to target."""
        return abs(target_x - self.gx) + abs(target_y - self.gy)
    
    def get_nearest_food(self):
        """
        Find nearest uneaten food item.
        
        Returns:
            dict or None: Food item with keys 'grid_x', 'grid_y', 'big'
        """
        nearest_food = None
        min_dist = float('inf')
        
        if not hasattr(self.maze, 'food_items') or self.maze.food_items is None:
            return None
        
        for food in self.maze.food_items:
            if not food.get('eaten', False):
                dist = self.get_manhattan_distance(food['grid_x'], food['grid_y'])
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = food
        
        return nearest_food
    
    def get_revisit_indicator(self):
        """
        Calculate how many times current position has been visited.
        Returns normalized value (0.0 = new, 1.0 = heavily revisited).
        """
        visit_count = self.visited_positions.get((self.gx, self.gy), 0)
        # Normalize: 0 visits = 0.0, 10+ visits = 1.0
        return min(visit_count / 10.0, 1.0)
    
    def get_inputs(self):
        """
        Build 12-dimensional input vector for the neural network.
        
        Returns:
            list: 12 normalized inputs:
                0-3: Distance to walls (up, down, left, right) normalized
                4-5: RAW directional distance to food (dx, dy) - CHANGED
                6: Manhattan distance to nearest food (normalized)
                7: Food size (1.0 if big, 0.0 if small)
                8: Energy critical flag (1.0 if < 30%)
                9: Energy healthy flag (1.0 if > 60%)
                10: Revisit indicator (0.0 = new, 1.0 = been here a lot)
                11: Bias (1.0)
        """
        # Calculate distances to walls (normalized)
        max_dimension = max(self.maze.rows, self.maze.cols, 1)
        
        distance_up = min(self.get_distance_to_wall(0) / max_dimension, 1.0)
        distance_down = min(self.get_distance_to_wall(1) / max_dimension, 1.0)
        distance_left = min(self.get_distance_to_wall(2) / max_dimension, 1.0)
        distance_right = min(self.get_distance_to_wall(3) / max_dimension, 1.0)
        
        # Find nearest food
        nearest_food = self.get_nearest_food()
        
        if nearest_food:
            # Raw directional distance (CHANGED)
            dx = nearest_food['grid_x'] - self.gx
            dy = nearest_food['grid_y'] - self.gy
            
            # Normalize by maze size
            max_distance = max(self.maze.cols + self.maze.rows, 1)
            food_dx_norm = dx / max_distance  # Can be negative!
            food_dy_norm = dy / max_distance  # Can be negative!
            
            # Manhattan distance to food (normalized)
            food_distance = abs(dx) + abs(dy)
            food_distance_norm = min(food_distance / max_distance, 1.0)
            
            # Food size
            food_size = 1.0 if nearest_food.get('big', False) else 0.0
        else:
            food_dx_norm = 0.0
            food_dy_norm = 0.0
            food_distance_norm = 1.0
            food_size = 0.0
        
        # Energy state flags
        energy_ratio = self.energy / max(self.max_energy, 1.0)
        energy_critical = 1.0 if energy_ratio < 0.25 else 0.0
        energy_healthy = 1.0 if energy_ratio > 0.75 else 0.0
        
        # Revisit indicator
        revisit_indicator = self.get_revisit_indicator()
        
        return [
            distance_up,
            distance_down,
            distance_left,
            distance_right,
            food_dx_norm,      # CHANGED: Raw dx (can be -1.0 to 1.0)
            food_dy_norm,      # CHANGED: Raw dy (can be -1.0 to 1.0)
            food_distance_norm,
            food_size,
            energy_critical,
            energy_healthy,
            revisit_indicator,
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
        
        # Deduct movement energy
        self.energy -= self.energy_per_step
        
        # Calculate new position
        new_gx, new_gy = self.gx, self.gy
        
        if direction_index == 0:  # up
            new_gy -= 1
        elif direction_index == 1:  # down
            new_gy += 1
        elif direction_index == 2:  # left
            new_gx -= 1
        elif direction_index == 3:  # right
            new_gx += 1
        
        # Handle wall collision
        if self.maze.is_wall(new_gx, new_gy):
            self.collisions += 1
            self.collision_steps.append(self.steps)
            self.energy -= self.energy_per_collision
            
            # Record failed attempt in trajectory
            self.trajectory.append((self.gx, self.gy))
            self.visited_positions[(self.gx, self.gy)] = self.visited_positions.get((self.gx, self.gy), 0) + 1
            
            if self.energy <= 0:
                self.alive = False
            return
        
        # Update position
        self.gx, self.gy = new_gx, new_gy
        self.trajectory.append((self.gx, self.gy))
        
        # Track visit to new position
        self.visited_positions[(self.gx, self.gy)] = self.visited_positions.get((self.gx, self.gy), 0) + 1
        
        # Check for food collection
        food = self.maze.get_food_at(self.gx, self.gy)
        if food:
            food['eaten'] = True
            if food.get('big', False):
                self.collected_big += 1
                self.energy = min(self.max_energy, self.energy + 80.0)
            else:
                self.collected_small += 1
                self.energy = min(self.max_energy, self.energy + 40.0)
    
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
        max_possible = max(self.initial_distance_to_food, 1)
        
        return max(0.0, improvement / max_possible)