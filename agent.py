"""
Agent with neural network control and adaptive energy.
"""
import math

class Agent:
    def __init__(self, maze, net, genome_id=None, max_steps=400):
        self.maze = maze
        self.net = net
        self.genome_id = genome_id
        self.max_steps = max_steps
        
        # Position
        self.gx, self.gy = maze.start_pos
        
        # Adaptive energy based on food scarcity
        total_food = sum(1 for f in maze.food_items if not f['eaten'])
        if total_food <= 6:
            self.max_energy = 100.0 + ((6 - total_food) * 10.0)  # Up to 160
        else:
            self.max_energy = 100.0
        self.energy = self.max_energy
        
        # Energy costs
        self.energy_per_step = 0.5
        self.energy_per_collision = 5.0
        
        # Tracking
        self.steps = 0
        self.collisions = 0
        self.collected_small = 0
        self.collected_big = 0
        self.alive = True
        self.trajectory = [(self.gx, self.gy)]
        self.collision_steps = []
        self.visited_positions = {(self.gx, self.gy): 1}
    
    def get_distance_to_wall(self, direction):
        """Distance to wall in given direction (0=up, 1=down, 2=left, 3=right)."""
        x, y = self.gx, self.gy
        distance = 0
        max_dist = max(self.maze.rows, self.maze.cols)
        
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        
        while distance < max_dist:
            x += dx
            y += dy
            distance += 1
            if self.maze.is_wall(x, y):
                return distance
        
        return distance
    
    def get_nearest_food(self):
        """Find nearest uneaten food."""
        nearest = None
        min_dist = float('inf')
        
        for food in self.maze.food_items:
            if not food['eaten']:
                dist = abs(food['grid_x'] - self.gx) + abs(food['grid_y'] - self.gy)
                if dist < min_dist:
                    min_dist = dist
                    nearest = food
        
        return nearest
    
    def get_inputs(self):
        """
        Build 13D input vector:
        [0-3] Wall distances (up, down, left, right)
        [4-5] Food direction (unit vector)
        [6]   Food proximity (inverse distance)
        [7]   Food size (1.0=big, 0.0=small)
        [8]   Energy critical (<30%)
        [9]   Energy healthy (>60%)
        [10]  Revisit indicator
        [11]  Food scarcity
        [12]  Bias (1.0)
        """
        max_dim = max(self.maze.rows, self.maze.cols, 1)
        
        # Wall distances
        walls = [min(self.get_distance_to_wall(i) / max_dim, 1.0) for i in range(4)]
        
        # Food info
        food = self.get_nearest_food()
        if food:
            dx = food['grid_x'] - self.gx
            dy = food['grid_y'] - self.gy
            dist = abs(dx) + abs(dy)
            mag = math.sqrt(dx*dx + dy*dy)
            
            food_dir = (dx/mag, dy/mag) if mag > 0.001 else (0.0, 0.0)
            food_prox = 1.0 / (1.0 + dist * 0.1)  # Inverse distance
            food_size = 1.0 if food['big'] else 0.0
        else:
            food_dir = (0.0, 0.0)
            food_prox = 0.0
            food_size = 0.0
        
        # Energy state
        energy_ratio = self.energy / self.max_energy
        energy_critical = 1.0 if energy_ratio < 0.3 else 0.0
        energy_healthy = 1.0 if energy_ratio > 0.6 else 0.0
        
        # Revisit indicator
        visit_count = self.visited_positions.get((self.gx, self.gy), 0)
        revisit = min(visit_count / 5.0, 1.0)
        
        # Food scarcity
        remaining = sum(1 for f in self.maze.food_items if not f['eaten'])
        total = len(self.maze.food_items)
        scarcity = 1.0 - (remaining / max(total, 1))
        
        return walls + list(food_dir) + [food_prox, food_size, energy_critical, 
                                          energy_healthy, revisit, scarcity, 1.0]
    
    def step(self, direction_index):
        """Execute one step (0-3=move, 4=stay)."""
        if not self.alive or self.energy <= 0:
            self.alive = False
            return
        
        self.steps += 1
        
        # Stay action
        if direction_index == 4:
            self.trajectory.append((self.gx, self.gy))
            self.visited_positions[(self.gx, self.gy)] = \
                self.visited_positions.get((self.gx, self.gy), 0) + 1
            return
        
        # Deduct movement energy
        self.energy -= self.energy_per_step
        
        # Calculate new position
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = moves[direction_index]
        new_x, new_y = self.gx + dx, self.gy + dy
        
        # Wall collision
        if self.maze.is_wall(new_x, new_y):
            self.collisions += 1
            self.collision_steps.append(self.steps)
            self.energy -= self.energy_per_collision
            
            if self.energy <= 0:
                self.alive = False
            
            self.trajectory.append((self.gx, self.gy))
            self.visited_positions[(self.gx, self.gy)] = \
                self.visited_positions.get((self.gx, self.gy), 0) + 1
            return
        
        # Move to new position
        self.gx, self.gy = new_x, new_y
        self.trajectory.append((self.gx, self.gy))
        self.visited_positions[(self.gx, self.gy)] = \
            self.visited_positions.get((self.gx, self.gy), 0) + 1
        
        # Check for food
        food = self.maze.get_food_at(self.gx, self.gy)
        if food:
            food['eaten'] = True
            if food['big']:
                self.collected_big += 1
                self.energy = min(self.max_energy, self.energy + 70.0)
            else:
                self.collected_small += 1
                self.energy = min(self.max_energy, self.energy + 35.0)
    
    def is_starving(self):
        return self.energy < 20.0