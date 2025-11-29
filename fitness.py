"""
Hybrid fitness function with curriculum learning for NEAT maze navigation.
FIXED: Prevents agents from exploiting "stay forever" strategy.
"""
import math
from collections import defaultdict

def compute_fitness(agent, maze, generation):
    """
    Hybrid fitness function with curriculum learning.
    
    Args:
        agent: Agent instance with trajectory data
        maze: Maze instance
        generation: Current generation number
    
    Returns:
        float: Fitness score
    """
    # Extract agent data
    trajectory = agent.trajectory
    collisions = agent.collision_steps
    small_food = agent.collected_small
    big_food = agent.collected_big
    total_food = small_food + big_food
    final_energy = agent.energy
    survival_steps = agent.steps
    max_steps = agent.max_steps
    
    # ========== MINIMUM ACTIVITY REQUIREMENT ==========
    unique_positions = len(set(trajectory)) if trajectory else 0
    
    # Generation-based minimum activity requirements
    if generation < 30:
        if unique_positions < 5:
            return 0.1  # Must move in early generations
    elif generation < 100:
        if unique_positions < 3:
            return 0.5
    else:
        if unique_positions < 2:
            return 1.0
    
    # ========== BASE REWARDS ==========
    # Logarithmic survival bonus (diminishing returns)
    survival_bonus = math.log(survival_steps + 1) * 15
    
    base_reward = (
        small_food * 50 +        # INCREASED from 15
        big_food * 200 +         # INCREASED from 60
        survival_bonus +         # CHANGED to logarithmic
        final_energy * 0.8       # DECREASED from 1.5
    )
    
    # ========== PATH EFFICIENCY ==========
    total_distance = len(trajectory)
    efficiency_bonus = 0
    
    if total_distance > 0 and total_food > 0:
        # Calculate optimal Manhattan distance to all collected food
        optimal_distance = 0
        spawn_pos = trajectory[0] if trajectory else (0, 0)
        
        # This is a simplified estimation
        for food in maze.food_items:
            if food['eaten']:
                optimal_distance += abs(food['grid_x'] - spawn_pos[0]) + \
                                   abs(food['grid_y'] - spawn_pos[1])
        
        if optimal_distance > 0:
            # Efficiency ratio (closer to 1.0 is better)
            path_efficiency_ratio = optimal_distance / total_distance
            efficiency_bonus = 80 * (path_efficiency_ratio ** 2)
    
    # ========== EXPLORATION ENTROPY ==========
    region_size = max(maze.cols // 5, 3)
    region_visits = defaultdict(int)
    
    for x, y in trajectory:
        region_x = x // region_size
        region_y = y // region_size
        region_visits[(region_x, region_y)] += 1
    
    # Calculate Shannon entropy
    exploration_bonus = 0
    if trajectory:
        total_steps = len(trajectory)
        visit_entropy = 0
        
        for count in region_visits.values():
            probability = count / total_steps
            if probability > 0:
                visit_entropy -= probability * math.log2(probability)
        
        # Normalize by maximum possible entropy
        num_regions_x = (maze.cols + region_size - 1) // region_size
        num_regions_y = (maze.rows + region_size - 1) // region_size
        total_regions = num_regions_x * num_regions_y
        max_entropy = math.log2(total_regions) if total_regions > 1 else 1
        
        normalized_entropy = visit_entropy / max_entropy if max_entropy > 0 else 0
        coverage_ratio = len(region_visits) / total_regions if total_regions > 0 else 0
        
        exploration_bonus = normalized_entropy * 40 + coverage_ratio * 40
    
    # ========== PENALTIES ==========
    # Collision penalty with exponential decay
    collision_penalty = 0
    for step in collisions:
        decay_factor = math.exp(-step / max_steps) if max_steps > 0 else 1.0
        collision_penalty += 4.0 * decay_factor
    
    # ========== HARSH STAGNATION PENALTY ==========
    if len(trajectory) > 1:
        unique_positions = len(set(trajectory))
        position_diversity = unique_positions / len(trajectory)
        
        # Exponential penalty for low diversity
        if position_diversity < 0.05:
            stagnation_penalty = 300  # SEVERE penalty for staying still
        elif position_diversity < 0.15:
            stagnation_penalty = 150
        elif position_diversity < 0.30:
            stagnation_penalty = 80
        else:
            stagnation_penalty = 30 * (1.0 - position_diversity)
        
        # Additional consecutive stay penalty
        consecutive_stays = 0
        max_consecutive = 0
        for i in range(len(trajectory) - 1):
            if trajectory[i] == trajectory[i + 1]:
                consecutive_stays += 1
                max_consecutive = max(max_consecutive, consecutive_stays)
            else:
                consecutive_stays = 0
        
        if max_consecutive > 10:
            stagnation_penalty += max_consecutive * 3
    else:
        stagnation_penalty = 300  # SEVERE penalty for no movement
    
    # ========== MOVEMENT DIVERSITY BONUS ==========
    if len(trajectory) > 1:
        directions = []
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            if (dx, dy) != (0, 0):
                directions.append((dx, dy))
        
        unique_directions = len(set(directions))
        movement_bonus = unique_directions * 10
    else:
        movement_bonus = 0
    
    # ========== ENERGY EFFICIENCY MULTIPLIER ==========
    efficiency_multiplier = 0.3  # DECREASED from 0.7 for no food
    if total_food > 0 and total_distance > 0:
        energy_per_food = total_distance / total_food
        efficiency_multiplier = 1.0 / (1.0 + energy_per_food / 12)
    
    # ========== CURRICULUM WEIGHTS ==========
    if generation < 20:
        # Very early: Force exploration above all else
        weights = {'base': 0.5, 'efficiency': 0.0, 'exploration': 3.0}
    elif generation < 50:
        # Early: Emphasize food collection
        weights = {'base': 1.2, 'efficiency': 0.3, 'exploration': 1.5}
    elif generation < 120:
        # Mid: Balance objectives
        weights = {'base': 1.8, 'efficiency': 1.0, 'exploration': 0.6}
    else:
        # Late: Optimize task performance
        weights = {'base': 2.0, 'efficiency': 1.5, 'exploration': 0.3}
    
    # ========== FINAL CALCULATION ==========
    fitness = (
        (base_reward * weights['base']) +
        (efficiency_bonus * weights['efficiency']) +
        (exploration_bonus * weights['exploration']) +
        movement_bonus -
        collision_penalty -
        stagnation_penalty
    ) * efficiency_multiplier
    
    # Ensure minimum positive fitness
    return max(fitness, 0.1)