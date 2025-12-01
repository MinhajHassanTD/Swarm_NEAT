"""
Normalized fitness function with adaptive curriculum.
"""
import math

def compute_fitness(agent, maze, generation, total_generations=500):
    """
    Calculate fitness score normalized for dynamic food availability.
    
    Returns:
        float: Fitness score (minimum 0.1)
    """
    if not agent or not hasattr(agent, 'trajectory') or not agent.trajectory:
        return 0.1
    
    # Extract metrics
    small_collected = agent.collected_small
    big_collected = agent.collected_big
    steps = agent.steps
    collisions = len(agent.collision_steps)
    unique_cells = len(set(agent.trajectory))
    
    # Count available food
    small_available = sum(1 for f in maze.food_items if not f['big'])
    big_available = sum(1 for f in maze.food_items if f['big'])
    total_available = max(small_available + big_available, 1)
    
    # Collection rates (0.0 to 1.0)
    small_rate = small_collected / max(small_available, 1)
    big_rate = big_collected / max(big_available, 1)
    total_rate = (small_collected + big_collected) / total_available
    
    # ===== FOOD SCORE (normalized & scaled) =====
    food_score = (small_available * 50 * small_rate) + (big_available * 200 * big_rate)
    
    # Bonus for high collection rate (smooth linear)
    if total_rate >= 0.5:
        food_score *= (1.0 + (total_rate - 0.5))  # 1.0x → 1.5x
    
    # Scale to reference (16 food baseline)
    food_score *= (16 / total_available)
    
    # ===== OTHER COMPONENTS =====
    survival_score = (steps / agent.max_steps) * 100
    exploration_score = (unique_cells / (maze.rows * maze.cols)) * 100
    
    # Movement diversity
    directions = set()
    for i in range(len(agent.trajectory) - 1):
        dx = agent.trajectory[i+1][0] - agent.trajectory[i][0]
        dy = agent.trajectory[i+1][1] - agent.trajectory[i][1]
        if (dx, dy) != (0, 0):
            directions.add((dx, dy))
    movement_bonus = (len(directions) / 4.0) * 40
    
    # ===== PENALTIES (fixed, not weighted) =====
    collision_penalty = collisions * 15.0
    
    stagnation_penalty = 0
    if len(agent.trajectory) > 10:
        if (unique_cells / len(agent.trajectory)) < 0.05:
            stagnation_penalty = 50
    
    # ===== ADAPTIVE CURRICULUM WEIGHTS =====
    progress = generation / max(total_generations, 1)
    
    if progress < 0.33:
        w = {'food': 2.0, 'survival': 0.5, 'explore': 1.0, 'movement': 0.5}
    elif progress < 0.67:
        w = {'food': 3.0, 'survival': 0.3, 'explore': 0.5, 'movement': 0.3}
    else:
        w = {'food': 4.0, 'survival': 0.2, 'explore': 0.2, 'movement': 0.2}
    
    # ===== FINAL SCORE =====
    fitness = (
        food_score * w['food'] +
        survival_score * w['survival'] +
        exploration_score * w['explore'] +
        movement_bonus * w['movement']
    ) - collision_penalty - stagnation_penalty
    
    return max(fitness, 0.1)