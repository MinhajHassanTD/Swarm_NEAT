"""
Fitness Function v2: Simple, stable curriculum learning
"""
import math

def compute_fitness(agent, maze, generation):
    """
    Calculate fitness with simple curriculum weights.
    
    Args:
        agent: Agent with trajectory, energy, food data
        maze: Maze environment
        generation: Current generation number
    
    Returns:
        float: Fitness score (always >= 0.1)
    """
    # Safety checks
    if agent is None or not hasattr(agent, 'trajectory'):
        return 0.1
    
    trajectory = agent.trajectory
    if not trajectory or len(trajectory) == 0:
        return 0.1
    
    # Extract agent metrics
    small_food = getattr(agent, 'collected_small', 0)
    big_food = getattr(agent, 'collected_big', 0)
    survival_steps = getattr(agent, 'steps', 0)
    collisions = len(getattr(agent, 'collision_steps', []))
    unique_positions = len(set(trajectory))
    
    # Component 1: Food (most important)
    food_score = (small_food * 50) + (big_food * 200)
    
    # Component 2: Survival
    survival_score = survival_steps * 0.5
    
    # Component 3: Exploration
    exploration_score = unique_positions * 2.0
    
    # Component 4: Movement diversity
    movement_bonus = 0
    if len(trajectory) > 1:
        directions = set()
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            if (dx, dy) != (0, 0):
                directions.add((dx, dy))
        movement_bonus = len(directions) * 10
    
    # Penalties
    collision_penalty = collisions * 5
    
    stagnation_penalty = 0
    if len(trajectory) > 10:
        position_diversity = unique_positions / len(trajectory)
        if position_diversity < 0.05:
            stagnation_penalty = 30
    
    # Simple curriculum weights (3 phases)
    if generation < 80:
        # Phase 1: Learn to explore and find food
        weights = {'food': 2.0, 'survival': 0.5, 'explore': 1.0, 'movement': 0.5}
    elif generation < 200:
        # Phase 2: Optimize food collection
        weights = {'food': 3.0, 'survival': 0.3, 'explore': 0.5, 'movement': 0.3}
    else:
        # Phase 3: Master efficiency
        weights = {'food': 4.0, 'survival': 0.2, 'explore': 0.2, 'movement': 0.2}
    
    # Calculate final fitness
    fitness = (
        (food_score * weights['food']) +
        (survival_score * weights['survival']) +
        (exploration_score * weights['explore']) +
        (movement_bonus * weights['movement']) -
        collision_penalty -
        stagnation_penalty
    )
    
    # Ensure positive
    return max(fitness, 0.1)