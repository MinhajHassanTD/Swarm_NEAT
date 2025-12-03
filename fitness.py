"""
Fitness Function v2: Simple, stable curriculum learning
"""
import math

def compute_fitness(agent, maze, generation, population_stats=None):
    """
    Calculate fitness with adaptive curriculum weights.
    
    Args:
        agent: Agent with trajectory, energy, food data
        maze: Maze environment
        generation: Current generation number
        population_stats: Optional dict with 'avg_food', 'max_food', 'avg_survival'
    
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
    
    # Get curriculum weights (adaptive if stats provided, otherwise fixed)
    if population_stats:
        weights = get_curriculum_weights(generation, population_stats)
    else:
        # Fallback to fixed curriculum
        if generation < 80:
            weights = {'food': 2.0, 'survival': 0.5, 'explore': 1.0, 'movement': 0.5}
        elif generation < 200:
            weights = {'food': 3.0, 'survival': 0.3, 'explore': 0.5, 'movement': 0.3}
        else:
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


def get_curriculum_weights(generation, population_stats):
    """
    Dynamic curriculum based on population performance.
    
    Args:
        generation: Current generation number
        population_stats: Dict with 'avg_food', 'max_food', 'avg_survival'
    """
    avg_food = population_stats.get('avg_food', 0)
    max_food = population_stats.get('max_food', 0)
    avg_survival = population_stats.get('avg_survival', 0)
    
    # Phase 1: Until population averages 2+ food
    if avg_food < 2.0:
        return {'food': 2.0, 'survival': 0.5, 'explore': 1.0, 'movement': 0.5}
    
    # Phase 2: Until max agent gets 6+ food
    elif max_food < 6.0:
        return {'food': 3.0, 'survival': 0.3, 'explore': 0.5, 'movement': 0.3}
    
    # Phase 3: Until population averages 5+ food
    elif avg_food < 5.0:
        return {'food': 3.5, 'survival': 0.25, 'explore': 0.4, 'movement': 0.25}
    
    # Phase 4: Mastery (8+ food average)
    else:
        return {'food': 4.0, 'survival': 0.2, 'explore': 0.2, 'movement': 0.2}