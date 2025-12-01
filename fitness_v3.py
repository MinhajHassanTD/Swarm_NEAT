"""
Fitness Function v3: Smooth curriculum with sigmoid transitions
Eliminates fitness cliffs at generation boundaries.
"""
import math

def sigmoid_transition(generation, midpoint, rate):
    """
    Smooth transition using sigmoid function.
    
    Args:
        generation: Current generation number
        midpoint: Generation where transition is 50% complete
        rate: Steepness of transition (higher = faster)
    
    Returns:
        float: Progress from 0.0 to 1.0
    """
    return 1.0 / (1.0 + math.exp(-rate * (generation - midpoint)))


def get_curriculum_weights(generation):
    """
    Calculate smooth curriculum weights based on generation.
    
    Phase transitions:
    - Gen 0-40: Exploration → Food focus
    - Gen 40-100: Food optimization
    - Gen 100+: Efficiency mastery
    
    Returns:
        dict: Weight multipliers for each fitness component
    """
    # Transition 1: Gen 20-40 (exploration → food)
    transition1 = sigmoid_transition(generation, midpoint=30, rate=0.2)
    
    # Transition 2: Gen 70-100 (food → efficiency)
    transition2 = sigmoid_transition(generation, midpoint=85, rate=0.15)
    
    # Smooth weight interpolation
    # Exploration: 2.0 → 1.2 → 0.6 → 0.3
    explore_weight = 2.0 - (0.8 * transition1) - (0.6 * transition2) - (0.3 * min(generation / 200, 1.0))
    
    # Food: 0.5 → 1.5 → 2.0 → 2.5
    food_weight = 0.5 + (1.0 * transition1) + (0.5 * transition2) + (0.5 * min(generation / 200, 1.0))
    
    # Survival: 1.0 → 0.8 → 0.5 → 0.3
    survival_weight = 1.0 - (0.2 * transition1) - (0.3 * transition2) - (0.2 * min(generation / 200, 1.0))
    
    # Movement: 0.3 → 0.5 → 0.4 → 0.3
    movement_weight = 0.3 + (0.2 * transition1) - (0.1 * transition2)
    
    # Proximity (early help): 0.5 → 0.3 → 0.0
    proximity_weight = max(0.0, 0.5 - (0.2 * transition1) - (0.3 * transition2))
    
    return {
        'food': max(0.1, food_weight),
        'survival': max(0.1, survival_weight),
        'explore': max(0.1, explore_weight),
        'movement': max(0.1, movement_weight),
        'proximity': max(0.0, proximity_weight)
    }


def compute_fitness_v3(agent, maze, generation, genome=None):
    """
    Advanced fitness function with smooth curriculum and component tracking.
    
    Args:
        agent: Agent instance
        maze: Maze environment
        generation: Current generation number
        genome: NEAT genome (optional, for storing components)
    
    Returns:
        float: Total fitness score
    """
    # Safety checks
    if agent is None or not hasattr(agent, 'trajectory'):
        return 0.1
    
    trajectory = agent.trajectory
    if not trajectory or len(trajectory) == 0:
        return 0.1
    
    # Extract metrics
    small_food = getattr(agent, 'collected_small', 0)
    big_food = getattr(agent, 'collected_big', 0)
    survival_steps = getattr(agent, 'steps', 0)
    collisions = len(getattr(agent, 'collision_steps', []))
    unique_positions = len(set(trajectory))
    
    # ========== COMPONENT 1: FOOD COLLECTION ==========
    food_score = (small_food * 50) + (big_food * 200)
    
    # ========== COMPONENT 2: SURVIVAL ==========
    survival_score = survival_steps * 0.5
    
    # ========== COMPONENT 3: EXPLORATION ==========
    exploration_score = unique_positions * 2.0
    
    # ========== COMPONENT 4: MOVEMENT DIVERSITY ==========
    movement_bonus = 0
    if len(trajectory) > 1:
        directions = set()
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            if (dx, dy) != (0, 0):
                directions.add((dx, dy))
        movement_bonus = len(directions) * 10
    
    # ========== COMPONENT 5: PROXIMITY BONUS (early gens) ==========
    proximity_bonus = 0
    if generation < 50 and (small_food + big_food) == 0:
        # Help early agents find food
        if hasattr(maze, 'food_items') and maze.food_items:
            uneaten_food = [f for f in maze.food_items if not f.get('eaten', False)]
            if uneaten_food and trajectory:
                min_distance = float('inf')
                for pos in trajectory:
                    for food in uneaten_food:
                        dist = abs(pos[0] - food['grid_x']) + abs(pos[1] - food['grid_y'])
                        min_distance = min(min_distance, dist)
                
                if min_distance < float('inf'):
                    # 50 points at distance 0, 0 points at distance 20+
                    proximity_bonus = max(0, 50 - (min_distance * 2.5))
    
    # ========== PENALTIES ==========
    collision_penalty = collisions * 5
    
    stagnation_penalty = 0
    if len(trajectory) > 10:
        position_diversity = unique_positions / len(trajectory)
        if position_diversity < 0.02:
            stagnation_penalty = 50
        elif position_diversity < 0.05:
            stagnation_penalty = 20
    
    # ========== GET CURRICULUM WEIGHTS ==========
    weights = get_curriculum_weights(generation)
    
    # ========== CALCULATE WEIGHTED COMPONENTS ==========
    food_weighted = food_score * weights['food']
    survival_weighted = survival_score * weights['survival']
    exploration_weighted = exploration_score * weights['explore']
    movement_weighted = movement_bonus * weights['movement']
    proximity_weighted = proximity_bonus * weights['proximity']
    
    # ========== STORE COMPONENTS IN GENOME ==========
    if genome is not None:
        genome.fitness_components = {
            'food_score': food_weighted,
            'survival_score': survival_weighted,
            'exploration_score': exploration_weighted,
            'movement_bonus': movement_weighted,
            'proximity_bonus': proximity_weighted,
            'collision_penalty': collision_penalty,
            'stagnation_penalty': stagnation_penalty,
            'weights': weights  # Store for diagnostics
        }
    
    # ========== FINAL FITNESS ==========
    fitness = (
        food_weighted +
        survival_weighted +
        exploration_weighted +
        movement_weighted +
        proximity_weighted -
        collision_penalty -
        stagnation_penalty
    )
    
    return max(fitness, 0.1)