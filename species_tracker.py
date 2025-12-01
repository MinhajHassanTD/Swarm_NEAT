"""
Species tracking, visualization, and fitness diagnostics.
Combines behavioral classification, visual properties, and component analysis.
"""
import colorsys

# ========== BEHAVIORAL CLASSIFICATION ==========

def classify_species_archetype(species_data):
    """
    Classify species into behavioral archetypes.
    
    Args:
        species_data: Dict with avg metrics
    
    Returns:
        str: Archetype name
    """
    food = species_data.get('avg_food_collected', 0)
    collisions = species_data.get('avg_collisions', 0)
    exploration = species_data.get('avg_exploration_pct', 0)
    position_diversity = species_data.get('avg_position_diversity', 0.5)
    
    # Classification logic
    if position_diversity < 0.15:
        return 'stuck'
    elif collisions > 20 and food > 5:
        return 'food_rusher'
    elif collisions < 5 and exploration < 0.3:
        return 'wall_hugger'
    elif exploration > 0.6 and food < 4:
        return 'explorer'
    else:
        return 'generalist'


def get_species_visual_properties(species_id, species_data, num_species, generation):
    """
    Generate visual properties for species rendering.
    
    Args:
        species_id: Species ID
        species_data: Dict with metrics
        num_species: Total number of species
        generation: Current generation
    
    Returns:
        dict: Visual properties (color, shape, size, effects)
    """
    archetype = classify_species_archetype(species_data)
    
    # Color: HSV wheel based on species ID
    hue = (species_id * 137.508) % 360  # Golden angle distribution
    
    # Saturation/Value based on performance
    fitness_ratio = species_data['avg_fitness'] / max(species_data['max_fitness_all_species'], 1.0)
    saturation = 0.5 + (0.5 * fitness_ratio)  # 0.5-1.0
    value = 0.6 + (0.4 * fitness_ratio)  # 0.6-1.0
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    color = tuple(int(c * 255) for c in rgb)
    
    # Shape mapping
    shape_map = {
        'explorer': 'circle',
        'wall_hugger': 'square',
        'food_rusher': 'diamond',
        'stuck': 'triangle',
        'generalist': 'octagon'
    }
    shape = shape_map.get(archetype, 'circle')
    
    # Size based on population ratio
    pop_ratio = species_data['member_count'] / max(species_data['total_population'], 1)
    size_multiplier = 0.7 + (0.8 * pop_ratio)
    
    # Effects
    effects = []
    if species_data.get('is_champion', False):
        effects.append({'type': 'glow', 'color': (255, 255, 255), 'width': 2})
    
    age = species_data.get('age', 0)
    if age < 5:
        effects.append({'type': 'pulse', 'frequency': 2.0})
    
    stagnation = species_data.get('stagnation', 0)
    if stagnation > 10:
        effects.append({'type': 'fade', 'alpha': 0.4})
    
    return {
        'color': color,
        'shape': shape,
        'size_multiplier': size_multiplier,
        'effects': effects,
        'archetype': archetype
    }


# ========== FITNESS COMPONENT TRACKING ==========

class FitnessComponentTracker:
    """Track fitness component balance across generations."""
    
    def __init__(self):
        self.history = []
    
    def log_generation(self, generation, genomes):
        """
        Record fitness components for all genomes.
        
        Args:
            generation: Current generation number
            genomes: List of (genome_id, genome) tuples OR dict of genomes
        """
        # ⭐ FIX: Handle both list and dict inputs
        if isinstance(genomes, list):
            # genomes is list of tuples [(genome_id, genome), ...]
            genome_list = [g for _, g in genomes]
        elif isinstance(genomes, dict):
            # genomes is dict {genome_id: genome, ...}
            genome_list = list(genomes.values())
        else:
            # Unknown type, skip
            return
        
        # Filter genomes with fitness_components
        genomes_with_components = [
            g for g in genome_list
            if hasattr(g, 'fitness_components') and g.fitness_components
        ]
        
        if not genomes_with_components:
            return
        
        # Analyze top 10%
        sorted_genomes = sorted(genomes_with_components, key=lambda g: g.fitness or 0, reverse=True)
        elite_count = max(1, len(sorted_genomes) // 10)
        elite_genomes = sorted_genomes[:elite_count]
        
        # Average components
        components = {
            'food': 0, 'survival': 0, 'exploration': 0,
            'movement': 0, 'proximity': 0,
            'collision_penalty': 0, 'stagnation_penalty': 0
        }
        
        for g in elite_genomes:
            comp = g.fitness_components
            components['food'] += comp.get('food_score', 0)
            components['survival'] += comp.get('survival_score', 0)
            components['exploration'] += comp.get('exploration_score', 0)
            components['movement'] += comp.get('movement_bonus', 0)
            components['proximity'] += comp.get('proximity_bonus', 0)
            components['collision_penalty'] += comp.get('collision_penalty', 0)
            components['stagnation_penalty'] += comp.get('stagnation_penalty', 0)
        
        # Average
        for key in components:
            components[key] /= len(elite_genomes)
        
        self.history.append({
            'generation': generation,
            'components': components
        })
    
    def diagnose_fitness_imbalance(self, generation):
        """
        Detect if any component dominates fitness.
        
        Returns:
            dict: Diagnosis with 'dominant_component', 'contribution', 'imbalanced', 'recommendation'
        """
        if not self.history:
            return {'imbalanced': False}
        
        latest = self.history[-1]['components']
        
        # Calculate positive contributions
        total_positive = (
            latest['food'] +
            latest['survival'] +
            latest['exploration'] +
            latest['movement'] +
            latest['proximity']
        )
        
        if total_positive == 0:
            return {'imbalanced': False}
        
        # Find dominant component
        contributions = {
            'food': latest['food'] / total_positive,
            'survival': latest['survival'] / total_positive,
            'exploration': latest['exploration'] / total_positive,
            'movement': latest['movement'] / total_positive,
            'proximity': latest['proximity'] / total_positive
        }
        
        dominant = max(contributions, key=contributions.get)
        dominant_pct = contributions[dominant] * 100
        
        # Check imbalance (>60%)
        imbalanced = dominant_pct > 60
        
        # Recommendations
        recommendations = {
            'food': "Reduce food weight slightly, increase exploration/efficiency rewards",
            'survival': "Population too cautious, reduce survival weight, increase food weight",
            'exploration': "Wandering without purpose, reduce exploration weight significantly",
            'movement': "Movement diversity over-rewarded, reduce movement weight",
            'proximity': "Proximity bonus too high (shouldn't dominate), reduce weight"
        }
        
        return {
            'dominant_component': dominant,
            'contribution': f"{dominant_pct:.1f}%",
            'imbalanced': imbalanced,
            'recommendation': recommendations.get(dominant, "Balance looks good")
        }