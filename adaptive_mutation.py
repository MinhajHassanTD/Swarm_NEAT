"""
Adaptive mutation system with tiered stagnation detection.
Automatically increases mutation pressure when population stagnates.
"""

def detect_stagnation(best_fitness_history, stagnation_counter):
    """
    Detect stagnation using historical comparison.
    
    Args:
        best_fitness_history: List of best fitness per generation
        stagnation_counter: Dict tracking stagnation state
    
    Returns:
        tuple: (is_stagnant, tier)
            - is_stagnant: bool
            - tier: 0 (none), 1 (mild), 2 (moderate), 3 (severe)
    """
    if len(best_fitness_history) < 30:
        return False, 0
    
    # Compare recent (last 10) vs historical (last 30)
    recent_best = max(best_fitness_history[-10:])
    historical_best = max(best_fitness_history[-30:-10]) if len(best_fitness_history) >= 30 else 0
    
    # Stagnation = recent < historical × 1.02 (< 2% improvement)
    is_stagnant = recent_best < historical_best * 1.02
    
    if is_stagnant:
        stagnation_counter['count'] += 1
    else:
        stagnation_counter['count'] = 0
    
    # Determine tier
    count = stagnation_counter['count']
    if count >= 25:
        tier = 3  # Severe
    elif count >= 10:
        tier = 2  # Moderate
    elif count >= 1:
        tier = 1  # Mild
    else:
        tier = 0  # None
    
    return is_stagnant, tier


def adjust_mutation_rates_v2(config, generation, best_fitness_history, stagnation_counter):
    """
    Adjust mutation rates based on stagnation tier.
    
    Args:
        config: NEAT config object
        generation: Current generation
        best_fitness_history: List of best fitness values
        stagnation_counter: Dict with 'count' and 'diversity_injection_needed'
    
    Returns:
        dict: Updated stagnation_counter
    """
    if generation < 20:
        return stagnation_counter
    
    is_stagnant, tier = detect_stagnation(best_fitness_history, stagnation_counter)
    
    # Tier 1: Mild stagnation (1-10 gens)
    if tier == 1:
        scale_factor = 1.20  # +20% per generation
        print(f"    ⚠️  Stagnation Tier 1 (duration: {stagnation_counter['count']} gens)")
    
    # Tier 2: Moderate stagnation (10-25 gens)
    elif tier == 2:
        scale_factor = 1.35  # +35% per generation
        print(f"    🔶 Stagnation Tier 2 (duration: {stagnation_counter['count']} gens)")
    
    # Tier 3: Severe stagnation (25+ gens)
    elif tier == 3:
        scale_factor = 1.60  # +60% per generation
        print(f"    🔴 Stagnation Tier 3 (duration: {stagnation_counter['count']} gens) - DIVERSITY INJECTION NEEDED")
        stagnation_counter['diversity_injection_needed'] = True
    
    else:
        # Check for progress (exploitation mode)
        if len(best_fitness_history) >= 10 and generation > 100:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
            if recent_improvement > 50:
                # Reduce mutation (exploitation)
                config.genome_config.conn_add_prob = max(0.40, config.genome_config.conn_add_prob * 0.95)
                config.genome_config.node_add_prob = max(0.15, config.genome_config.node_add_prob * 0.95)
                config.genome_config.weight_mutate_power = max(0.70, config.genome_config.weight_mutate_power * 0.95)
                print(f"    ✅ Progress detected (+{recent_improvement:.0f}) - Reducing mutation (exploitation mode)")
        
        return stagnation_counter
    
    # Apply exponential scaling
    old_conn = config.genome_config.conn_add_prob
    
    config.genome_config.conn_add_prob = min(0.95, config.genome_config.conn_add_prob * scale_factor)
    config.genome_config.node_add_prob = min(0.70, config.genome_config.node_add_prob * scale_factor)
    config.genome_config.weight_mutate_power = min(2.5, config.genome_config.weight_mutate_power * scale_factor)
    
    print(f"       Mutation rates increased: conn_add {old_conn:.2f}→{config.genome_config.conn_add_prob:.2f}")
    
    return stagnation_counter


def inject_diversity(population, config, generation):
    """
    Inject new random genomes to combat diversity loss.
    
    Args:
        population: NEAT population object
        config: NEAT config
        generation: Current generation
    
    Returns:
        population: Modified population
    """
    print(f"\n    🧬 DIVERSITY INJECTION at Gen {generation}")
    
    # Get current population
    pop_list = list(population.population.items())
    
    # Keep top 70%
    ranked = sorted(pop_list, key=lambda x: x[1].fitness if x[1].fitness else 0, reverse=True)
    keep_count = int(len(pop_list) * 0.70)
    new_population = dict(ranked[:keep_count])
    
    # Create 30% new random genomes
    next_genome_id = max(population.population.keys()) + 1
    
    for i in range(keep_count, config.pop_size):
        new_genome = config.genome_type(next_genome_id)
        new_genome.configure_new(config.genome_config)
        new_genome.fitness = 0.1
        new_population[next_genome_id] = new_genome
        next_genome_id += 1
    
    population.population = new_population
    population.species.speciate(config, population.population, population.generation)
    
    print(f"    ✅ Injected {config.pop_size - keep_count} new genomes\n")
    
    return population