"""
NEAT evaluation with advanced fitness and tracking.
"""
import pygame
import neat
import time
import pickle
import sys
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_all_agents, draw_hud
from fitness_v3 import compute_fitness_v3  # ⭐ USE V3
from adaptive_mutation import adjust_mutation_rates_v2, inject_diversity
from species_tracker import FitnessComponentTracker

# Simulation parameters
MAX_STEPS = 1000
FPS = 60

# Global tracking
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None

# ⭐ NEW: Stagnation and diagnostics
stagnation_counter = {'count': 0, 'diversity_injection_needed': False}
best_fitness_history = []
fitness_tracker = FitnessComponentTracker()

def eval_genomes(genomes, config):
    """Evaluate all genomes with advanced fitness."""
    global generation_counter, global_best_fitness, global_best_genome
    global stagnation_counter, best_fitness_history, fitness_tracker
    
    gen_start_time = time.time()
    
    # Create master maze
    master_maze = Maze(DEFAULT_MAZE, cell_size=20)
    
    # Initialize display
    screen_width = master_maze.cols * master_maze.cell_size
    screen_height = master_maze.rows * master_maze.cell_size + 120
    screen = pygame.display.get_surface()
    
    if screen is None:
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("NEAT Maze Navigation - Advanced")
    
    clock = pygame.time.Clock()
    
    # Create agents
    nets = []
    agents = []
    ge = []
    
    agent_colors = [
        (30, 100, 200), (50, 150, 255), (100, 180, 255),
        (70, 130, 230), (135, 206, 250),
    ]
    
    # Initialize all fitness
    for genome_id, genome in genomes:
        genome.fitness = 0.1
    
    for idx, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        net.reset()
        
        agent_maze = master_maze.copy_with_fresh_food()
        agent = Agent(agent_maze, net, genome_id, MAX_STEPS)
        agent.color = agent_colors[idx % len(agent_colors)]
        
        nets.append(net)
        agents.append(agent)
        ge.append(genome)
    
    RENDER_EVERY = 5
    
    # Run simulation
    for step in range(MAX_STEPS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                for i, agent in enumerate(agents):
                    fitness = compute_fitness_v3(agent, agent.maze, generation_counter, ge[i])
                    ge[i].fitness = max(0.1, fitness)
                pygame.quit()
                sys.exit(0)
        
        # Update agents
        active_agents = 0
        for i, agent in enumerate(agents):
            if not agent.alive:
                continue
            
            active_agents += 1
            inputs = agent.get_inputs()
            
            outputs = None
            for _ in range(5):
                outputs = nets[i].activate(inputs)
            
            direction_idx = outputs.index(max(outputs))
            agent.step(direction_idx)
        
        # Render
        if step % RENDER_EVERY == 0 or step == MAX_STEPS - 1:
            screen.fill((40, 40, 40))
            draw_maze(screen, master_maze)
            draw_food(screen, master_maze)
            draw_all_agents(screen, agents, master_maze)
            
            best_agent = max(agents, key=lambda x: (x.collected_small + x.collected_big, x.steps) if x.alive else (0, 0))
            best_fitness = compute_fitness_v3(best_agent, best_agent.maze, generation_counter)
            
            sample = agents[:min(20, len(agents))]
            sample_fitnesses = [compute_fitness_v3(a, a.maze, generation_counter) for a in sample if a.alive]
            avg_fitness = sum(sample_fitnesses) / len(sample_fitnesses) if sample_fitnesses else 0
            
            total_small = sum(a.collected_small for a in agents)
            total_big = sum(a.collected_big for a in agents)
            elapsed = time.time() - gen_start_time
            
            draw_hud(screen, generation_counter, best_fitness, avg_fitness,
                     global_best_fitness, total_small, total_big, elapsed,
                     alive_count=active_agents, total_agents=len(agents))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        if active_agents == 0:
            break
    
    # Calculate final fitness with component tracking
    for i, agent in enumerate(agents):
        try:
            fitness = compute_fitness_v3(agent, agent.maze, generation_counter, ge[i])
            ge[i].fitness = max(0.1, float(fitness))
        except Exception as e:
            print(f"⚠️  Error: {e}")
            ge[i].fitness = 0.1
    
    # Verify
    for genome_id, genome in genomes:
        if genome.fitness is None or genome.fitness <= 0:
            genome.fitness = 0.1
    
    # Track best
    best_fitness = max(g.fitness for g in ge)
    if best_fitness > global_best_fitness:
        global_best_fitness = best_fitness
        best_idx = ge.index(max(ge, key=lambda g: g.fitness))
        global_best_genome = genomes[best_idx][1]
        
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(global_best_genome, f)
        print(f"    🏆 New best! Fitness: {global_best_fitness:.1f}")
    
    # ⭐ LOG FITNESS HISTORY
    best_fitness_history.append(best_fitness)
    
    # ⭐ TRACK COMPONENTS
    fitness_tracker.log_generation(generation_counter, genomes)
    
    # ⭐ ADJUST MUTATION RATES
    stagnation_counter = adjust_mutation_rates_v2(
        config, generation_counter, best_fitness_history, stagnation_counter
    )
    
    # ⭐ DIVERSITY INJECTION CHECK
    if stagnation_counter.get('diversity_injection_needed', False):
        # This needs to be handled in main.py population loop
        pass
    
    # ⭐ PERIODIC DIAGNOSTICS
    if generation_counter % 25 == 0 and generation_counter > 0:
        diagnosis = fitness_tracker.diagnose_fitness_imbalance(generation_counter)
        if diagnosis.get('imbalanced', False):
            print(f"\n    🔍 FITNESS IMBALANCE (Gen {generation_counter}):")
            print(f"       Dominant: {diagnosis['dominant_component']} ({diagnosis['contribution']})")
            print(f"       ⚠️  {diagnosis['recommendation']}\n")
    
    # Summary
    avg_fitness = sum(g.fitness for g in ge) / len(ge)
    best_agent = agents[ge.index(max(ge, key=lambda g: g.fitness))]
    alive_count = sum(1 for a in agents if a.alive)
    elapsed = time.time() - gen_start_time
    
    print(f"    Gen {generation_counter:3d} │ "
          f"Best: {best_fitness:6.1f} │ "
          f"Avg: {avg_fitness:6.1f} │ "
          f"Food: {best_agent.collected_small}s+{best_agent.collected_big}b │ "
          f"Alive: {alive_count:3d}/{len(agents):3d} │ "
          f"Time: {elapsed:5.1f}s")
    
    generation_counter += 1