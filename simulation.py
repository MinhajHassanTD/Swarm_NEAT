"""
NEAT evaluation function - SIMPLIFIED VERSION
"""
import pygame
import neat
import time
import pickle
import sys
import copy
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_all_agents, draw_hud
from fitness import compute_fitness

# Simulation parameters
MAX_STEPS = 600
FPS = 30
HEADLESS = False  # Set to True to disable visualization

FOOD_RANDOMIZE_EVERY = 3  # Randomize food every N generations (0 = never)
SAVED_FOOD_POSITIONS = None  # Store food positions

# Global tracking
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None
top_5_genomes = []  

def eval_genomes(genomes, config):
    """Evaluate all genomes by running maze simulation."""
    global generation_counter, global_best_fitness, global_best_genome, SAVED_FOOD_POSITIONS, top_5_genomes
    
    gen_start_time = time.time()
    
    # Create master maze
    master_maze = Maze(DEFAULT_MAZE, cell_size=20, num_small_food=43, num_big_food=12)
    
    if SAVED_FOOD_POSITIONS is None:
        # First time: save initial random positions
        SAVED_FOOD_POSITIONS = copy.deepcopy(master_maze.food_items)
        print(f"\n    💾 Food positions initialized!\n")
    elif FOOD_RANDOMIZE_EVERY > 0 and generation_counter > 0 and generation_counter % FOOD_RANDOMIZE_EVERY == 0:
        # Regenerate and save new positions
        master_maze.randomize_food()
        SAVED_FOOD_POSITIONS = copy.deepcopy(master_maze.food_items)
        print(f"\n    🔄 Food positions randomized! (Generation {generation_counter})\n")
    else:
        # Reuse saved positions
        master_maze.food_items = copy.deepcopy(SAVED_FOOD_POSITIONS)
    
    # Initialize display (only if not headless)
    screen = None
    clock = None
    
    if not HEADLESS:
        screen_width = master_maze.cols * master_maze.cell_size
        screen_height = master_maze.rows * master_maze.cell_size + 120
        screen = pygame.display.get_surface()
        
        if screen is None:
            screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("NEAT Maze Navigation")
        
        clock = pygame.time.Clock()
    
    # Create agents
    nets = []
    agents = []
    ge = []
    
    agent_colors = [
        (30, 100, 200), (50, 150, 255), (100, 180, 255),
        (70, 130, 230), (135, 206, 250),
    ]
    
    # Initialize all fitness to 0.1
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
    
    # Render every 5 frames for speed (only when not headless)
    RENDER_EVERY = 5
    
    # Run simulation
    for step in range(MAX_STEPS):
        # Handle quit (only if rendering)
        if not HEADLESS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    # Compute final fitness before quitting
                    population_stats = {
                        'avg_food': sum(a.collected_small + a.collected_big for a in agents) / max(len(agents), 1),
                        'max_food': max((a.collected_small + a.collected_big) for a in agents) if agents else 0,
                        'avg_survival': sum(a.steps for a in agents) / max(len(agents), 1)
                    }
                    for i, agent in enumerate(agents):
                        fitness = compute_fitness(agent, agent.maze, generation_counter, population_stats)
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
            
            # Multi-cycle activation
            outputs = None
            for _ in range(5):
                outputs = nets[i].activate(inputs)
            
            direction_idx = outputs.index(max(outputs))
            agent.step(direction_idx)
        
        # Render (skip frames for speed, only if not headless)
        if not HEADLESS and (step % RENDER_EVERY == 0 or step == MAX_STEPS - 1):
            screen.fill((40, 40, 40))
            draw_maze(screen, master_maze)
            draw_food(screen, master_maze)
            draw_all_agents(screen, agents, master_maze)
            
            # Calculate metrics for display
            population_stats_temp = {
                'avg_food': sum(a.collected_small + a.collected_big for a in agents) / max(len(agents), 1),
                'max_food': max((a.collected_small + a.collected_big) for a in agents) if agents else 0,
                'avg_survival': sum(a.steps for a in agents) / max(len(agents), 1)
            }
            
            best_agent = max(agents, key=lambda x: compute_fitness(x, x.maze, generation_counter, population_stats_temp))
            best_fitness = compute_fitness(best_agent, best_agent.maze, generation_counter, population_stats_temp)
            
            # Quick avg fitness (sample 20)
            sample = agents[:min(20, len(agents))]
            sample_fitnesses = [compute_fitness(a, a.maze, generation_counter, population_stats_temp) for a in sample]
            avg_fitness = sum(sample_fitnesses) / len(sample_fitnesses) if sample_fitnesses else 0
            
            total_small = sum(a.collected_small for a in agents)
            total_big = sum(a.collected_big for a in agents)
            elapsed = time.time() - gen_start_time
            
            draw_hud(screen, generation_counter, best_fitness, avg_fitness,
                     global_best_fitness, total_small, total_big, elapsed,
                     alive_count=active_agents, total_agents=len(agents))
            
            pygame.display.flip()
            clock.tick(FPS)
        
        # Stop if all dead
        if active_agents == 0:
            break
    
    # Calculate population stats BEFORE computing fitness
    total_food = sum(a.collected_small + a.collected_big for a in agents)
    avg_food = total_food / max(len(agents), 1)
    max_food = max((a.collected_small + a.collected_big) for a in agents) if agents else 0
    avg_survival = sum(a.steps for a in agents) / max(len(agents), 1)
    
    population_stats = {
        'avg_food': avg_food,
        'max_food': max_food,
        'avg_survival': avg_survival
    }
    
    # Compute fitness for each agent WITH STATS
    for i, agent in enumerate(agents):
        fitness = compute_fitness(agent, agent.maze, generation_counter, population_stats)
        ge[i].fitness = fitness
    
    # Verify all have fitness
    for genome_id, genome in genomes:
        if genome.fitness is None or genome.fitness <= 0:
            genome.fitness = 0.1
    
    # Find best genome this generation
    best_fitness_this_gen = 0.0
    best_genome_this_gen = None
    best_agent_this_gen = None
    
    for i, (genome_id, genome) in enumerate(genomes):
        if genome.fitness > best_fitness_this_gen:
            best_fitness_this_gen = genome.fitness
            best_genome_this_gen = genome
            best_agent_this_gen = agents[i]
    
    if best_genome_this_gen:
        # Check if this genome is already in top 5 (by genome key/id)
        genome_ids_in_top5 = [g.key for _, g in top_5_genomes]
        is_duplicate = best_genome_this_gen.key in genome_ids_in_top5
        
        if not is_duplicate:
            # Check if this genome deserves to be in top 5
            should_add = False
            
            if len(top_5_genomes) < 5:
                # List not full yet, always add
                should_add = True
            else:
                # Check if better than 5th place
                fifth_place_fitness = top_5_genomes[4][0]
                if best_fitness_this_gen > fifth_place_fitness:
                    should_add = True
            
            if should_add:
                top_5_genomes.append((best_fitness_this_gen, best_genome_this_gen))
                top_5_genomes.sort(key=lambda x: x[0], reverse=True)
                top_5_genomes = top_5_genomes[:5]
                
                # Save top 5 genomes
                with open('top_5_genomes.pkl', 'wb') as f:
                    pickle.dump(top_5_genomes, f)
                
                print(f"    Added to Top 5! (Fitness: {best_fitness_this_gen:.1f}, ID: {best_genome_this_gen.key})")
    
    # Update global best if this generation is better
    if best_fitness_this_gen > global_best_fitness:
        global_best_fitness = best_fitness_this_gen
        global_best_genome = best_genome_this_gen
        
        # Save best genome (backward compatibility)
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(global_best_genome, f)
        
        print(f"    New Global Best! Fitness: {global_best_fitness:.1f} | "
              f"Food: {best_agent_this_gen.collected_small}s+{best_agent_this_gen.collected_big}b | "
              f"Steps: {best_agent_this_gen.steps}")
    
    # Summary
    avg_fitness = sum(genome.fitness for _, genome in genomes) / len(genomes)
    alive_count = sum(1 for a in agents if a.alive)
    elapsed = time.time() - gen_start_time
    
    print(f"    Gen {generation_counter:3d} │ "
          f"Best: {best_fitness_this_gen:6.1f} │ "
          f"Avg: {avg_fitness:6.1f} │ "
          f"AvgFood: {avg_food:4.1f} │ "
          f"MaxFood: {max_food:2d} │ "
          f"Alive: {alive_count:3d}/{len(agents):3d} │ "
          f"Time: {elapsed:5.1f}s")
    
    generation_counter += 1