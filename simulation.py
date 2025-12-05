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
HEADLESS = False

FOOD_RANDOMIZE_EVERY = 3  # Randomize food every N generations (0 = never)
SPAWN_RANDOMIZE_EVERY = 3  # Randomize spawn every N generations (0 = never)
SAVED_FOOD_POSITIONS = None
SAVED_SPAWN_POSITION = None 

# Global tracking
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None
top_5_genomes = []  # Top 5 on current config
top_5_robust_genomes = []  

def test_genome_robustness(genome, config, num_tests=3):
    """Test genome on multiple random configurations and return average fitness."""
    total_fitness = 0.0
    
    for test in range(num_tests):
        # Create maze with random food/spawn
        test_maze = Maze(DEFAULT_MAZE, cell_size=20, num_small_food=43, num_big_food=12)
        test_maze.randomize_food()
        test_maze.randomize_spawn()
        
        # Run agent
        net = neat.nn.RecurrentNetwork.create(genome, config)
        net.reset()
        agent = Agent(test_maze, net, max_steps=MAX_STEPS)
        
        for step in range(MAX_STEPS):
            if not agent.alive:
                break
            inputs = agent.get_inputs()
            output = net.activate(inputs)
            direction = output.index(max(output))
            agent.step(direction)
        
        # Calculate fitness
        fitness = compute_fitness(agent, test_maze, generation_counter, None)
        total_fitness += fitness
    
    return total_fitness / num_tests


def eval_genomes(genomes, config):
    """Evaluate all genomes by running maze simulation."""
    global generation_counter, global_best_fitness, global_best_genome
    global SAVED_FOOD_POSITIONS, SAVED_SPAWN_POSITION, top_5_genomes, top_5_robust_genomes
    
    gen_start_time = time.time()
    
    # Create master maze
    master_maze = Maze(DEFAULT_MAZE, cell_size=20, num_small_food=43, num_big_food=12)
    
    if SAVED_FOOD_POSITIONS is None:
        SAVED_FOOD_POSITIONS = copy.deepcopy(master_maze.food_items)
        print(f"\n    Food positions initialized!\n")
    elif FOOD_RANDOMIZE_EVERY > 0 and generation_counter > 0 and generation_counter % FOOD_RANDOMIZE_EVERY == 0:
        master_maze.randomize_food()
        SAVED_FOOD_POSITIONS = copy.deepcopy(master_maze.food_items)
        print(f"\n    Food positions randomized! (Generation {generation_counter})\n")
    else:
        master_maze.food_items = copy.deepcopy(SAVED_FOOD_POSITIONS)
    
    # SPAWN RANDOMIZATION
    if SAVED_SPAWN_POSITION is None:
        SAVED_SPAWN_POSITION = master_maze.start_pos
        print(f"    Spawn position initialized: {SAVED_SPAWN_POSITION}\n")
    elif SPAWN_RANDOMIZE_EVERY > 0 and generation_counter > 0 and generation_counter % SPAWN_RANDOMIZE_EVERY == 0:
        master_maze.randomize_spawn()
        SAVED_SPAWN_POSITION = master_maze.start_pos
        print(f"    Spawn position randomized: {SAVED_SPAWN_POSITION}\n")
    else:
        master_maze.start_pos = SAVED_SPAWN_POSITION
    
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
    
    # ========== TOP 5 GLOBAL (Current Config) ==========
    if best_genome_this_gen:
        genome_ids_in_top5 = [g.key for _, g in top_5_genomes]
        is_duplicate = best_genome_this_gen.key in genome_ids_in_top5
        
        if not is_duplicate:
            should_add = False
            
            if len(top_5_genomes) < 5:
                should_add = True
            else:
                fifth_place_fitness = top_5_genomes[4][0]
                if best_fitness_this_gen > fifth_place_fitness:
                    should_add = True
            
            if should_add:
                top_5_genomes.append((best_fitness_this_gen, best_genome_this_gen))
                top_5_genomes.sort(key=lambda x: x[0], reverse=True)
                top_5_genomes = top_5_genomes[:5]
                
                with open('top_5_genomes.pkl', 'wb') as f:
                    pickle.dump(top_5_genomes, f)
                
                print(f"    Added to Top 5 Global! (Fitness: {best_fitness_this_gen:.1f}, ID: {best_genome_this_gen.key})")
    
    # ========== TOP 5 ROBUST (Tested on Multiple Configs) ==========
    if best_genome_this_gen and generation_counter > 0 and generation_counter % 5 == 0:
        print(f"\n    Testing robustness of best genome...")
        robust_fitness = test_genome_robustness(best_genome_this_gen, config, num_tests=3)
        print(f"    Robust fitness (avg of 3 tests): {robust_fitness:.1f}")
        print(f"    Original fitness: {best_fitness_this_gen:.1f}\n")
        
        # Check if should add to robust top 5
        genome_ids_in_robust = [g.key for _, g in top_5_robust_genomes]
        is_duplicate_robust = best_genome_this_gen.key in genome_ids_in_robust
        
        if not is_duplicate_robust:
            should_add_robust = False
            
            if len(top_5_robust_genomes) < 5:
                should_add_robust = True
            else:
                fifth_place_robust = top_5_robust_genomes[4][0]
                if robust_fitness > fifth_place_robust:
                    should_add_robust = True
            
            if should_add_robust:
                top_5_robust_genomes.append((robust_fitness, best_genome_this_gen))
                top_5_robust_genomes.sort(key=lambda x: x[0], reverse=True)
                top_5_robust_genomes = top_5_robust_genomes[:5]
                
                with open('top_5_robust_genomes.pkl', 'wb') as f:
                    pickle.dump(top_5_robust_genomes, f)
                
                print(f"    Added to Top 5 Robust! (Fitness: {robust_fitness:.1f}, ID: {best_genome_this_gen.key})")
    
    # Update global best (use original fitness)
    if best_fitness_this_gen > global_best_fitness:
        global_best_fitness = best_fitness_this_gen
        global_best_genome = best_genome_this_gen
        
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