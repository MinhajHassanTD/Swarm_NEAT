"""
NEAT evaluation with adaptive food curriculum.
"""
import pygame
import neat
import time
import pickle
import sys
import random
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_all_agents, draw_hud
from fitness import compute_fitness

# Parameters
MAX_STEPS = 400
FPS = 30
ACTIVATION_CYCLES = 5
HEADLESS = False

# Global state
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None
TOTAL_GENERATIONS = 50

def get_adaptive_food_counts(current_gen, total_gens):
    """Calculate food counts based on training progress."""
    progress = min(current_gen / max(total_gens, 1), 1.0)
    
    # Linear interpolation: 12s+7b → 3s+1b
    small_max = max(int(12 - (9 * progress)), 3)
    small_min = max(int(10 - (7 * progress)), 3)
    big_max = max(int(7 - (6 * progress)), 1)
    big_min = max(int(5 - (4 * progress)), 1)
    
    # Ensure min <= max
    if small_min > small_max:
        small_min = small_max
    if big_min > big_max:
        big_min = big_max
    
    return random.randint(small_min, small_max), random.randint(big_min, big_max)

def eval_genomes(genomes, config):
    """Evaluate all genomes."""
    global generation_counter, global_best_fitness, global_best_genome, TOTAL_GENERATIONS
    
    gen_start = time.time()
    
    # Generate maze with dynamic food
    num_small, num_big = get_adaptive_food_counts(generation_counter, TOTAL_GENERATIONS)
    master_maze = Maze(DEFAULT_MAZE, cell_size=20, 
                       num_small_food=num_small, num_big_food=num_big)
    
    # Initialize display (if visual)
    screen = clock = None
    if not HEADLESS:
        width = master_maze.cols * master_maze.cell_size
        height = master_maze.rows * master_maze.cell_size + 120
        screen = pygame.display.get_surface() or pygame.display.set_mode((width, height))
        pygame.display.set_caption("NEAT Maze Navigation")
        clock = pygame.time.Clock()
    
    # Create agents
    nets, agents, ge = [], [], []
    colors = [(30, 100, 200), (50, 150, 255), (100, 180, 255), (70, 130, 230), (135, 206, 250)]
    
    for genome_id, genome in genomes:
        genome.fitness = 0.1
    
    for idx, (genome_id, genome) in enumerate(genomes):
        net = neat.nn.RecurrentNetwork.create(genome, config)
        net.reset()
        
        agent = Agent(master_maze.copy_with_fresh_food(), net, genome_id, MAX_STEPS)
        agent.color = colors[idx % len(colors)]
        
        nets.append(net)
        agents.append(agent)
        ge.append(genome)
    
    # Run simulation
    for step in range(MAX_STEPS):
        # Handle quit events
        if not HEADLESS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    print("\n⚠️  Early exit - partial fitness applied")
                    for i, agent in enumerate(agents):
                        partial = compute_fitness(agent, agent.maze, 
                                                 generation_counter, TOTAL_GENERATIONS)
                        ge[i].fitness = max(0.1, partial * 0.8)
                    pygame.quit()
                    sys.exit(0)
        
        # Update agents
        active = 0
        for i, agent in enumerate(agents):
            if not agent.alive:
                continue
            
            active += 1
            inputs = agent.get_inputs()
            
            for _ in range(ACTIVATION_CYCLES):
                outputs = nets[i].activate(inputs)
            
            agent.step(outputs.index(max(outputs)))
        
        # Render (every 5 frames)
        if not HEADLESS and (step % 5 == 0 or step == MAX_STEPS - 1):
            screen.fill((40, 40, 40))
            draw_maze(screen, master_maze)
            draw_food(screen, master_maze)
            draw_all_agents(screen, agents, master_maze)
            
            best_agent = max(agents, key=lambda x: (x.collected_small + x.collected_big, x.steps) 
                            if x.alive else (0, 0))
            best_fit = compute_fitness(best_agent, best_agent.maze, generation_counter, TOTAL_GENERATIONS)
            
            sample = agents[:min(20, len(agents))]
            avg_fit = sum(compute_fitness(a, a.maze, generation_counter, TOTAL_GENERATIONS) 
                         for a in sample if a.alive) / max(len([a for a in sample if a.alive]), 1)
            
            draw_hud(screen, generation_counter, best_fit, avg_fit, global_best_fitness,
                    sum(a.collected_small for a in agents), sum(a.collected_big for a in agents),
                    time.time() - gen_start, active, len(agents), num_small, num_big)
            
            pygame.display.flip()
            clock.tick(FPS)
        
        if active == 0:
            break
    
    # Calculate final fitness
    for i, agent in enumerate(agents):
        try:
            ge[i].fitness = max(0.1, compute_fitness(agent, agent.maze, 
                                                     generation_counter, TOTAL_GENERATIONS))
        except Exception as e:
            ge[i].fitness = 0.1
    
    # Verify fitness
    for _, genome in genomes:
        if genome.fitness is None or genome.fitness <= 0:
            genome.fitness = 0.1
    
    # Track best
    best_fitness = max(g.fitness for g in ge)
    if best_fitness > global_best_fitness:
        global_best_fitness = best_fitness
        global_best_genome = genomes[ge.index(max(ge, key=lambda g: g.fitness))][1]
        
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(global_best_genome, f)
    
    # ⭐ SIMPLIFIED OUTPUT
    avg_fit = sum(g.fitness for g in ge) / len(ge)
    best_agent = agents[ge.index(max(ge, key=lambda g: g.fitness))]
    progress = (generation_counter / max(TOTAL_GENERATIONS, 1)) * 100
    
    # Calculate progress through current phase
    progress = generation_counter / max(TOTAL_GENERATIONS, 1)
    
    if progress < 0.33:
        phase = "1"
        phase_progress = (progress / 0.33) * 100
    elif progress < 0.67:
        phase = "2"
        phase_progress = ((progress - 0.33) / 0.34) * 100
    else:
        phase = "3"
        phase_progress = ((progress - 0.67) / 0.33) * 100
    
    total = num_small + num_big
    collected = best_agent.collected_small + best_agent.collected_big
    rate = (collected / total * 100) if total > 0 else 0
    
    # Compact single-line output
    status = "🏆" if best_fitness > global_best_fitness else "  "
    print(f"{status} Gen {generation_counter:3d} (P{phase}:{phase_progress:3.0f}%) │ "
          f"Food: {collected:2d}/{total:2d} ({rate:3.0f}%) │ "
          f"Fit: {best_fitness:6.1f} (avg {avg_fit:5.1f}) │ "
          f"{time.time() - gen_start:4.1f}s")
    
    generation_counter += 1