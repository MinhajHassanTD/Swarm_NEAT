"""
NEAT evaluation function - SIMPLIFIED VERSION
"""
import pygame
import neat
import time
import pickle
import sys
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_all_agents, draw_hud
from fitness import compute_fitness

# Simulation parameters
MAX_STEPS = 600
FPS = 30
HEADLESS = False  # Set to True to disable visualization

# Global tracking
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None

def eval_genomes(genomes, config):
    """Evaluate all genomes by running maze simulation."""
    global generation_counter, global_best_fitness, global_best_genome
    
    gen_start_time = time.time()
    
    # Create master maze
    master_maze = Maze(DEFAULT_MAZE, cell_size=20)
    
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
                    for i, agent in enumerate(agents):
                        fitness = compute_fitness(agent, agent.maze, generation_counter)
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
            
            # Calculate metrics
            best_agent = max(agents, key=lambda x: (x.collected_small + x.collected_big, x.steps) if x.alive else (0, 0))
            best_fitness = compute_fitness(best_agent, best_agent.maze, generation_counter)
            
            # Quick avg fitness (sample 20)
            sample = agents[:min(20, len(agents))]
            sample_fitnesses = [compute_fitness(a, a.maze, generation_counter) for a in sample if a.alive]
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
    
    # Calculate final fitness
    for i, agent in enumerate(agents):
        try:
            fitness = compute_fitness(agent, agent.maze, generation_counter)
            ge[i].fitness = max(0.1, float(fitness))
        except Exception as e:
            print(f"⚠️  Error computing fitness for genome {ge[i].key}: {e}")
            ge[i].fitness = 0.1
    
    # Verify all have fitness
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