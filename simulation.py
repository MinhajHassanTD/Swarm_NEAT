"""
NEAT evaluation function for maze navigation with energy management.
"""
import pygame
import neat
import time
import pickle
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_all_agents, draw_hud
from fitness import compute_fitness

# Simulation parameters
MAX_STEPS = 400
FPS = 20

# Global tracking across generations
generation_counter = 0
global_best_fitness = 0.0
global_best_genome = None

def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population by running maze simulation.
    
    This function is called by NEAT for each generation. Each genome
    controls one agent that navigates the maze. Fitness is calculated
    based on food collection, exploration, and survival.
    """
    global generation_counter, global_best_fitness, global_best_genome
    
    gen_start_time = time.time()
    
    # Create master maze for display
    master_maze = Maze(DEFAULT_MAZE, cell_size=20)
    
    # Initialize Pygame display
    screen_width = master_maze.cols * master_maze.cell_size
    screen_height = master_maze.rows * master_maze.cell_size + 120
    screen = pygame.display.get_surface()
    
    if screen is None:
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("NEAT Maze Navigation - Natural Learning")
    
    clock = pygame.time.Clock()
    
    # Create agents for each genome (with independent maze instances)
    nets = []
    agents = []
    ge = []
    
    agent_colors = [
        (30, 100, 200), (50, 150, 255), (100, 180, 255),
        (70, 130, 230), (135, 206, 250),
    ]
    
    for idx, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0.0
        
        # Create RecurrentNetwork instead of FeedForwardNetwork
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        # Each agent gets its own maze instance with independent food
        agent_maze = master_maze.copy_with_fresh_food()
        agent = Agent(agent_maze, net, genome_id, MAX_STEPS)
        agent.color = agent_colors[idx % len(agent_colors)]
        
        nets.append(net)
        agents.append(agent)
        ge.append(genome)
    
    # Run simulation for this generation
    for step in range(MAX_STEPS):
        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        
        # Update each agent
        active_agents = 0
        for i, agent in enumerate(agents):
            if not agent.alive:
                continue
            
            active_agents += 1
            
            # Get sensor inputs (now 11 inputs)
            inputs = agent.get_inputs()
            
            # Network decides direction (now 5 outputs including stay)
            output = nets[i].activate(inputs)
            direction_idx = output.index(max(output))
            
            # Execute movement
            agent.step(direction_idx)
        
        # Render current state (use master_maze for visualization)
        screen.fill((40, 40, 40))
        draw_maze(screen, master_maze)
        draw_food(screen, master_maze)
        draw_all_agents(screen, agents, master_maze)
        
        # Calculate display metrics
        current_fitnesses = []
        for agent in agents:
            # Use new fitness function for display
            display_fitness = compute_fitness(agent, agent.maze, generation_counter)
            current_fitnesses.append(display_fitness)
        
        best_fitness = max(current_fitnesses) if current_fitnesses else 0
        avg_fitness = sum(current_fitnesses) / len(current_fitnesses) if current_fitnesses else 0
        total_small = sum(agent.collected_small for agent in agents)
        total_big = sum(agent.collected_big for agent in agents)
        elapsed_time = time.time() - gen_start_time
        
        draw_hud(screen, generation_counter, best_fitness, avg_fitness, 
                 global_best_fitness, total_small, total_big, elapsed_time,
                 alive_count=active_agents, total_agents=len(agents))
        
        pygame.display.flip()
        clock.tick(FPS)
        
        # Stop early if all agents died
        if active_agents == 0:
            break
    
    # Calculate final fitness for each genome using new fitness function
    for i, agent in enumerate(agents):
        fitness = compute_fitness(agent, agent.maze, generation_counter)
        ge[i].fitness = max(0.0, fitness)
    
    # Track best genome across all generations
    best_fitness = max(g.fitness for g in ge)
    if best_fitness > global_best_fitness:
        global_best_fitness = best_fitness
        best_idx = ge.index(max(ge, key=lambda g: g.fitness))
        global_best_genome = genomes[best_idx][1]
        
        # Save best genome to file
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(global_best_genome, f)
        print(f"  New best genome found! Fitness: {global_best_fitness:.2f}")
    
    # Print generation summary
    avg_fitness = sum(g.fitness for g in ge) / len(ge)
    best_agent = agents[ge.index(max(ge, key=lambda g: g.fitness))]
    alive_count = sum(1 for a in agents if a.alive)
    
    print(f"Gen {generation_counter}: "
          f"Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, "
          f"Global={global_best_fitness:.2f}, "
          f"Food={best_agent.collected_small}s/{best_agent.collected_big}b, "
          f"Alive={alive_count}/{len(agents)}, "
          f"Collisions={best_agent.collisions}")
    
    generation_counter += 1