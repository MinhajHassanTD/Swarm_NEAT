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

# Simulation parameters
MAX_STEPS = 200
FPS = 15

# Fitness reward values
SMALL_FOOD_REWARD = 150.0
BIG_FOOD_REWARD = 300.0
COMPLETION_BONUS = 200.0
EXPLORATION_REWARD = 30.0
SURVIVAL_REWARD = 50.0
#penalties
COLLISION_PENALTY_WEIGHT = 4.0
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
    maze = Maze(DEFAULT_MAZE, cell_size=20)
    
    # Initialize Pygame display
    screen_width = maze.cols * maze.cell_size
    screen_height = maze.rows * maze.cell_size + 120
    screen = pygame.display.get_surface()
    
    if screen is None:
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("NEAT Maze Navigation - Natural Learning")
    
    clock = pygame.time.Clock()
    
    # Create agents for each genome
    nets = []
    agents = []
    ge = []
    
    agent_colors = [
        (30, 100, 200), (50, 150, 255), (100, 180, 255),
        (70, 130, 230), (135, 206, 250),
    ]
    
    for idx, (genome_id, genome) in enumerate(genomes):
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        agent = Agent(maze, net, genome_id, MAX_STEPS)
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
            
            # Get sensor inputs
            inputs = agent.get_inputs()
            
            # Network decides direction
            output = nets[i].activate(inputs)
            direction_idx = output.index(max(output))
            
            # Execute movement
            agent.step(direction_idx)
        
        # Render current state
        screen.fill((40, 40, 40))
        draw_maze(screen, maze)
        draw_food(screen, maze)
        draw_all_agents(screen, agents, maze)
        
        # Calculate display metrics
        current_fitnesses = []
        for i, agent in enumerate(agents):
            display_fitness = 0.0
            display_fitness += agent.collected_small * SMALL_FOOD_REWARD
            display_fitness += agent.collected_big * BIG_FOOD_REWARD
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
        
        # Stop early if all agents died or maze completed
        if active_agents == 0:
            break
        
        if any(a.collected_small + a.collected_big == len(maze.food_items) for a in agents):
            break
    
    # Calculate final fitness for each genome
    for i, agent in enumerate(agents):
        fitness = 0.0
        
        # Reward food collection
        fitness += agent.collected_small * SMALL_FOOD_REWARD
        fitness += agent.collected_big * BIG_FOOD_REWARD
        
        # Reward exploration progress
        exploration_score = agent.get_exploration_score()
        fitness += exploration_score * EXPLORATION_REWARD
        # Penalize collisions
        fitness -= agent.collisions * COLLISION_PENALTY_WEIGHT
        
        
        # Reward survival with remaining energy
        if agent.alive and agent.energy > 0:
            energy_ratio = agent.energy / agent.max_energy
            fitness += SURVIVAL_REWARD * energy_ratio
        
        # Bonus for completing maze
        total_food = len(maze.food_items)
        if agent.collected_small + agent.collected_big == total_food:
            fitness += COMPLETION_BONUS
            
            # Extra bonus for efficient completion
            efficiency = (MAX_STEPS - agent.steps) / MAX_STEPS
            fitness += efficiency * 50.0
        
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
    maze.reset_food()