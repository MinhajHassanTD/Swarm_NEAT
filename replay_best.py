"""
Replay best genome with configurable food counts.
"""
import os
import sys
import neat
import pygame
import pickle
import time
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_agent, draw_hud
from fitness import compute_fitness

ACTIVATION_CYCLES = 5

def replay_genome(genome, config, num_runs=3, max_steps=400, fps=10, 
                  num_small=10, num_big=6):
    """Replay trained genome."""
    pygame.init()
    
    maze = Maze(DEFAULT_MAZE, cell_size=20)
    screen = pygame.display.set_mode((maze.cols * maze.cell_size, 
                                     maze.rows * maze.cell_size + 120))
    pygame.display.set_caption("NEAT Replay")
    clock = pygame.time.Clock()
    
    stats = {'collected': 0, 'steps': 0, 'collisions': 0, 'rate': 0.0}
    
    for run in range(num_runs):
        print(f"▶ Run {run + 1}/{num_runs}... ", end='', flush=True)
        
        maze = Maze(DEFAULT_MAZE, cell_size=20, 
                   num_small_food=num_small, num_big_food=num_big)
        
        net = neat.nn.RecurrentNetwork.create(genome, config)
        net.reset()
        agent = Agent(maze, net, max_steps=max_steps)
        agent.color = (30, 100, 200)
        
        start = time.time()
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
            
            if not agent.alive:
                break
            
            inputs = agent.get_inputs()
            for _ in range(ACTIVATION_CYCLES):
                outputs = net.activate(inputs)
            agent.step(outputs.index(max(outputs)))
            
            screen.fill((40, 40, 40))
            draw_maze(screen, maze)
            draw_food(screen, maze)
            draw_agent(screen, agent, maze)
            
            fitness = compute_fitness(agent, maze, 999, 999)
            draw_hud(screen, 999, fitness, fitness, fitness,
                    agent.collected_small, agent.collected_big, time.time() - start,
                    1 if agent.alive else 0, 1, num_small, num_big)
            
            pygame.display.flip()
            clock.tick(fps)
        
        collected = agent.collected_small + agent.collected_big
        total = num_small + num_big
        rate = (collected / total * 100) if total > 0 else 0
        
        print(f"Food: {collected}/{total} ({rate:.0f}%) │ Steps: {agent.steps} │ Collisions: {agent.collisions}")
        
        stats['collected'] += collected
        stats['steps'] += agent.steps
        stats['collisions'] += agent.collisions
        stats['rate'] += rate
    
    total = num_small + num_big
    print(f"\n{'='*70}")
    print(f"  Average: {stats['collected']/num_runs:.1f}/{total} food ({stats['rate']/num_runs:.0f}%) │ "
          f"{stats['steps']/num_runs:.0f} steps │ {stats['collisions']/num_runs:.1f} collisions")
    print(f"{'='*70}\n")
    
    pygame.quit()


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'config-maze.txt')
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    if not os.path.exists('best_genome.pkl'):
        print("❌ No saved genome found")
        sys.exit(1)
    
    with open('best_genome.pkl', 'rb') as f:
        genome = pickle.load(f)
    
    print(f"\n{'='*70}")
    print(f"  Genome Replay")
    print(f"{'='*70}\n")
    
    try:
        num_runs = int(input("Runs (default 3): ").strip() or "3")
        fps = int(input("FPS (default 10): ").strip() or "10")
        num_small = int(input("Small food (default 10): ").strip() or "10")
        num_big = int(input("Big food (default 6): ").strip() or "6")
    except ValueError:
        num_runs, fps, num_small, num_big = 3, 10, 10, 6
    
    print()  # Blank line before runs
    replay_genome(genome, config, num_runs, 400, fps, num_small, num_big)