"""
Replay best genome - Uses fitness_v3 for consistency
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
from fitness_v3 import compute_fitness_v3  # ⭐ USE V3

def replay_genome(genome, config, num_runs=3, max_steps=400, fps=10):
    """Replay trained genome."""
    pygame.init()
    
    maze = Maze(DEFAULT_MAZE, cell_size=20)
    screen_width = maze.cols * maze.cell_size
    screen_height = maze.rows * maze.cell_size + 120
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NEAT Replay")
    clock = pygame.time.Clock()
    
    stats = {'food': 0, 'steps': 0, 'collisions': 0}
    
    for run in range(num_runs):
        print(f"\n▶ Run {run + 1}/{num_runs}")
        
        maze = Maze(DEFAULT_MAZE, cell_size=20)
        net = neat.nn.RecurrentNetwork.create(genome, config)
        net.reset()
        
        agent = Agent(maze, net, max_steps=max_steps)
        agent.color = (30, 100, 200)
        
        run_start = time.time()
        
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    sys.exit(0)
            
            if not agent.alive:
                break
            
            inputs = agent.get_inputs()
            
            output = None
            for _ in range(5):
                output = net.activate(inputs)
            
            direction_idx = output.index(max(output))
            agent.step(direction_idx)
            
            # Render
            screen.fill((40, 40, 40))
            draw_maze(screen, maze)
            draw_food(screen, maze)
            draw_agent(screen, agent, maze)
            
            # ⭐ USE V3 (generation 999 for replay mode)
            fitness = compute_fitness_v3(agent, maze, 999)
            elapsed = time.time() - run_start
            
            draw_hud(screen, 999, fitness, fitness, fitness,
                     agent.collected_small, agent.collected_big, elapsed,
                     alive_count=1 if agent.alive else 0, total_agents=1)
            
            pygame.display.flip()
            clock.tick(fps)
        
        run_food = agent.collected_small + agent.collected_big
        print(f"  Food: {agent.collected_small}s + {agent.collected_big}b = {run_food}")
        print(f"  Steps: {agent.steps}/{max_steps}")
        print(f"  Collisions: {agent.collisions}")
        
        stats['food'] += run_food
        stats['steps'] += agent.steps
        stats['collisions'] += agent.collisions
    
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({num_runs} runs)")
    print(f"{'='*60}")
    print(f"Avg food: {stats['food'] / num_runs:.2f}")
    print(f"Avg steps: {stats['steps'] / num_runs:.1f}")
    print(f"Avg collisions: {stats['collisions'] / num_runs:.1f}")
    print(f"{'='*60}\n")
    
    pygame.quit()


def show_menu():
    """Display menu."""
    print("\n" + "="*60)
    print("  GENOME REPLAY")
    print("="*60)
    print("\n  1. Replay best genome")
    print("  2. Exit")
    print("="*60)
    
    while True:
        choice = input("\nChoice (1-2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("❌ Invalid")


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-maze.txt')
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    genome_file = 'best_genome.pkl'
    if not os.path.exists(genome_file):
        print(f"❌ No saved genome: {genome_file}")
        sys.exit(1)
    
    print(f"✅ Loading {genome_file}...")
    with open(genome_file, 'rb') as f:
        genome = pickle.load(f)
    
    choice = show_menu()
    
    if choice == '1':
        num_runs = int(input("Runs (default 3): ").strip() or "3")
        fps = int(input("FPS (default 10): ").strip() or "10")
        replay_genome(genome, config, num_runs=num_runs, fps=fps)
    
    elif choice == '2':
        sys.exit(0)