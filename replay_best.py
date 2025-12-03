"""
Replay best genome - SIMPLIFIED VERSION
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

def replay_genome(genome, config, genome_rank=1, num_runs=3, max_steps=600, fps=10):
    """Replay trained genome."""
    pygame.init()
    
    maze = Maze(DEFAULT_MAZE, cell_size=20, num_small_food=43, num_big_food=12)
    screen_width = maze.cols * maze.cell_size
    screen_height = maze.rows * maze.cell_size + 120
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(f"NEAT Replay - Genome #{genome_rank}")
    clock = pygame.time.Clock()
    
    stats = {'food': 0, 'steps': 0, 'collisions': 0}
    
    for run in range(num_runs):
        print(f"\n▶ Run {run + 1}/{num_runs}")
        
        maze = Maze(DEFAULT_MAZE, cell_size=20, num_small_food=43, num_big_food=12)
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
            
            fitness = compute_fitness(agent, maze, 999)
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


def show_genome_menu(top_5_genomes):
    """Show menu to select genome from top 5."""
    print("\n" + "="*60)
    print("  SELECT GENOME TO REPLAY")
    print("="*60)
    
    for i, (fitness, _) in enumerate(top_5_genomes, 1):
        print(f"  {i}. Genome #{i} - Fitness: {fitness:.1f}")
    
    print(f"  {len(top_5_genomes) + 1}. Back")
    print("="*60)
    
    while True:
        choice = input(f"\nChoice (1-{len(top_5_genomes) + 1}): ").strip()
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(top_5_genomes):
                return choice_num - 1  # Return index
            elif choice_num == len(top_5_genomes) + 1:
                return None
        except ValueError:
            pass
        print("❌ Invalid")


def show_menu():
    """Display menu."""
    print("\n" + "="*60)
    print("  GENOME REPLAY")
    print("="*60)
    print("\n  1. Replay from Top 5 genomes")
    print("  2. Replay best_genome.pkl (legacy)")
    print("  3. Exit")
    print("="*60)
    
    while True:
        choice = input("\nChoice (1-3): ").strip()
        if choice in ['1', '2', '3']:
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
    
    choice = show_menu()
    
    if choice == '1':
        # Load top 5 genomes
        if not os.path.exists('top_5_genomes.pkl'):
            print("❌ No top_5_genomes.pkl found. Train first!")
            sys.exit(1)
        
        with open('top_5_genomes.pkl', 'rb') as f:
            top_5_genomes = pickle.load(f)
        
        if not top_5_genomes:
            print("❌ No genomes saved yet!")
            sys.exit(1)
        
        genome_idx = show_genome_menu(top_5_genomes)
        if genome_idx is None:
            sys.exit(0)
        
        fitness, genome = top_5_genomes[genome_idx]
        print(f"\n✅ Selected Genome #{genome_idx + 1} (Fitness: {fitness:.1f})")
        
        num_runs = int(input("Runs (default 3): ").strip() or "3")
        fps = int(input("FPS (default 10): ").strip() or "10")
        replay_genome(genome, config, genome_rank=genome_idx + 1, num_runs=num_runs, fps=fps)
    
    elif choice == '2':
        # Legacy: Load best_genome.pkl
        if not os.path.exists('best_genome.pkl'):
            print("❌ No best_genome.pkl found!")
            sys.exit(1)
        
        with open('best_genome.pkl', 'rb') as f:
            genome = pickle.load(f)
        
        print("✅ Loaded best_genome.pkl")
        num_runs = int(input("Runs (default 3): ").strip() or "3")
        fps = int(input("FPS (default 10): ").strip() or "10")
        replay_genome(genome, config, genome_rank=1, num_runs=num_runs, fps=fps)
    
    elif choice == '3':
        sys.exit(0)