"""
Replay and visualize the best trained genome.
"""
import pygame
import neat
import pickle
import time
from maze import Maze, DEFAULT_MAZE
from agent import Agent
from visualize import draw_maze, draw_food, draw_agent, draw_hud

def replay_best_genome(genome_file='best_genome.pkl', config_file='config-maze.txt', 
                       max_steps=500, fps=15, num_runs=3):
    """
    Load and visualize the best genome navigating the maze.
    
    Args:
        genome_file: Path to saved genome pickle file
        config_file: Path to NEAT config file
        max_steps: Maximum steps per run
        fps: Frames per second for visualization
        num_runs: Number of times to replay
    """
    # Load the saved genome
    try:
        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)
        print(f"✓ Loaded genome from {genome_file}")
    except FileNotFoundError:
        print(f"✗ Error: {genome_file} not found!")
        print("  Make sure you've trained a model first.")
        return
    
    # Load NEAT config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Initialize pygame
    pygame.init()
    maze = Maze(DEFAULT_MAZE, cell_size=20)
    screen_width = maze.cols * maze.cell_size
    screen_height = maze.rows * maze.cell_size + 120
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("NEAT Best Agent Replay")
    clock = pygame.time.Clock()
    
    print(f"\n{'='*60}")
    print(f"  REPLAYING BEST GENOME")
    print(f"{'='*60}")
    print(f"Runs: {num_runs} | Max Steps: {max_steps} | FPS: {fps}")
    print(f"Controls: Q=Quit, SPACE=Pause, R=Restart, +/- = Speed")
    print(f"{'='*60}\n")
    
    for run in range(num_runs):
        print(f"\n▶ Run {run + 1}/{num_runs}")
        
        # Reset maze with fresh food
        maze = Maze(DEFAULT_MAZE, cell_size=20)
        
        # Create network from genome
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        # Create agent
        agent = Agent(maze, net, max_steps=max_steps)
        agent.color = (50, 255, 100)  # Green for best agent
        
        # Simulation variables
        step = 0
        paused = False
        current_fps = fps
        
        # Run simulation
        while step < max_steps and agent.alive:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print(f"  {'⏸ Paused' if paused else '▶ Resumed'}")
                    elif event.key == pygame.K_r:
                        print(f"  🔄 Restarting run...")
                        break
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        current_fps = min(current_fps * 2, 240)
                        print(f"  ⏩ Speed: {current_fps} FPS")
                    elif event.key == pygame.K_MINUS:
                        current_fps = max(current_fps // 2, 1)
                        print(f"  ⏪ Speed: {current_fps} FPS")
            
            if paused:
                clock.tick(10)
                continue
            
            # Get network inputs
            inputs = agent.get_inputs()
            
            # Get network output
            output = net.activate(inputs)
            direction_idx = output.index(max(output))
            
            # Execute action
            agent.step(direction_idx)
            step += 1
            
            # Render
            screen.fill((40, 40, 40))
            draw_maze(screen, maze)
            draw_food(screen, maze)
            draw_agent(screen, agent, maze)
            
            # Draw custom HUD for replay
            font = pygame.font.Font(None, 24)
            small_font = pygame.font.Font(None, 18)
            
            # Title
            title = font.render(f"BEST AGENT REPLAY - Run {run+1}/{num_runs}", True, (255, 255, 255))
            screen.blit(title, (10, screen_height - 110))
            
            # Stats
            stats = [
                f"Step: {step}/{max_steps}",
                f"Food: {agent.collected_small}s + {agent.collected_big}b = {agent.collected_small + agent.collected_big} total",
                f"Energy: {agent.energy:.1f}/{agent.max_energy}",
                f"Collisions: {agent.collisions}",
                f"Alive: {'Yes' if agent.alive else 'No'}",
            ]
            
            for i, stat in enumerate(stats):
                color = (100, 255, 100) if agent.alive else (255, 100, 100)
                text = small_font.render(stat, True, color)
                screen.blit(text, (10, screen_height - 85 + i * 15))
            
            # Controls hint
            hint = small_font.render("SPACE=Pause | R=Restart | +/-=Speed | Q=Quit", True, (150, 150, 150))
            screen.blit(hint, (10, screen_height - 15))
            
            pygame.display.flip()
            clock.tick(current_fps)
        
        # Run summary
        print(f"  ✓ Completed: {step} steps")
        print(f"    Food: {agent.collected_small} small + {agent.collected_big} big")
        print(f"    Energy: {agent.energy:.1f}")
        print(f"    Collisions: {agent.collisions}")
        print(f"    Status: {'Survived' if agent.alive else 'Died'}")
        
        # Pause between runs
        if run < num_runs - 1:
            print(f"\n  ⏳ Next run in 2 seconds...")
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"  ALL RUNS COMPLETED")
    print(f"{'='*60}\n")
    
    # Wait for user to close
    print("Press Q or close window to exit...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    waiting = False
        clock.tick(10)
    
    pygame.quit()
    print("✓ Replay complete!")


def show_genome_info(genome_file='best_genome.pkl', config_file='config-maze.txt'):
    """
    Display information about the saved genome.
    """
    try:
        with open(genome_file, 'rb') as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Error: {genome_file} not found!")
        return
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    print(f"\n{'='*60}")
    print(f"  GENOME INFORMATION")
    print(f"{'='*60}")
    print(f"Genome ID: {genome.key}")
    print(f"Fitness: {genome.fitness:.2f}" if hasattr(genome, 'fitness') and genome.fitness else "Fitness: N/A")
    
    # Count different node types
    num_hidden = len(genome.nodes)  # ✅ FIXED: genome.nodes only contains hidden nodes
    
    # Count connections
    total_connections = len(genome.connections)
    enabled_connections = sum(1 for c in genome.connections.values() if c.enabled)
    disabled_connections = total_connections - enabled_connections
    
    print(f"\nNetwork Structure:")
    print(f"  Inputs: 11 (configured)")
    print(f"  Outputs: 5 (configured)")
    print(f"  Hidden Nodes: {num_hidden}")
    print(f"  Total Nodes: {11 + 5 + num_hidden}")
    print(f"\nConnections:")
    print(f"  Total: {total_connections}")
    print(f"  Enabled: {enabled_connections} ({enabled_connections/total_connections*100:.1f}%)" if total_connections > 0 else "  Enabled: 0")
    print(f"  Disabled: {disabled_connections} ({disabled_connections/total_connections*100:.1f}%)" if total_connections > 0 else "  Disabled: 0")
    
    # Show hidden node details
    if num_hidden > 0:
        print(f"\nHidden Node Details:")
        for node_id, node in list(genome.nodes.items())[:10]:
            print(f"  Node {node_id}: activation={node.activation}, bias={node.bias:.3f}, response={node.response:.3f}")
        if num_hidden > 10:
            print(f"  ... and {num_hidden - 10} more hidden nodes")
    
    # Show connection details
    print(f"\nConnection Details (showing first 10):")
    for i, (conn_key, conn) in enumerate(list(genome.connections.items())[:10]):
        status = "✓" if conn.enabled else "✗"
        in_node, out_node = conn_key
        print(f"  {status} {in_node:3d} → {out_node:3d}  weight: {conn.weight:7.3f}")
    if total_connections > 10:
        print(f"  ... and {total_connections - 10} more connections")
    
    # Network complexity metrics
    if total_connections > 0:
        avg_weight = sum(abs(c.weight) for c in genome.connections.values()) / total_connections
        max_weight = max(abs(c.weight) for c in genome.connections.values())
        min_weight = min(abs(c.weight) for c in genome.connections.values())
        
        print(f"\nWeight Statistics:")
        print(f"  Average: {avg_weight:.3f}")
        print(f"  Max: {max_weight:.3f}")
        print(f"  Min: {min_weight:.3f}")
    
    print(f"{'='*60}\n")


def compare_with_random(genome_file='best_genome.pkl', config_file='config-maze.txt', 
                        max_steps=500, num_trials=5):
    """
    Compare trained genome with random agent.
    """
    # Load trained genome
    try:
        with open(genome_file, 'rb') as f:
            trained_genome = pickle.load(f)
    except FileNotFoundError:
        print(f"✗ Error: {genome_file} not found!")
        return
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Create random genome
    random_genome = config.genome_type(0)
    random_genome.configure_new(config.genome_config)
    
    print(f"\n{'='*60}")
    print(f"  TRAINED vs RANDOM COMPARISON")
    print(f"{'='*60}")
    print(f"Trials: {num_trials} | Max Steps: {max_steps}\n")
    
    trained_results = []
    random_results = []
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}:")
        
        # Test trained genome
        maze = Maze(DEFAULT_MAZE, cell_size=20)
        net = neat.nn.RecurrentNetwork.create(trained_genome, config)
        agent = Agent(maze, net, max_steps=max_steps)
        
        for _ in range(max_steps):
            if not agent.alive:
                break
            inputs = agent.get_inputs()
            output = net.activate(inputs)
            direction_idx = output.index(max(output))
            agent.step(direction_idx)
        
        trained_food = agent.collected_small + agent.collected_big
        trained_results.append(trained_food)
        print(f"  Trained: {agent.collected_small}s + {agent.collected_big}b = {trained_food} food, {agent.steps} steps")
        
        # Test random genome
        maze = Maze(DEFAULT_MAZE, cell_size=20)
        net = neat.nn.RecurrentNetwork.create(random_genome, config)
        agent = Agent(maze, net, max_steps=max_steps)
        
        for _ in range(max_steps):
            if not agent.alive:
                break
            inputs = agent.get_inputs()
            output = net.activate(inputs)
            direction_idx = output.index(max(output))
            agent.step(direction_idx)
        
        random_food = agent.collected_small + agent.collected_big
        random_results.append(random_food)
        print(f"  Random:  {agent.collected_small}s + {agent.collected_big}b = {random_food} food, {agent.steps} steps")
    
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Trained Agent:")
    print(f"  Avg Food: {sum(trained_results)/len(trained_results):.2f}")
    print(f"  Best: {max(trained_results)} | Worst: {min(trained_results)}")
    print(f"\nRandom Agent:")
    print(f"  Avg Food: {sum(random_results)/len(random_results):.2f}")
    print(f"  Best: {max(random_results)} | Worst: {min(random_results)}")
    
    improvement = (sum(trained_results) - sum(random_results)) / len(trained_results)
    print(f"\n✨ Improvement: +{improvement:.2f} food per trial")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    print("\n" + "="*60)
    print("  NEAT BEST GENOME REPLAY")
    print("="*60)
    print("\nOptions:")
    print("  1. Replay best genome (visual)")
    print("  2. Show genome information")
    print("  3. Compare with random agent")
    print("  4. Exit")
    print("="*60)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        num_runs = input("Number of runs (default 3): ").strip()
        num_runs = int(num_runs) if num_runs else 3
        
        max_steps = input("Max steps per run (default 500): ").strip()
        max_steps = int(max_steps) if max_steps else 500
        
        fps = input("FPS (default 15): ").strip()
        fps = int(fps) if fps else 15
        
        replay_best_genome(num_runs=num_runs, max_steps=max_steps, fps=fps)
    
    elif choice == '2':
        show_genome_info()
    
    elif choice == '3':
        num_trials = input("Number of trials (default 5): ").strip()
        num_trials = int(num_trials) if num_trials else 5
        
        max_steps = input("Max steps per trial (default 500): ").strip()
        max_steps = int(max_steps) if max_steps else 500
        
        compare_with_random(num_trials=num_trials, max_steps=max_steps)
    
    elif choice == '4':
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice!")