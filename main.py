"""
Main entry point for NEAT maze navigation training.
"""
import os
import sys
import neat
import pygame
import pickle
from simulation import eval_genomes

def save_checkpoint(population, generation, filename='checkpoint.pkl'):
    """Save current population state to file."""
    checkpoint_data = {
        'population': population,
        'generation': generation,
        'config': population.config,
        'species': population.species,
        'generation_count': population.generation
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at generation {generation}")

def load_checkpoint(filename='checkpoint.pkl'):
    """Load population state from checkpoint file."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data

def run_neat(config_path, num_generations=50, resume=False):
    """
    Run NEAT algorithm to evolve maze navigation agents.
    
    Args:
        config_path: Path to NEAT configuration file
        num_generations: Number of generations to run
        resume: Whether to resume from checkpoint
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Initialize pygame
    pygame.init()
    
    # Create or restore population
    if resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            print(f"Resuming from generation {checkpoint_data['generation']}")
            
            # Use NEAT's built-in checkpoint restore instead
            # This properly handles node ID indexing
            checkpoint_file = 'neat-checkpoint-' + str(checkpoint_data['generation'])
            if os.path.exists(checkpoint_file):
                population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
                print(f"✓ Restored from {checkpoint_file}")
            else:
                print(f"✗ NEAT checkpoint {checkpoint_file} not found")
                print("  Starting fresh training instead...")
                population = neat.Population(config)
        else:
            print("No checkpoint found, starting fresh training...")
            population = neat.Population(config)
    else:
        population = neat.Population(config)
    
    # Add reporters for statistics
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Add checkpointer to save every 5 generations
    checkpointer = neat.Checkpointer(generation_interval=5, 
                                      time_interval_seconds=None,
                                      filename_prefix='neat-checkpoint-')
    population.add_reporter(checkpointer)
    
    try:
        # Run NEAT evolution
        print(f"\n{'='*60}")
        print(f"  Starting NEAT Training")
        print(f"{'='*60}")
        print(f"Generations: {num_generations}")
        print(f"Population: {config.pop_size}")
        print(f"Resume: {resume}")
        print(f"{'='*60}\n")
        
        winner = population.run(eval_genomes, num_generations)
        
        # Save final best genome
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"{'='*60}")
        print(f"Best fitness: {winner.fitness:.2f}")
        print(f"Saved to: best_genome.pkl")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving checkpoint...")
        
        # Save checkpoint
        save_checkpoint(population, population.generation)
        
        print("✓ Checkpoint saved. You can resume training later.")
        sys.exit(0)
    
    finally:
        pygame.quit()

def show_menu():
    """Display main menu and get user choice."""
    print("\n" + "="*60)
    print("  NEAT MAZE NAVIGATION - TRAINING MENU")
    print("="*60)
    print("\nOptions:")
    print("  1. Start NEW training")
    print("  2. RESUME from checkpoint")
    print("  3. Exit")
    print("="*60)
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def get_num_generations():
    """Get number of generations from user."""
    while True:
        try:
            num = input("Enter number of generations (default 50): ").strip()
            if num == '':
                return 50
            num = int(num)
            if num > 0:
                return num
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == '__main__':
    # Path to NEAT config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-maze.txt')
    
    # Verify config file exists
    if not os.path.exists(config_path):
        print(f"✗ Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Show menu
    choice = show_menu()
    
    if choice == '1':
        # New training
        num_gens = get_num_generations()
        print(f"\n🚀 Starting new training for {num_gens} generations...")
        run_neat(config_path, num_generations=num_gens, resume=False)
    
    elif choice == '2':
        # Resume training
        # Check for NEAT checkpoints
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
        if not checkpoint_files:
            print("\n✗ No NEAT checkpoints found!")
            print("  Starting new training instead...")
            num_gens = get_num_generations()
            run_neat(config_path, num_generations=num_gens, resume=False)
        else:
            # Find latest checkpoint
            checkpoint_nums = [int(f.split('-')[-1]) for f in checkpoint_files]
            latest_gen = max(checkpoint_nums)
            print(f"\n✓ Found checkpoint at generation {latest_gen}")
            
            num_gens = get_num_generations()
            print(f"\n🔄 Resuming training from generation {latest_gen}...")
            print(f"   Will run for {num_gens} more generations")
            run_neat(config_path, num_generations=num_gens, resume=True)
    
    elif choice == '3':
        print("\nGoodbye!")
        sys.exit(0)