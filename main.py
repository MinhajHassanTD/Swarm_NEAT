"""
Main entry point for NEAT maze navigation training.
"""
import os
import pygame
import neat
import pickle
from simulation import eval_genomes

def save_checkpoint(population, generation, filename='checkpoint.pkl'):
    """Save current population state to file."""
    checkpoint_data = {
        'population': population,
        'generation': generation
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"Checkpoint saved at generation {generation}")

def load_checkpoint(filename='checkpoint.pkl'):
    """Load population state from checkpoint file."""
    if not os.path.exists(filename):
        return None, 0
    
    with open(filename, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data['population'], checkpoint_data['generation']

def run_neat(config_path, num_generations=50, resume=False):
    """
    Run NEAT evolution for maze navigation.
    
    Args:
        config_path: Path to NEAT configuration file
        num_generations: Number of generations to evolve
        resume: Whether to resume from checkpoint
    """
    pygame.init()
    
    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Initialize or load population
    if resume:
        population, start_generation = load_checkpoint()
        if population is None:
            print("No checkpoint found. Starting fresh...")
            population = neat.Population(config)
            start_generation = 0
        else:
            print(f"Resuming from generation {start_generation}")
            population.config = config
    else:
        population = neat.Population(config)
        start_generation = 0
    
    # Add reporters for progress tracking
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Save checkpoints every 5 generations
    checkpoint_reporter = neat.Checkpointer(
        generation_interval=5,
        time_interval_seconds=None,
        filename_prefix='neat-checkpoint-'
    )
    population.add_reporter(checkpoint_reporter)
    
    # Run evolution
    print(f"\nStarting NEAT evolution for {num_generations} generations...")
    print(f"Starting from generation {start_generation}")
    print("=" * 60)
    
    try:
        winner = population.run(eval_genomes, num_generations)
        
        # Save results
        print("\n" + "=" * 60)
        print("Evolution complete!")
        print(f"Best genome fitness: {winner.fitness:.2f}")
        
        with open('winner_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        print("Winner genome saved to 'winner_genome.pkl'")
        
        final_generation = start_generation + num_generations
        save_checkpoint(population, final_generation, 'checkpoint.pkl')
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint...")
        
        current_generation = start_generation + population.generation
        save_checkpoint(population, current_generation, 'checkpoint.pkl')
        
        # Save best genome found so far
        best = None
        best_fitness = float('-inf')
        for g in population.population.values():
            if g.fitness is not None and g.fitness > best_fitness:
                best = g
                best_fitness = g.fitness
        
        if best is not None:
            with open('winner_genome.pkl', 'wb') as f:
                pickle.dump(best, f)
            print(f"Best genome (fitness: {best_fitness:.2f}) saved to 'winner_genome.pkl'")
    
    pygame.quit()
    return winner

def show_menu():
    """Display training menu and get user choice."""
    print("\n" + "=" * 60)
    print("NEAT MAZE NAVIGATION - TRAINING MENU")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Start NEW training (fresh population)")
    print("  2. RESUME from checkpoint (continue previous training)")
    print("  3. EXIT")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")

def get_num_generations():
    """Prompt user for number of generations to train."""
    while True:
        try:
            num = input("\nHow many generations to train? (default: 50): ").strip()
            if num == '':
                return 50
            num = int(num)
            if num > 0:
                return num
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == '__main__':
    # Locate configuration file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-maze.txt')
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create 'config-maze.txt' with NEAT parameters.")
        exit(1)
    
    # Get user choices
    choice = show_menu()
    
    if choice == '3':
        print("Exiting...")
        exit(0)
    
    num_gens = get_num_generations()
    
    # Start training
    if choice == '1':
        print("\nStarting NEW training...")
        run_neat(config_path, num_generations=num_gens, resume=False)
    elif choice == '2':
        if not os.path.exists('checkpoint.pkl') and not os.path.exists('neat-checkpoint-0'):
            print("\nNo checkpoint found!")
            print("Starting fresh training instead...")
            run_neat(config_path, num_generations=num_gens, resume=False)
        else:
            print("\nResuming from checkpoint...")
            run_neat(config_path, num_generations=num_gens, resume=True)
    
    print("\nTraining completed!")
    print("\nSaved files:")
    print("  - winner_genome.pkl (best genome)")
    print("  - checkpoint.pkl (latest checkpoint)")
    print("  - neat-checkpoint-* (periodic checkpoints)")