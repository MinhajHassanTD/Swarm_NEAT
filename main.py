"""
Main entry point - SIMPLIFIED VERSION
"""
import os
import sys
import neat
import pygame
import pickle
from simulation import eval_genomes

def run_neat(config_path, num_generations=50, resume=False, headless=False, 
             food_randomize_every=0, spawn_randomize_every=0):
    """Run NEAT evolution."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Set parameters in simulation module
    import simulation
    simulation.HEADLESS = headless
    simulation.FOOD_RANDOMIZE_EVERY = food_randomize_every
    simulation.SPAWN_RANDOMIZE_EVERY = spawn_randomize_every
    
    # Only initialize pygame if not headless
    if not headless:
        pygame.init()
    else:
        print("\nRunning in HEADLESS mode (no visuals) - much faster!\n")
    
    if resume:
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
        if checkpoint_files:
            checkpoint_nums = [int(f.split('-')[-1]) for f in checkpoint_files]
            latest_gen = max(checkpoint_nums)
            checkpoint_file = f'neat-checkpoint-{latest_gen}'
            
            print(f"\nRestoring from {checkpoint_file}")
            population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            population.config = config
            
            import simulation
            simulation.generation_counter = latest_gen
        else:
            print("\nNo checkpoints, starting fresh")
            population = neat.Population(config)
    else:
        population = neat.Population(config)
        import simulation
        simulation.generation_counter = 0
    
    # Add reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    checkpointer = neat.Checkpointer(generation_interval=10,
                                      filename_prefix='neat-checkpoint-')
    population.add_reporter(checkpointer)
    
    try:
        print(f"\n{'='*70}")
        print(f"  NEAT TRAINING")
        print(f"{'='*70}")
        print(f"  Population: {config.pop_size} | Generations: {num_generations}")
        print(f"{'='*70}\n")
        
        # Run evolution
        population.run(eval_genomes, num_generations)
        
        winner = population.best_genome
        
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        print(f"\n{'='*70}")
        print(f"  COMPLETE!")
        print(f"{'='*70}")
        print(f"  Best Fitness: {winner.fitness:.1f}")
        print(f"  Saved: best_genome.pkl")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        if not headless:
            pygame.quit()


def show_menu():
    """Display menu."""
    print("\n" + "="*70)
    print("  NEAT MAZE NAVIGATION")
    print("="*70)
    print("\n  1. Start NEW training (with visuals)")
    print("  2. Start NEW training (headless - faster)")
    print("  3. RESUME from checkpoint (with visuals)")
    print("  4. RESUME from checkpoint (headless - faster)")
    print("  5. Exit")
    print("="*70)
    
    while True:
        choice = input("\nChoice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid")


def get_num_generations():
    """Get number of generations."""
    while True:
        try:
            num = input("Generations (default 50): ").strip()
            if num == '':
                return 50
            num = int(num)
            if num > 0:
                return num
            print("Must be positive")
        except ValueError:
            print("Invalid number")


def get_food_randomization():
    """Ask user about food randomization."""
    while True:
        try:
            num = input("Randomize food every N generations (0=never, default 3): ").strip()
            if num == '':
                return 3
            num = int(num)
            if num >= 0:
                return num
            print("Must be >= 0")
        except ValueError:
            print("Invalid number")


def get_spawn_randomization():
    """Ask user about spawn randomization."""
    while True:
        try:
            num = input("Randomize spawn every N generations (0=never, default 3): ").strip()
            if num == '':
                return 3
            num = int(num)
            if num >= 0:
                return num
            print("Must be >= 0")
        except ValueError:
            print("Invalid number")


def get_spawn_randomization():
    """Ask user about spawn randomization."""
    while True:
        try:
            num = input("Randomize spawn every N generations (0=never, default 3): ").strip()
            if num == '':
                return 3
            num = int(num)
            if num >= 0:
                return num
            print("Must be >= 0")
        except ValueError:
            print("Invalid number")


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-maze.txt')
    
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    
    choice = show_menu()
    
    if choice == '1':
        num_gens = get_num_generations()
        food_rand = get_food_randomization()
        spawn_rand = get_spawn_randomization()
        run_neat(config_path, num_generations=num_gens, resume=False, headless=False, 
                 food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
    
    elif choice == '2':
        num_gens = get_num_generations()
        food_rand = get_food_randomization()
        spawn_rand = get_spawn_randomization()
        run_neat(config_path, num_generations=num_gens, resume=False, headless=True,
                 food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
    
    elif choice == '3':
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
        if not checkpoint_files:
            print("\nNo checkpoints")
            num_gens = get_num_generations()
            food_rand = get_food_randomization()
            spawn_rand = get_spawn_randomization()
            run_neat(config_path, num_generations=num_gens, resume=False, headless=False,
                     food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
        else:
            num_gens = get_num_generations()
            food_rand = get_food_randomization()
            spawn_rand = get_spawn_randomization()
            run_neat(config_path, num_generations=num_gens, resume=True, headless=False,
                     food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
    
    elif choice == '4':
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
        if not checkpoint_files:
            print("\nNo checkpoints")
            num_gens = get_num_generations()
            food_rand = get_food_randomization()
            spawn_rand = get_spawn_randomization()
            run_neat(config_path, num_generations=num_gens, resume=False, headless=True,
                     food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
        else:
            num_gens = get_num_generations()
            food_rand = get_food_randomization()
            spawn_rand = get_spawn_randomization()
            run_neat(config_path, num_generations=num_gens, resume=True, headless=True,
                     food_randomize_every=food_rand, spawn_randomize_every=spawn_rand)
    
    elif choice == '5':
        sys.exit(0)