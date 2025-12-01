"""
Main entry point with adaptive curriculum
"""
import neat
import os
import sys
import simulation

def run_neat(config_path, num_generations=50, resume=False, headless=False):
    """Run NEAT evolution."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    simulation.HEADLESS = headless
    simulation.TOTAL_GENERATIONS = num_generations
    
    if not headless:
        import pygame
        pygame.init()
    
    if resume:
        checkpoints = [f for f in os.listdir('.') if f.startswith('neat-checkpoint-')]
        if not checkpoints:
            print("⚠️  No checkpoints found - starting fresh\n")
            population = neat.Population(config)
            simulation.generation_counter = 0
        else:
            latest = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
            population = neat.Checkpointer.restore_checkpoint(latest)
            gen_num = int(latest.split('-')[-1])
            simulation.generation_counter = gen_num
            print(f"📂 Resumed from generation {gen_num}\n")
    else:
        population = neat.Population(config)
        simulation.generation_counter = 0
    
    # ⭐ SIMPLIFIED: Remove verbose reporters
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    checkpointer = neat.Checkpointer(generation_interval=10,
                                      filename_prefix='neat-checkpoint-')
    population.add_reporter(checkpointer)
    
    # ⭐ CLEAN HEADER
    mode = "Headless" if headless else "Visual"
    print(f"{'='*70}")
    print(f"  NEAT Training - {mode} Mode")
    print(f"{'='*70}")
    print(f"  Population: {config.pop_size} │ Generations: {num_generations} │ Food: 12→3")
    print(f"{'='*70}\n")
    
    try:
        winner = population.run(simulation.eval_genomes, num_generations)
        
        print(f"\n{'='*70}")
        print(f"  ✅ Training Complete")
        print(f"{'='*70}")
        print(f"  Best Fitness: {simulation.global_best_fitness:.2f}")
        print(f"  Saved: best_genome.pkl")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted")
        sys.exit(0)

def show_menu():
    """Display main menu."""
    print(f"\n{'='*70}")
    print(f"  NEAT Maze Navigation")
    print(f"{'='*70}\n")
    print(f"  1. New training (visual)")
    print(f"  2. New training (headless - faster)")
    print(f"  3. Resume training (visual)")
    print(f"  4. Resume training (headless)")
    print(f"  5. Exit")
    print(f"{'='*70}\n")
    
    while True:
        try:
            choice = input("Choice: ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return int(choice)
            print("Invalid. Enter 1-5.")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

def get_num_generations():
    """Get generations count."""
    while True:
        try:
            user_input = input("Generations (default 50): ").strip()
            if not user_input:
                return 50
            num_gens = int(user_input)
            if num_gens > 0:
                return num_gens
            print("Enter positive number.")
        except ValueError:
            print("Invalid input.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'config-maze.txt')
    
    choice = show_menu()
    
    if choice == 5:
        print("Goodbye!")
        sys.exit(0)
    
    num_gens = get_num_generations()
    
    if choice == 1:
        run_neat(config_path, num_generations=num_gens, resume=False, headless=False)
    elif choice == 2:
        run_neat(config_path, num_generations=num_gens, resume=False, headless=True)
    elif choice == 3:
        run_neat(config_path, num_generations=num_gens, resume=True, headless=False)
    elif choice == 4:
        run_neat(config_path, num_generations=num_gens, resume=True, headless=True)