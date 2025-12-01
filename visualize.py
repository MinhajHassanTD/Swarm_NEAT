"""
Pygame visualization functions for maze navigation simulation.
"""
import pygame

# Color definitions
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
BLUE = (50, 150, 255)
YELLOW = (255, 220, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

def draw_maze(surface, maze):
    """Render the maze grid with walls and paths."""
    cell_size = maze.cell_size
    
    for y, row in enumerate(maze.layout):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            
            if cell == '1':
                pygame.draw.rect(surface, BLACK, rect)
            else:
                pygame.draw.rect(surface, LIGHT_GRAY, rect)
                pygame.draw.rect(surface, GRAY, rect, 1)

def draw_food(surface, maze):
    """Render all uneaten food items."""
    for food in maze.food_items:
        if not food['eaten']:
            pixel_x, pixel_y = maze.to_pixel(food['grid_x'], food['grid_y'])
            
            if food['big']:
                radius = max(5, maze.cell_size // 4)
                pygame.draw.circle(surface, RED, (pixel_x, pixel_y), radius)
            else:
                radius = max(3, maze.cell_size // 6)
                pygame.draw.circle(surface, YELLOW, (pixel_x, pixel_y), radius)

def draw_agent(surface, agent, maze):
    """Render a single agent with energy bar."""
    if not agent.alive:
        return
    
    pixel_x, pixel_y = maze.to_pixel(agent.gx, agent.gy)
    radius = max(6, maze.cell_size // 3)
    
    # Color indicates energy level
    color = getattr(agent, 'color', BLUE)
    if agent.is_starving():
        color = ORANGE
    
    pygame.draw.circle(surface, color, (pixel_x, pixel_y), radius)
    
    # Draw energy bar above agent
    bar_width = maze.cell_size - 4
    bar_height = 3
    bar_x = pixel_x - bar_width // 2
    bar_y = pixel_y - radius - 6
    
    # Red background
    pygame.draw.rect(surface, RED, (bar_x, bar_y, bar_width, bar_height))
    
    # Green foreground shows remaining energy
    energy_ratio = max(0.0, min(1.0, agent.energy / agent.max_energy))
    energy_width = int(bar_width * energy_ratio)
    if energy_width > 0:
        pygame.draw.rect(surface, GREEN, (bar_x, bar_y, energy_width, bar_height))

def draw_all_agents(surface, agents, maze):
    """Render all agents in the simulation."""
    for agent in agents:
        draw_agent(surface, agent, maze)

def draw_hud(surface, generation, best_fitness, avg_fitness, global_best_fitness, 
             total_small=0, total_big=0, elapsed_time=0, alive_count=0, total_agents=0,
             available_small=10, available_big=6):  # ⭐ NEW PARAMS
    """
    Render heads-up display with generation statistics.
    
    Shows current generation info, fitness metrics, food collection,
    and legend for visual elements.
    """
    width = surface.get_width()
    height = surface.get_height()
    
    # Create semi-transparent panel at bottom
    hud_height = 120
    hud_surface = pygame.Surface((width, hud_height))
    hud_surface.set_alpha(200)
    hud_surface.fill(GRAY)
    surface.blit(hud_surface, (0, height - hud_height))
    
    # Font setup
    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 18)
    tiny_font = pygame.font.Font(None, 16)
    
    y_offset = height - hud_height + 5
    
    # Title
    text = font.render("NEAT Maze Navigation", True, WHITE)
    surface.blit(text, (10, y_offset))
    
    # Generation info
    text = small_font.render(f"Gen: {generation} | Time: {elapsed_time:.1f}s | Alive: {alive_count}/{total_agents}", True, GREEN)
    surface.blit(text, (10, y_offset + 22))
    
    # Fitness metrics
    text = font.render("Fitness:", True, CYAN)
    surface.blit(text, (10, y_offset + 40))
    
    text = small_font.render(f"  Best: {best_fitness:.2f}", True, GREEN)
    surface.blit(text, (10, y_offset + 58))
    
    text = small_font.render(f"  Avg: {avg_fitness:.2f}", True, WHITE)
    surface.blit(text, (10, y_offset + 76))
    
    text = small_font.render(f"  Global Best: {global_best_fitness:.2f}", True, YELLOW)
    surface.blit(text, (10, y_offset + 94))
    
    # ⭐ NEW: Food collection stats with rate
    mid_x = width // 2 - 80
    text = small_font.render(f"Food Collected:", True, CYAN)
    surface.blit(text, (mid_x, y_offset + 40))
    
    # Calculate collection rate (⭐ EXPLICIT PARENTHESES)
    total_available = available_small + available_big
    total_collected = total_small + total_big
    collection_rate = ((total_collected / total_available) * 100) if total_available > 0 else 0.0
    
    text = small_font.render(f"{total_small}/{available_small} small, {total_big}/{available_big} big", True, YELLOW)
    surface.blit(text, (mid_x, y_offset + 58))
    
    # Show collection rate
    rate_color = GREEN if collection_rate >= 70 else YELLOW if collection_rate >= 40 else RED
    text = small_font.render(f"Rate: {collection_rate:.1f}%", True, rate_color)
    surface.blit(text, (mid_x, y_offset + 76))
    
    # Legend (unchanged)
    legend_x = width - 200
    text = font.render("Legend:", True, WHITE)
    surface.blit(text, (legend_x, y_offset))
    
    pygame.draw.circle(surface, BLUE, (legend_x + 10, y_offset + 25), 5)
    text = tiny_font.render("= Agent (energized)", True, WHITE)
    surface.blit(text, (legend_x + 20, y_offset + 20))
    
    pygame.draw.circle(surface, ORANGE, (legend_x + 10, y_offset + 40), 5)
    text = tiny_font.render("= Agent (starving)", True, WHITE)
    surface.blit(text, (legend_x + 20, y_offset + 35))
    
    pygame.draw.circle(surface, YELLOW, (legend_x + 10, y_offset + 55), 3)
    text = tiny_font.render("= Small (+35 energy)", True, WHITE)
    surface.blit(text, (legend_x + 20, y_offset + 50))
    
    pygame.draw.circle(surface, RED, (legend_x + 10, y_offset + 70), 5)
    text = tiny_font.render("= Big (+70 energy)", True, WHITE)
    surface.blit(text, (legend_x + 20, y_offset + 65))
    
    text = tiny_font.render("Press Q to quit", True, LIGHT_GRAY)
    surface.blit(text, (legend_x, y_offset + 92))