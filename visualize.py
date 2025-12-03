"""
Pygame visualization functions for maze navigation simulation.
Refined for a modern, 'Dark Mode' aesthetic with responsive layout.
"""
import pygame
import math

# --- Modern Color Palette (Dracula/Cyberpunk inspired) ---
COLORS = {
    'BG': (30, 33, 40),            # Dark Slate Background
    'WALL': (55, 60, 70),          # Lighter Slate Wall
    'WALL_HIGHLIGHT': (80, 85, 95),# Top/Left Wall Edge
    'WALL_SHADOW': (20, 23, 28),   # Bottom/Right Wall Edge
    'GRID_LINES': (40, 44, 52),    # Subtle floor grid
    
    'AGENT_FULL': (0, 220, 255),   # Neon Cyan
    'AGENT_LOW': (255, 160, 0),    # Neon Orange
    'AGENT_GLOW': (0, 220, 255),   # Transparent Glow
    
    'FOOD_SMALL': (255, 230, 80),  # Soft Gold
    'FOOD_BIG': (255, 80, 100),    # Soft Red/Pink
    
    'HUD_BG': (20, 22, 26),        # Very dark panel
    'TEXT_MAIN': (240, 240, 240),  # Off-white
    'TEXT_DIM': (150, 160, 170),   # Grey text
    'TEXT_ACCENT': (100, 255, 150) # Green text
}

# Font cache to prevent reloading every frame
_FONTS = {}

def get_font(size, bold=False):
    """Helper to load system fonts efficiently."""
    key = (size, bold)
    if key not in _FONTS:
        # Try to use modern system fonts, fallback to Arial
        try:
            _FONTS[key] = pygame.font.SysFont("segoeui", size, bold=bold)
        except:
            _FONTS[key] = pygame.font.SysFont("arial", size, bold=bold)
    return _FONTS[key]

def draw_glow(surface, color, pos, radius, alpha=100):
    """Draws a transparent glowing circle."""
    glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*color, alpha), (radius, radius), radius)
    surface.blit(glow_surf, (pos[0] - radius, pos[1] - radius))

def draw_maze(surface, maze):
    """Render the maze with a 3D beveled block style."""
    surface.fill(COLORS['BG'])
    cell_size = maze.cell_size
    
    for y, row in enumerate(maze.layout):
        for x, cell in enumerate(row):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            
            if cell == '1':
                # Main Wall Body
                pygame.draw.rect(surface, COLORS['WALL'], rect)
                
                # 3D Bevel Effect
                # Top and Left (Highlight)
                pygame.draw.line(surface, COLORS['WALL_HIGHLIGHT'], rect.topleft, rect.topright, 2)
                pygame.draw.line(surface, COLORS['WALL_HIGHLIGHT'], rect.topleft, rect.bottomleft, 2)
                # Bottom and Right (Shadow)
                pygame.draw.line(surface, COLORS['WALL_SHADOW'], rect.bottomleft, rect.bottomright, 2)
                pygame.draw.line(surface, COLORS['WALL_SHADOW'], rect.topright, rect.bottomright, 2)
            else:
                # Subtle grid pattern for floor
                pygame.draw.rect(surface, COLORS['GRID_LINES'], rect, 1)

def draw_food(surface, maze):
    """Render food with a glowing effect."""
    # Pulse animation based on time
    pulse = math.sin(pygame.time.get_ticks() * 0.005) * 2
    
    for food in maze.food_items:
        if not food['eaten']:
            pixel_x, pixel_y = maze.to_pixel(food['grid_x'], food['grid_y'])
            center = (pixel_x, pixel_y)
            
            if food['big']:
                base_radius = max(5, maze.cell_size // 4)
                radius = base_radius + pulse
                draw_glow(surface, COLORS['FOOD_BIG'], center, int(radius + 4), 60)
                pygame.draw.circle(surface, COLORS['FOOD_BIG'], center, int(radius))
                # Add a white spec for shine
                pygame.draw.circle(surface, (255, 255, 255), (pixel_x - 2, pixel_y - 2), 2)
            else:
                base_radius = max(3, maze.cell_size // 6)
                pygame.draw.circle(surface, COLORS['FOOD_SMALL'], center, base_radius)

def draw_agent(surface, agent, maze):
    """Render agent with an aura and a sleek energy bar."""
    if not agent.alive:
        return
    
    pixel_x, pixel_y = maze.to_pixel(agent.gx, agent.gy)
    radius = max(6, maze.cell_size // 3)
    
    # Determine Color based on state
    base_color = COLORS['AGENT_FULL']
    if hasattr(agent, 'color'): base_color = agent.color
    
    if agent.is_starving():
        base_color = COLORS['AGENT_LOW']
    
    # Draw Glow Aura
    draw_glow(surface, base_color, (pixel_x, pixel_y), radius + 6, 50)
    
    # Draw Main Body (Antialiased for smoothness)
    pygame.draw.circle(surface, base_color, (pixel_x, pixel_y), radius)
    # Inner darker circle to make it look like a ring/eye
    pygame.draw.circle(surface, (0, 0, 0), (pixel_x, pixel_y), radius - 2)
    pygame.draw.circle(surface, base_color, (pixel_x, pixel_y), radius - 4)

    # --- Sleek Energy Bar ---
    bar_width = maze.cell_size
    bar_height = 4
    bar_x = pixel_x - bar_width // 2
    bar_y = pixel_y - radius - 8
    
    # Background (Dark Grey)
    pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
    
    # Energy Fill
    energy_ratio = max(0.0, min(1.0, agent.energy / agent.max_energy))
    energy_width = int(bar_width * energy_ratio)
    
    # Color gradient for health (Green -> Yellow -> Red)
    bar_color = (
        min(255, int(255 * (2 * (1 - energy_ratio)))), 
        min(255, int(255 * (2 * energy_ratio))), 
        0
    )
    
    if energy_width > 0:
        pygame.draw.rect(surface, bar_color, (bar_x, bar_y, energy_width, bar_height), border_radius=2)

def draw_all_agents(surface, agents, maze):
    """Render all agents."""
    for agent in agents:
        draw_agent(surface, agent, maze)

def draw_hud(surface, generation, best_fitness, avg_fitness, global_best_fitness, 
             total_small=0, total_big=0, elapsed_time=0, alive_count=0, total_agents=0):
    """
    Render a responsive dashboard-style HUD.
    Adjusts layout based on screen width to prevent overlapping.
    """
    width = surface.get_width()
    height = surface.get_height()
    hud_height = 110
    
    # 1. Draw Dashboard Background
    overlay = pygame.Surface((width, hud_height))
    overlay.fill(COLORS['HUD_BG'])
    overlay.set_alpha(240)
    
    # Top Border Line
    pygame.draw.line(overlay, COLORS['WALL_HIGHLIGHT'], (0, 0), (width, 0), 2)
    surface.blit(overlay, (0, height - hud_height))
    
    # 2. Calculate Layout (Divide screen into 4 equal sections)
    section_w = width / 4
    pad_x = 15
    y_start = height - hud_height + 15
    line_h = 22 # Vertical space between lines
    
    # Fonts
    title_font = get_font(22, bold=True)
    label_font = get_font(15)
    value_font = get_font(16, bold=True)
    
    # --- Helper to draw Label + Value pairs ---
    def draw_stat(col_idx, row_idx, label, value, val_color=COLORS['TEXT_MAIN']):
        x = (col_idx * section_w) + pad_x
        y = y_start + (row_idx * line_h)
        
        # Draw Label
        lbl_surf = label_font.render(label, True, COLORS['TEXT_DIM'])
        surface.blit(lbl_surf, (x, y))
        
        # Draw Value (offset by 85px from column start)
        val_surf = value_font.render(str(value), True, val_color)
        surface.blit(val_surf, (x + 45, y))

    # --- Column 1: General Info ---
    # Title (span top)
    surface.blit(title_font.render("NEAT Sim", True, COLORS['TEXT_MAIN']), (pad_x, y_start - 5))
    
    col = 0
    # Row 1: Generation
    draw_stat(col, 1, "Gen:", generation)
    # Row 2: Time
    draw_stat(col, 2, "Time:", f"{elapsed_time:.1f}s")
    # Row 3: Alive
    alive_color = COLORS['TEXT_ACCENT'] if alive_count > 0 else COLORS['FOOD_BIG']
    draw_stat(col, 3, "Alive:", f"{alive_count}/{total_agents}", alive_color)

    # --- Column 2: Fitness Stats ---
    col = 1
    # Header
    surface.blit(label_font.render("Fitness Metrics", True, COLORS['AGENT_FULL']), (col * section_w + pad_x, y_start))
    # Rows
    draw_stat(col, 1, "Best:", f"{best_fitness:.2f}", COLORS['TEXT_ACCENT'])
    draw_stat(col, 2, "Avg:", f"{avg_fitness:.2f}")
    draw_stat(col, 3, "Global:", f"{global_best_fitness:.2f}", COLORS['FOOD_SMALL'])

    # --- Column 3: Resources ---
    col = 2
    # Header
    surface.blit(label_font.render("Food Collected", True, COLORS['AGENT_FULL']), (col * section_w + pad_x, y_start))
    
    # Small Food Row
    y_sm = y_start + line_h
    x_sm = col * section_w + pad_x
    pygame.draw.circle(surface, COLORS['FOOD_SMALL'], (int(x_sm + 5), int(y_sm + 8)), 4)
    surface.blit(value_font.render(f"Small: {total_small}", True, COLORS['TEXT_DIM']), (x_sm + 20, y_sm))
    
    # Big Food Row
    y_bg = y_start + (line_h * 2)
    pygame.draw.circle(surface, COLORS['FOOD_BIG'], (int(x_sm + 5), int(y_bg + 8)), 5)
    surface.blit(value_font.render(f"Big:   {total_big}", True, COLORS['TEXT_DIM']), (x_sm + 20, y_bg))

    # --- Column 4: Legend / Controls ---
    col = 3
    x_leg = col * section_w + pad_x
    
    # Draw simple colored rectangles for legend
    def draw_legend(y_offset, color, text):
        pygame.draw.rect(surface, color, (x_leg, y_start + y_offset + 4, 8, 8), border_radius=2)
        surface.blit(label_font.render(text, True, COLORS['TEXT_DIM']), (x_leg + 15, y_start + y_offset))

    draw_legend(0, COLORS['AGENT_FULL'], "High Energy")
    draw_legend(20, COLORS['AGENT_LOW'], "Low Energy")
    
    # Quit instruction at bottom
    quit_surf = label_font.render("Press [Q] to Quit", True, (100, 100, 100))
    surface.blit(quit_surf, (x_leg, y_start + 65))

    # --- Draw Vertical Separator Lines ---
    for i in range(1, 4):
        x_pos = i * section_w
        pygame.draw.line(surface, (40, 44, 52), (x_pos, height - hud_height + 10), (x_pos, height - 10), 1)
