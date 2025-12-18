"""
Adaptive Maze Game with AI Director
====================================
A procedurally-generated maze game featuring dynamic difficulty adjustment,
terrain mechanics, and smooth physics-based movement.

Author: Advanced Game Systems
Version: 2.0
Date: December 2025
"""

import pygame
import random
import time
import heapq
import numpy as np
import json
from collections import deque
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
FPS = 60

# Colors
COLOR_BG = (15, 15, 20)
COLOR_WALL = (40, 40, 50)
COLOR_PATH = (220, 220, 220)
COLOR_MUD = (101, 67, 33)      # Brown
COLOR_ICE = (135, 206, 235)    # Light Blue
COLOR_PLAYER = (255, 255, 0)   # Yellow
COLOR_GOAL = (0, 255, 0)       # Green
COLOR_HUD = (255, 255, 255)    # White
COLOR_DIFFICULTY_LOW = (0, 200, 0)
COLOR_DIFFICULTY_HIGH = (200, 0, 0)

# Physics Costs
COST_PATH = 1
COST_MUD = 5    # Mud is 5x harder to walk through
COST_ICE = 1    # Ice is fast but slippery

# Game States
class GameState(Enum):
    PLAYING = 1
    PAUSED = 2
    LEVEL_COMPLETE = 3

# =============================================================================
# AI DIRECTOR - DYNAMIC DIFFICULTY SYSTEM
# =============================================================================

class FluidDirector:
    """
    Advanced AI Director System
    ----------------------------
    Uses performance metrics to smoothly adjust difficulty:
    - Precision: How efficiently did player navigate?
    - Speed: How quickly did player complete the level?

    Outputs concrete level parameters (grid size, complexity, terrain density).
    """

    def __init__(self, starting_difficulty=0.3):
        self.current_difficulty = starting_difficulty  # 0.0 to 1.0
        self.target_difficulty = starting_difficulty

        # Performance history for smoothing
        self.history_precision = deque(maxlen=5)
        self.history_speed = deque(maxlen=5)

        # Tracking
        self.levels_completed = 0
        self.total_performance = 0.0

    def analyze_and_adapt(self, metrics):
        """
        Analyzes player performance and adjusts difficulty.

        Args:
            metrics (dict): Contains 'precision' and 'time_efficiency' (0.0-1.0)
        """
        avg_precision = metrics['precision']
        avg_speed_ratio = metrics['time_efficiency']

        # Store history
        self.history_precision.append(avg_precision)
        self.history_speed.append(avg_speed_ratio)

        # Composite performance score (weighted)
        performance_score = (avg_precision * 0.6) + (avg_speed_ratio * 0.4)

        self.levels_completed += 1
        self.total_performance += performance_score

        print(f"\n[AI Director] Level {self.levels_completed} Analysis:")
        print(f"  - Precision: {avg_precision:.2f} (Path efficiency)")
        print(f"  - Speed: {avg_speed_ratio:.2f} (Time efficiency)")
        print(f"  - Overall Performance: {performance_score:.2f}")

        # Determine difficulty adjustment
        delta = 0.0
        if performance_score > 0.8:
            delta = 0.08
            print("  → Player excelling! Increasing challenge.")
        elif performance_score > 0.6:
            delta = 0.04
            print("  → Good performance. Slight increase.")
        elif performance_score < 0.3:
            delta = -0.10
            print("  → Struggling detected. Reducing difficulty.")
        elif performance_score < 0.5:
            delta = -0.05
            print("  → Below average. Easing challenge.")
        else:
            print("  → Balanced performance. Maintaining difficulty.")

        # Apply smooth transition (exponential moving average)
        self.target_difficulty = self.current_difficulty + delta
        self.current_difficulty = (self.current_difficulty * 0.7 + 
                                   self.target_difficulty * 0.3)

        # Clamp to valid range
        self.current_difficulty = max(0.1, min(0.95, self.current_difficulty))

        print(f"  → New Difficulty: {self.current_difficulty:.2f}")

    def get_level_parameters(self):
        """
        Maps abstract difficulty (0-1) to concrete generation parameters.

        Returns:
            tuple: (rows, cols, braid_chance, terrain_density)
        """
        diff = self.current_difficulty

        # 1. Grid Size (10x10 to 45x45)
        size_val = 10 + (diff * 35)
        rows = int(size_val)
        cols = int(size_val)

        # Ensure odd dimensions (required for maze algorithm)
        if rows % 2 == 0: rows += 1
        if cols % 2 == 0: cols += 1

        # 2. Maze Complexity (Braiding)
        # Low braid = More dead ends (harder)
        # High braid = More loops (easier)
        braid_chance = max(0.0, 0.6 - (diff * 0.6))

        # 3. Terrain Density
        # More terrain obstacles = Harder
        terrain_noise = 0.1 + (diff * 0.3)

        return rows, cols, braid_chance, terrain_noise

    def get_average_performance(self):
        """Returns average performance across all levels."""
        if self.levels_completed == 0:
            return 0.0
        return self.total_performance / self.levels_completed

    def save_progress(self, filename="game_progress.json"):
        """Save current state to file."""
        data = {
            "difficulty": self.current_difficulty,
            "levels_completed": self.levels_completed,
            "total_performance": self.total_performance,
            "history_precision": list(self.history_precision),
            "history_speed": list(self.history_speed)
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Progress saved to {filename}")
        except Exception as e:
            print(f"Could not save progress: {e}")

    def load_progress(self, filename="game_progress.json"):
        """Load previous state from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.current_difficulty = data.get("difficulty", 0.3)
                self.levels_completed = data.get("levels_completed", 0)
                self.total_performance = data.get("total_performance", 0.0)
                self.history_precision = deque(data.get("history_precision", []), maxlen=5)
                self.history_speed = deque(data.get("history_speed", []), maxlen=5)
            print(f"Progress loaded from {filename}")
            return True
        except FileNotFoundError:
            print("No previous progress found. Starting fresh.")
            return False
        except Exception as e:
            print(f"Could not load progress: {e}")
            return False

# =============================================================================
# PROCEDURAL MAZE GENERATION
# =============================================================================

class AdvancedMazeGen:
    """
    Procedural Maze Generator
    -------------------------
    Uses Recursive Backtracker (DFS) algorithm with braiding and terrain.

    Tile Types:
        0 = Path
        1 = Wall
        2 = Mud (slow, expensive)
        3 = Ice (fast, slippery)
    """

    def __init__(self, rows, cols, use_terrain=False):
        self.rows = rows
        self.cols = cols
        self.use_terrain = use_terrain

        # Initialize grid (all walls)
        self.grid = np.ones((rows, cols), dtype=int)

        # Define start and end positions
        self.start = (1, 1)
        self.end = (rows - 2, cols - 2)

    def generate(self, braid_chance, terrain_density):
        """
        Generate the complete maze.

        Args:
            braid_chance (float): Probability of adding loops (0-1)
            terrain_density (float): Amount of terrain obstacles (0-1)

        Returns:
            np.ndarray: Generated grid
        """
        # Step 1: Create perfect maze with Recursive Backtracker
        self._recursive_backtracker()

        # Step 2: Add loops (braiding)
        self._apply_braiding(braid_chance)

        # Step 3: Add terrain (mud/ice)
        if self.use_terrain:
            self._apply_terrain(terrain_density)

        # Ensure start and end are clear
        self.grid[self.start] = 0
        self.grid[self.end] = 0

        return self.grid

    def _recursive_backtracker(self):
        """DFS-based perfect maze generation."""
        stack = [self.start]
        self.grid[self.start] = 0

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        while stack:
            r, c = stack[-1]

            # Find unvisited neighbors (2 cells away)
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 < nr < self.rows - 1 and 
                    0 < nc < self.cols - 1 and 
                    self.grid[nr, nc] == 1):
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                # Choose random neighbor
                random.shuffle(neighbors)
                nr, nc, dr, dc = neighbors[0]

                # Carve path between current and neighbor
                self.grid[r + dr // 2, c + dc // 2] = 0
                self.grid[nr, nc] = 0

                stack.append((nr, nc))
            else:
                # Backtrack
                stack.pop()

    def _apply_braiding(self, chance):
        """
        Remove dead ends to create loops.
        Higher chance = More loops (easier navigation).
        """
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.grid[r, c] == 1:  # If wall
                    # Check if removing creates useful connection
                    if (self.grid[r + 1, c] != 1 and self.grid[r - 1, c] != 1):
                        if random.random() < chance:
                            self.grid[r, c] = 0
                    elif (self.grid[r, c + 1] != 1 and self.grid[r, c - 1] != 1):
                        if random.random() < chance:
                            self.grid[r, c] = 0

    def _apply_terrain(self, density):
        """
        Add mud and ice tiles using vectorized operations.
        """
        # Create mask for pathable tiles only
        mask = (self.grid == 0)

        # Generate random values for entire grid
        random_values = np.random.random((self.rows, self.cols))

        # Apply terrain based on density thresholds
        mud_mask = mask & (random_values < density / 2)
        ice_mask = mask & (random_values >= density / 2) & (random_values < density)

        self.grid[mud_mask] = 2  # Mud
        self.grid[ice_mask] = 3  # Ice

# =============================================================================
# OPTIMAL PATH SOLVER
# =============================================================================

class WeightedSolver:
    """
    Dijkstra's Algorithm for weighted pathfinding.
    Calculates optimal cost considering terrain weights.
    """

    @staticmethod
    def get_optimal_cost(grid, start, end):
        """
        Find minimum cost path from start to end.

        Args:
            grid (np.ndarray): Maze grid with terrain
            start (tuple): Starting position (r, c)
            end (tuple): Goal position (r, c)

        Returns:
            int: Minimum cost to reach goal
        """
        rows, cols = grid.shape

        # Priority queue: (cumulative_cost, position)
        pq = [(0, start)]
        costs = {start: 0}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while pq:
            current_cost, (r, c) = heapq.heappop(pq)

            # Reached goal
            if (r, c) == end:
                return current_cost

            # Skip if we've found better path
            if current_cost > costs.get((r, c), float('inf')):
                continue

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                # Check bounds and walls
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if grid[nr, nc] == 1:  # Wall
                    continue

                # Determine movement cost
                tile_type = grid[nr, nc]
                if tile_type == 2:
                    move_cost = COST_MUD
                elif tile_type == 3:
                    move_cost = COST_ICE
                else:
                    move_cost = COST_PATH

                new_cost = current_cost + move_cost

                # Update if better path found
                if new_cost < costs.get((nr, nc), float('inf')):
                    costs[(nr, nc)] = new_cost
                    heapq.heappush(pq, (new_cost, (nr, nc)))

        return 9999  # No path exists

# =============================================================================
# PLAYER CONTROLLER
# =============================================================================

class Player:
    """
    Player with smooth interpolated movement and terrain physics.
    """

    def __init__(self, pos, grid, tile_size):
        # Position tracking
        self.grid_pos = list(pos)  # Logical position
        self.pixel_pos = [pos[1] * tile_size, pos[0] * tile_size]  # Visual position

        # References
        self.grid = grid
        self.tile_size = tile_size

        # Movement state
        self.target_grid_pos = list(pos)
        self.is_moving = False
        self.slide_direction = None  # For ice sliding

        # Movement parameters
        self.base_move_speed = 0.2  # Base interpolation speed

        # Statistics
        self.steps_taken = 0
        self.cost_incurred = 0
        self.move_history = []  # For replay

    def update(self, dt):
        """
        Update player position (smooth interpolation).

        Args:
            dt (float): Delta time in milliseconds
        """
        # Target pixel position
        tx = self.target_grid_pos[1] * self.tile_size
        ty = self.target_grid_pos[0] * self.tile_size

        # Calculate distance
        dx = tx - self.pixel_pos[0]
        dy = ty - self.pixel_pos[1]
        dist = (dx**2 + dy**2)**0.5

        # Arrived at target
        if dist < 2:
            self.pixel_pos = [tx, ty]
            self.grid_pos = list(self.target_grid_pos)
            self.is_moving = False

            # ICE MECHANIC: Continue sliding
            current_tile = self.grid[self.grid_pos[0], self.grid_pos[1]]
            if current_tile == 3 and self.slide_direction:
                self.attempt_move(self.slide_direction)
            else:
                self.slide_direction = None
        else:
            # Move towards target
            speed = self.base_move_speed * dt * self.tile_size * 0.05

            # MUD SLOWDOWN
            current_tile = self.grid[self.grid_pos[0], self.grid_pos[1]]
            if current_tile == 2:  # Mud
                speed *= 0.3

            # Apply movement
            norm = [dx / dist, dy / dist]
            self.pixel_pos[0] += norm[0] * speed
            self.pixel_pos[1] += norm[1] * speed

    def attempt_move(self, direction):
        """
        Attempt to move in given direction.

        Args:
            direction (tuple): (dr, dc) movement vector
        """
        if self.is_moving:
            return

        nr = self.grid_pos[0] + direction[0]
        nc = self.grid_pos[1] + direction[1]

        # Boundary check
        if not (0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1]):
            return

        # Wall check
        if self.grid[nr, nc] == 1:
            self.slide_direction = None  # Stop sliding on wall
            return

        # Valid move
        self.target_grid_pos = [nr, nc]
        self.is_moving = True
        self.steps_taken += 1

        # Record cost
        tile_type = self.grid[nr, nc]
        if tile_type == 2:
            self.cost_incurred += COST_MUD
        elif tile_type == 3:
            self.cost_incurred += COST_ICE
        else:
            self.cost_incurred += COST_PATH

        # ICE: Remember direction for sliding
        if tile_type == 3:
            self.slide_direction = direction

        # Record move for history
        self.move_history.append({"dir": direction, "time": time.time()})

# =============================================================================
# RENDERING SYSTEM
# =============================================================================

class GameRenderer:
    """Optimized rendering with caching."""

    def __init__(self):
        self.maze_surface = None
        self.maze_dirty = True

    def render_maze(self, grid, tile_size):
        """
        Render maze to surface (with caching).
        Only re-renders when maze_dirty flag is set.
        """
        if self.maze_dirty:
            rows, cols = grid.shape
            self.maze_surface = pygame.Surface((cols * tile_size, rows * tile_size))
            self.maze_surface.fill(COLOR_PATH)

            # Draw all tiles
            for r in range(rows):
                for c in range(cols):
                    rect = (c * tile_size, r * tile_size, tile_size + 1, tile_size + 1)
                    val = grid[r, c]

                    if val == 1:
                        pygame.draw.rect(self.maze_surface, COLOR_WALL, rect)
                    elif val == 2:
                        pygame.draw.rect(self.maze_surface, COLOR_MUD, rect)
                    elif val == 3:
                        pygame.draw.rect(self.maze_surface, COLOR_ICE, rect)

            self.maze_dirty = False

        return self.maze_surface

    def reset(self):
        """Mark maze for re-rendering."""
        self.maze_dirty = True

def lerp_color(color1, color2, t):
    """Linear interpolation between two colors."""
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

def draw_difficulty_bar(screen, director, font):
    """Draw visual difficulty indicator."""
    bar_x = 10
    bar_y = WINDOW_HEIGHT - 50
    bar_width = 300
    bar_height = 25

    # Background
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))

    # Filled portion
    fill_width = int(bar_width * director.current_difficulty)
    color = lerp_color(COLOR_DIFFICULTY_LOW, COLOR_DIFFICULTY_HIGH, director.current_difficulty)
    pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))

    # Border
    pygame.draw.rect(screen, COLOR_HUD, (bar_x, bar_y, bar_width, bar_height), 2)

    # Label
    label = font.render(f"Difficulty: {director.current_difficulty:.2f}", True, COLOR_HUD)
    screen.blit(label, (bar_x, bar_y - 25))

def draw_minimap(screen, grid, player_pos, goal_pos, scale=4):
    """Draw minimap in corner."""
    rows, cols = grid.shape

    # Create minimap surface
    minimap_width = cols // scale
    minimap_height = rows // scale
    minimap = pygame.Surface((minimap_width, minimap_height))
    minimap.fill(COLOR_BG)

    # Draw simplified maze
    for r in range(0, rows, scale):
        for c in range(0, cols, scale):
            if grid[r, c] == 1:  # Wall
                minimap.set_at((c // scale, r // scale), COLOR_WALL)
            else:
                minimap.set_at((c // scale, r // scale), COLOR_PATH)

    # Draw goal
    minimap.set_at((goal_pos[1] // scale, goal_pos[0] // scale), COLOR_GOAL)

    # Draw player (larger dot)
    px, py = player_pos[1] // scale, player_pos[0] // scale
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if 0 <= px + dc < minimap_width and 0 <= py + dr < minimap_height:
                minimap.set_at((px + dc, py + dr), COLOR_PLAYER)

    # Scale up for visibility
    display_size = 150
    minimap_scaled = pygame.transform.scale(minimap, (display_size, display_size))

    # Draw to screen
    screen.blit(minimap_scaled, (WINDOW_WIDTH - display_size - 10, 10))

# =============================================================================
# MAIN GAME LOOP
# =============================================================================

def main():
    """Main game function."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Adaptive Maze Game - AI Director System")

    # Fonts
    font = pygame.font.SysFont("Verdana", 16)
    big_font = pygame.font.SysFont("Verdana", 40)
    small_font = pygame.font.SysFont("Verdana", 12)

    clock = pygame.time.Clock()

    # Initialize systems
    director = FluidDirector(starting_difficulty=0.3)
    renderer = GameRenderer()

    # Try to load previous progress
    director.load_progress()

    # Game settings
    mode_terrain = True
    show_minimap = True
    game_state = GameState.PLAYING

    running = True

    print("\n" + "="*60)
    print("ADAPTIVE MAZE GAME - AI DIRECTOR SYSTEM")
    print("="*60)
    print("Controls:")
    print("  Arrow Keys - Move")
    print("  M - Toggle Terrain Mode")
    print("  P - Pause/Resume")
    print("  N - Show Minimap")
    print("  ESC - Quit")
    print("="*60 + "\n")

    while running:
        # === LEVEL GENERATION ===
        rows, cols, braid, terrain_dens = director.get_level_parameters()

        # Adjust tile size to fit screen
        tile_size = min(WINDOW_WIDTH // cols, (WINDOW_HEIGHT - 100) // rows)

        # Generate maze
        gen = AdvancedMazeGen(rows, cols, use_terrain=mode_terrain)
        grid = gen.generate(braid, terrain_dens)

        # Calculate optimal solution
        optimal_cost = WeightedSolver.get_optimal_cost(grid, gen.start, gen.end)

        # Initialize player
        player = Player(gen.start, grid, tile_size)

        # Reset renderer
        renderer.reset()

        # Level timing
        start_time = time.time()
        level_active = True

        # === LEVEL LOOP ===
        while level_active and running:
            dt = clock.tick(FPS)

            # === INPUT HANDLING ===
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    level_active = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        level_active = False

                    elif event.key == pygame.K_m:
                        mode_terrain = not mode_terrain
                        print(f"Terrain Mode: {'ON' if mode_terrain else 'OFF'}")
                        level_active = False  # Regenerate level

                    elif event.key == pygame.K_p:
                        if game_state == GameState.PLAYING:
                            game_state = GameState.PAUSED
                            print("PAUSED")
                        elif game_state == GameState.PAUSED:
                            game_state = GameState.PLAYING
                            print("RESUMED")

                    elif event.key == pygame.K_n:
                        show_minimap = not show_minimap

            # Skip updates if paused
            if game_state == GameState.PAUSED:
                # Draw pause screen
                screen.fill(COLOR_BG)
                pause_text = big_font.render("PAUSED", True, COLOR_HUD)
                text_rect = pause_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
                screen.blit(pause_text, text_rect)

                hint_text = font.render("Press P to resume", True, COLOR_HUD)
                hint_rect = hint_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
                screen.blit(hint_text, hint_rect)

                pygame.display.flip()
                continue

            # Movement input
            keys = pygame.key.get_pressed()
            move = None
            if keys[pygame.K_UP]:
                move = (-1, 0)
            elif keys[pygame.K_DOWN]:
                move = (1, 0)
            elif keys[pygame.K_LEFT]:
                move = (0, -1)
            elif keys[pygame.K_RIGHT]:
                move = (0, 1)

            if move:
                player.attempt_move(move)

            # === UPDATE ===
            player.update(dt)

            # === WIN CONDITION ===
            if tuple(player.grid_pos) == gen.end:
                level_active = False

                # Calculate metrics
                time_taken = time.time() - start_time
                actual_cost = player.cost_incurred

                # Precision: How close to optimal path?
                precision = min(1.0, optimal_cost / max(1, actual_cost))

                # Speed: Expected time vs actual time
                expected_time = optimal_cost * 0.2  # 0.2s per optimal step
                speed_ratio = min(1.0, expected_time / max(0.1, time_taken))

                metrics = {
                    'precision': precision,
                    'time_efficiency': speed_ratio
                }

                # Update director
                director.analyze_and_adapt(metrics)

                # Show completion
                print(f"\nLevel Complete!")
                print(f"  Time: {time_taken:.1f}s")
                print(f"  Steps: {player.steps_taken}")
                print(f"  Cost: {actual_cost} (Optimal: {optimal_cost})")

            # === RENDERING ===
            screen.fill(COLOR_BG)

            # Calculate centering offset
            maze_width = cols * tile_size
            maze_height = rows * tile_size
            off_x = (WINDOW_WIDTH - maze_width) // 2
            off_y = (WINDOW_HEIGHT - maze_height - 60) // 2

            # Draw maze (cached)
            maze_surf = renderer.render_maze(grid, tile_size)

            # Draw goal on maze surface
            goal_rect = (gen.end[1] * tile_size, gen.end[0] * tile_size, 
                        tile_size, tile_size)
            pygame.draw.rect(maze_surf, COLOR_GOAL, goal_rect)

            # Draw player
            player_rect = (player.pixel_pos[0] + 2, player.pixel_pos[1] + 2,
                          tile_size - 4, tile_size - 4)
            pygame.draw.rect(maze_surf, COLOR_PLAYER, player_rect)

            # Blit maze to screen
            screen.blit(maze_surf, (off_x, off_y))

            # === HUD ===
            hud_lines = [
                f"Level: {director.levels_completed + 1}",
                f"Grid: {rows}x{cols}",
                f"Terrain: {'ON' if mode_terrain else 'OFF'}",
                f"Cost: {player.cost_incurred} / Optimal: {optimal_cost}",
                f"Time: {time.time() - start_time:.1f}s"
            ]

            for i, line in enumerate(hud_lines):
                text = font.render(line, True, COLOR_HUD)
                screen.blit(text, (10, 10 + i * 22))

            # Controls reminder
            controls = [
                "Controls: Arrows=Move | M=Terrain | P=Pause | N=Minimap | ESC=Quit"
            ]
            for i, line in enumerate(controls):
                text = small_font.render(line, True, (150, 150, 150))
                screen.blit(text, (10, WINDOW_HEIGHT - 80 + i * 15))

            # Difficulty bar
            draw_difficulty_bar(screen, director, font)

            # Minimap
            if show_minimap and rows > 15:  # Only for larger mazes
                draw_minimap(screen, grid, player.grid_pos, gen.end)

            pygame.display.flip()

    # Save progress before exit
    director.save_progress()

    # Final statistics
    print("\n" + "="*60)
    print("GAME SESSION COMPLETE")
    print("="*60)
    print(f"Levels Completed: {director.levels_completed}")
    print(f"Final Difficulty: {director.current_difficulty:.2f}")
    print(f"Average Performance: {director.get_average_performance():.2f}")
    print("="*60 + "\n")

    pygame.quit()

if __name__ == "__main__":
    main()
