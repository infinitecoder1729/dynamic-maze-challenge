# Adaptive Maze Game

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.7+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

**A procedurally-generated maze game featuring dynamic difficulty adjustment, terrain mechanics, and smooth physics-based movement.**

[Features](#features) • [Installation](#installation) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Gameplay](#gameplay)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)

---

## Overview

This game implements an **ML based difficulty system** inspired by Left 4 Dead's dynamic difficulty adjustment. The game analyzes your performance in real-time and adapts the challenge to keep you in an optimal "flow state" - not too easy, not too hard.

### Key Concepts

- **Procedural Generation**: Every maze is unique, generated using algorithms
- **Adaptive Difficulty**: The AI monitors your performance and adjusts complexity
- **Terrain Physics**: Navigate mud (slow), ice (slippery), and standard paths
- **Weighted Pathfinding**: The game calculates optimal solutions considering terrain costs

---

## Features

### ML System

- **Real-time Performance Analysis**
  - Tracks path efficiency (precision)
  - Monitors completion speed
  - Calculates composite performance scores

- **Smooth Difficulty Scaling**
  - Grid size: 10×10 to 45×45
  - Maze complexity (braiding factor)
  - Terrain density adjustment
  - Exponential moving average for smooth transitions

- **Persistent Progress**
  - Saves difficulty state between sessions
  - Tracks historical performance
  - JSON-based save system

### Procedural Generation

- **Recursive Backtracker Algorithm**
  - Generates perfect mazes with guaranteed solutions
  - Longest possible solution paths
  - High complexity and challenge

- **Braiding System**
  - Adds loops and multiple paths
  - Reduces dead ends
  - Adjustable based on difficulty

- **Terrain Generation**
  - **Mud** (Brown): 5× movement cost, 70% speed penalty
  - **Ice** (Light Blue): Auto-sliding mechanic
  - **Paths** (White): Standard movement

### Gameplay Mechanics

- **Smooth Physics**
  - Sub-pixel interpolated movement
  - Terrain-specific behaviors
  - Visual feedback for all actions

- **Ice Sliding**
  - Continuous movement until hitting non-ice
  - Strategic challenge element
  - Stops at walls

- **Cost Tracking**
  - Real-time cost accumulation
  - Comparison to optimal solution
  - Performance metrics calculation

### Visualization

- **Minimap** (for large mazes)
  - Real-time position tracking
  - Goal location indicator
  - Toggleable display

- **Difficulty Bar**
  - Visual difficulty indicator
  - Color gradient (green → red)
  - Smooth animations

- **Comprehensive HUD**
  - Level counter
  - Grid size
  - Current cost vs optimal
  - Time elapsed
  - Control reminders

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download

```bash
# Option 1: If using git
git clone https://github.com/infinitecoder1729/dynamic-maze-challenge
cd dynamic-maze-challenge

# Option 2: If downloaded as ZIP
unzip dynamic-maze-challenge
cd dynamic-maze-challenge
```

### Step 2: Install Dependencies

```bash
pip install pygame>=2.0.0 numpy>=1.19.0
```

### Verify Installation

```bash
python adaptive_maze_game.py
```

If the game window opens, you're ready to play!

---
### First-Time Players

1. **Start Simple**: The game begins at 30% difficulty
2. **Learn Mechanics**: First few levels introduce terrain gradually
3. **Feel the Adaptation**: Complete 3-5 levels to see difficulty adjust
4. **Master the Challenge**: Game adapts to your skill ceiling

### Controls

| Key | Action |
|-----|--------|
| **↑ ↓ ← →** | Move player |
| **M** | Toggle terrain mode (mud/ice on/off) |
| **P** | Pause/Resume game |
| **N** | Toggle minimap display |
| **ESC** | Quit game |

---

## Gameplay

### Objective

Navigate from the **yellow starting position** to the **green goal** as efficiently as possible.

### Scoring System

Your performance is evaluated on two dimensions:

1. **Precision (60% weight)**
   - Measures path efficiency
   - Formula: `optimal_cost / actual_cost`
   - Perfect score = 1.0 (took optimal path)

2. **Speed (40% weight)**
   - Measures completion time
   - Formula: `expected_time / actual_time`
   - Perfect score = 1.0 (completed in expected time)

### Performance Thresholds

| Performance Score | Difficulty Change | Description |
|-------------------|-------------------|-------------|
| 0.8+ | +0.08 | Excelling - Increase challenge |
| 0.6-0.8 | +0.04 | Good - Slight increase |
| 0.5-0.6 | 0.00 | Balanced - Maintain |
| 0.3-0.5 | -0.05 | Below average - Ease challenge |
| < 0.3 | -0.10 | Struggling - Reduce difficulty |

### Terrain Mechanics

#### Mud (Brown)
- **Movement Cost**: 5× standard
- **Visual Speed**: 30% of normal
- **Strategy**: Avoid when possible, use for shortcuts only

#### Ice (Light Blue)
- **Movement Cost**: 1× (same as path)
- **Special Mechanic**: Auto-sliding
- **Behavior**: Cannot stop until reaching non-ice tile
- **Strategy**: Plan ahead, use walls to stop momentum

#### Path (White)
- **Movement Cost**: 1× (baseline)
- **Speed**: Normal
- **Strategy**: Preferred route when available

### Tips for Success

1. **Plan Your Route**
   - Survey the maze before moving
   - Identify mud patches to avoid
   - Use ice strategically for fast traversal

2. **Balance Speed and Precision**
   - Rushing increases cost but saves time
   - Taking optimal path improves precision
   - Find your personal balance point

3. **Understand the AI Director**
   - Consistency is key - avoid extreme variance
   - The director adapts over 5-level windows
   - Intentionally playing poorly won't help long-term

4. **Use Minimap on Large Mazes**
   - Press **N** to toggle
   - Helps maintain orientation
   - Shows goal location

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GAME LOOP                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Input      │→ │   Update     │→ │   Render     │ │
│  │   Handler    │  │   Physics    │  │   System     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────┐
│                  ML                                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Analyze Performance → Adjust Difficulty         │  │
│  │  (Precision + Speed) → (Size + Complexity)       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────┐
│              PROCEDURAL GENERATION                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Maze Gen    │→ │   Braiding   │→ │   Terrain    │ │
│  │     (DFS)    │  │   (Loops)    │  │  (Mud/Ice)   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Classes

#### `FluidDirector`
**Purpose**: Adaptive difficulty management

**Key Methods**:
- `analyze_and_adapt(metrics)`: Updates difficulty based on performance
- `get_level_parameters()`: Returns concrete generation parameters
- `save_progress()` / `load_progress()`: Persistence layer

**Algorithm**:
```python
performance_score = (precision × 0.6) + (speed × 0.4)

if performance_score > threshold:
    difficulty += delta

difficulty_smooth = (difficulty × 0.7) + (target_difficulty × 0.3)
```

#### `AdvancedMazeGen`
**Purpose**: Procedural maze generation

**Algorithm**: Recursive Backtracker (DFS)
```
1. Start with grid of walls
2. Pick random starting cell, mark as path
3. Push to stack
4. While stack not empty:
   - Look at cell on top of stack
   - If has unvisited neighbors:
     - Choose random neighbor
     - Carve path between them
     - Push neighbor to stack
   - Else:
     - Pop from stack (backtrack)
5. Apply braiding (add loops)
6. Apply terrain (mud/ice)
```

**Complexity**: O(rows × cols)

#### `WeightedSolver`
**Purpose**: Calculate optimal path cost

**Algorithm**: Dijkstra's shortest path
```
1. Initialize priority queue with start position
2. While queue not empty:
   - Pop cell with lowest cost
   - If reached goal, return cost
   - For each neighbor:
     - Calculate new_cost = current_cost + terrain_cost
     - If better than known cost:
       - Update cost table
       - Push to priority queue
```

**Complexity**: O((rows × cols) × log(rows × cols))

#### `Player`
**Purpose**: Physics-based player controller

**Key Features**:
- Smooth interpolation between grid positions
- Terrain-specific speed modifiers
- Ice sliding state machine
- Cost accumulation tracking

#### `GameRenderer`
**Purpose**: Optimized rendering with caching

**Optimization**: Pre-renders static maze to surface, only redraws dynamic elements (player, HUD)

**Performance Gain**: ~50-70% FPS improvement

---

## Configuration

### Difficulty Settings

Edit these constants in `adaptive_maze_game.py`:

```python
# In FluidDirector.__init__:
self.current_difficulty = 0.3  # Starting difficulty (0.0-1.0)
                              # 0.3 = Easy start
                              # 0.5 = Medium start
                              # 0.7 = Hard start

# Performance thresholds (in analyze_and_adapt):
if performance_score > 0.8: delta = 0.08  # Increase sensitivity
elif performance_score > 0.6: delta = 0.04
# Adjust these values to change adaptation speed
```

### Terrain Costs

```python
# Physics costs
COST_PATH = 1
COST_MUD = 5    # Change to 3 for easier mud, 7 for harder
COST_ICE = 1    # Keep at 1 (ice is meant to be fast)

# In Player.update():
if current_tile == 2:  # Mud
    speed *= 0.3  # Change to 0.5 for less penalty, 0.2 for more
```

### Visual Settings

```python
# Window configuration
WINDOW_WIDTH = 1000   # Default 1000
WINDOW_HEIGHT = 800   # Default 800
FPS = 60              # Target frame rate

# Colors (RGB tuples)
COLOR_PLAYER = (255, 255, 0)  # Yellow - change to your preference
COLOR_GOAL = (0, 255, 0)      # Green
COLOR_MUD = (101, 67, 33)     # Brown
COLOR_ICE = (135, 206, 235)   # Light Blue
```

### Grid Size Limits

```python
# In FluidDirector.get_level_parameters():
size_val = 10 + (diff * 35)  # Min 10, Max 45
# Change formula for different ranges:
# size_val = 15 + (diff * 25)  # Min 15, Max 40
```

---

## Development

### Code Organization

```python
# === CONFIGURATION ===
# Constants, colors, enums

# === AI DIRECTOR ===
class FluidDirector

# === PROCEDURAL GENERATION ===
class AdvancedMazeGen

# === OPTIMAL PATH SOLVER ===
class WeightedSolver

# === PLAYER CONTROLLER ===
class Player

# === RENDERING SYSTEM ===
class GameRenderer
+ Helper functions

# === MAIN GAME LOOP ===
def main()
```

### Adding New Features

#### Example: Add New Terrain Type (Lava)

```python
# 1. Add color constant
COLOR_LAVA = (255, 69, 0)  # Orange-red

# 2. Add cost constant
COST_LAVA = 10  # Very expensive

# 3. Modify AdvancedMazeGen._apply_terrain()
lava_mask = mask & (random_values >= density) & (random_values < density * 1.2)
self.grid[lava_mask] = 4  # Lava tile type

# 4. Update rendering in GameRenderer.render_maze()
elif val == 4:
    pygame.draw.rect(self.maze_surface, COLOR_LAVA, rect)

# 5. Update WeightedSolver.get_optimal_cost()
if tile_type == 4: move_cost = COST_LAVA

# 6. Update Player movement (optional special behavior)
if current_tile == 4:  # Lava
    speed *= 0.1  # Very slow
    # Could add damage over time, etc.
```

### Testing

```bash
# Test with different starting difficulties
python adaptive_maze_game.py  # Default (0.3)

# Modify in code temporarily:
director = FluidDirector(starting_difficulty=0.7)  # Hard start
```

### Performance Profiling

```python
# Add at top of file:
import cProfile
import pstats

# Replace main() call:
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

---

## Algorithm Details

### Recursive Backtracker (Maze Generation)

**Why This Algorithm?**
- Generates "perfect" mazes (one solution path between any two points)
- Creates long, winding corridors (high challenge)
- Efficient: O(n) time complexity
- Simple to implement

**Visual Walkthrough**:
```
Step 1: All walls       Step 2: Start point    Step 3: Carve paths
███████████            ███████████            ███████████
███████████            █   ███████            █   █   ███
███████████      →     ███████████      →     █████   ███
███████████            ███████████            █████   ███
███████████            ███████████            █████    ██

Step 4: Backtrack      Step 5: Complete
███████████            ███████████
█   █   ███            █   █   █ █
█   █   ███     →      █   █   █ █
█   █   ███            █   █   █ █
█████   █ █            █████   █ █
```

### Braiding (Loop Addition)

**Purpose**: Reduce difficulty by adding alternative paths

**Method**:
1. Scan grid for dead-end walls
2. Check if removal connects two paths
3. Remove with probability = `braid_chance`

**Effect**:
- `braid_chance = 0.0`: Pure maze (hardest)
- `braid_chance = 0.6`: Multiple paths (easier)

### Dijkstra's Algorithm (Optimal Path)

**Why Not A*?**
- We need **exact cost** to all reachable cells
- Terrain weights are not uniform (can't use heuristic effectively)
- Dijkstra guarantees optimal solution

**Pseudocode**:
```
function dijkstra(grid, start, goal):
    costs = {start: 0}
    pq = [(0, start)]

    while pq not empty:
        current_cost, current_pos = pq.pop_min()

        if current_pos == goal:
            return current_cost

        for neighbor in get_neighbors(current_pos):
            new_cost = current_cost + terrain_cost(neighbor)

            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                pq.push((new_cost, neighbor))

    return INFINITY  # No path
```

---

## Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions/classes
- Comment complex algorithms
- Keep functions under 50 lines when possible

---

<div align="center">

**Made with ❤️ by @infinitecoder1729**

**Note on AI Assistance > This project utilized Large Language Models (LLMs) to refine code documentation, standardize variable naming conventions, and assist with code formatting and beautification.**

</div>
