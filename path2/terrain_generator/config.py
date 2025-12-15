"""
Configuration constants for terrain generation
"""

# Map dimensions
MAP_SIZE = 1000
MAP_HALF_SIZE = MAP_SIZE / 2
MAX_ALT = 150.0
MAX_TREE_HEIGHT = 110
MOUNTAIN_BUILDING_CEILING = 60

# Terrain parameters
PARK_RADIUS = 30
TRANSITION_ZONE = 40

# Lakes definition
LAKES = [
    {"x": 10, "z": 5, "rx": 20, "rz": 18, "rot": 0},
    {"x": 250, "z": -250, "rx": 65, "rz": 60, "rot": 0},
    {"x": -300, "z": 120, "rx": 25, "rz": 100, "rot": 0.7853981633974483}  # π/4
]

# Building generation parameters
CITY_BUILDING_COUNT = 1300
CITY_BUILDING_MAX_ATTEMPTS = 20000
MOUNTAIN_BUILDING_COUNT = 200
MOUNTAIN_BUILDING_MAX_ATTEMPTS = 10000

# Tree generation parameters
TREE_COUNT = 4000
TREE_MAX_ATTEMPTS = 30000

# User generation parameters
CITY_USER_COUNT = 4
MOUNTAIN_USER_COUNT = 2

# Terrain sampling for export
TERRAIN_SAMPLE_STEP = 10  # Sample every 10 meters

# Output file
OUTPUT_FILE = "data/terrain_data.json"

