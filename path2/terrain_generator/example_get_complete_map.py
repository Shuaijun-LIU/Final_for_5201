#!/usr/bin/env python3
"""
Example: How to get complete terrain map from JSON file

This script demonstrates that you CAN get the complete terrain map
(including mountain structures) from the JSON file.
"""

import json
import sys
import os

# Add current directory to path to import terrain_utils
sys.path.insert(0, os.path.dirname(__file__))

from terrain_utils import TerrainGenerator, Mulberry32Random
from opensimplex import OpenSimplex


def load_complete_terrain(json_path='data/terrain_data.json'):
    """
    Load complete terrain from JSON file.
    
    Returns:
        dict: Contains terrain generator and all data from JSON
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        world_state = json.load(f)
    
    # Initialize terrain generator using seed from JSON
    rng = Mulberry32Random(int(world_state['seed']))
    simplex = OpenSimplex(int(world_state['seed']))
    terrain_gen = TerrainGenerator(simplex, world_state['lakes'], rng)
    
    return {
        'terrain_gen': terrain_gen,
        'data': world_state
    }


def get_terrain_at_point(terrain_gen, x, z):
    """
    Get terrain height at any point (x, z).
    This gives you the complete terrain - you can query any point!
    """
    return terrain_gen.get_terrain_height(x, z)


def check_mountain_structure(terrain_gen, x, z, city_limit=200):
    """
    Check if a point is in mountain area.
    Mountain = high terrain outside city boundaries.
    """
    dist = (x**2 + z**2)**0.5
    height = terrain_gen.get_terrain_height(x, z)
    
    is_mountain = (dist > city_limit) and (height > 30)
    return {
        'x': x,
        'z': z,
        'height': height,
        'is_mountain': is_mountain,
        'distance_from_center': dist
    }


def example_usage():
    """Example: Get complete terrain map from JSON"""
    
    print("=" * 60)
    print("Example: Getting Complete Terrain Map from JSON")
    print("=" * 60)
    print()
    
    # Load complete terrain
    print("1. Loading terrain from JSON file...")
    terrain_data = load_complete_terrain()
    terrain_gen = terrain_data['terrain_gen']
    data = terrain_data['data']
    
    print(f"   ✓ Loaded seed: {data['seed']}")
    print(f"   ✓ Map size: {data['mapSize']}m x {data['mapSize']}m")
    print(f"   ✓ Mountain regions: {len(data.get('mountainRegions', []))}")
    print()
    
    # Example 1: Get height at specific points
    print("2. Getting terrain height at specific points:")
    test_points = [
        (0, 0),      # Center
        (100, 100),  # City area
        (300, 300),  # Mountain area
        (-400, -400) # Far mountain area
    ]
    
    for x, z in test_points:
        height = get_terrain_at_point(terrain_gen, x, z)
        info = check_mountain_structure(terrain_gen, x, z)
        print(f"   Point ({x:6.1f}, {z:6.1f}): Height = {height:6.2f}m, "
              f"Mountain = {info['is_mountain']}")
    print()
    
    # Example 2: Use mountain regions from JSON
    print("3. Using mountain regions from JSON:")
    if data.get('mountainRegions'):
        for i, region in enumerate(data['mountainRegions'][:3]):  # Show first 3
            print(f"   Region {i+1}:")
            print(f"     Bounds: X[{region['minX']:.1f}, {region['maxX']:.1f}], "
                  f"Z[{region['minZ']:.1f}, {region['maxZ']:.1f}]")
            print(f"     Height: {region['minHeight']:.1f}m - {region['maxHeight']:.1f}m "
                  f"(avg: {region['avgHeight']:.1f}m)")
    print()
    
    # Example 3: Generate full terrain grid (small sample)
    print("4. Generating terrain grid sample (10m resolution, 100x100 area):")
    sample_size = 100
    resolution = 10
    grid = []
    
    for x in range(-sample_size, sample_size + 1, resolution):
        for z in range(-sample_size, sample_size + 1, resolution):
            height = get_terrain_at_point(terrain_gen, x, z)
            grid.append((x, z, height))
    
    print(f"   ✓ Generated {len(grid)} points")
    print(f"   ✓ Height range: {min(h for _, _, h in grid):.2f}m - "
          f"{max(h for _, _, h in grid):.2f}m")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    print("✓ YES, you CAN get the complete terrain map from JSON!")
    print("✓ The JSON file contains:")
    print("  - Seed: Allows exact terrain regeneration")
    print("  - Mountain regions: Pre-analyzed areas")
    print("  - Sample points: Quick reference (terrainMap)")
    print("  - All buildings, trees, users")
    print()
    print("✓ To get complete terrain:")
    print("  - Use getTerrainHeight(x, z) with seed from JSON")
    print("  - This gives you EXACT height at ANY point")
    print("  - Mountain structures = high terrain values")
    print()
    print("✓ You can:")
    print("  - Query any point on-demand")
    print("  - Pre-generate full grid at any resolution")
    print("  - Use mountainRegions for quick identification")
    print("=" * 60)


if __name__ == '__main__':
    try:
        example_usage()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python3 generate_terrain.py' first to generate terrain data.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

