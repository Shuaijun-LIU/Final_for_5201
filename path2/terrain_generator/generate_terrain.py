#!/usr/bin/env python3
"""
Terrain data generation script
Generates terrain, buildings, trees, and users, then saves to JSON
"""

import json
import math
import os
import sys
import time
from opensimplex import OpenSimplex
from config import *
from terrain_utils import (
    Mulberry32Random,
    TerrainGenerator,
    BuildingGenerator,
    TreeGenerator,
    UserGenerator
)


def generate_terrain_map(terrain_gen):
    """Generate terrain height map for export (legacy format, 10m resolution)"""
    terrain_map = []
    half = int(MAP_HALF_SIZE)
    step = int(TERRAIN_SAMPLE_STEP)
    
    print("Generating terrain height map (legacy format, 10m resolution)...")
    for x in range(-half, half + 1, step):
        for z in range(-half, half + 1, step):
            h = terrain_gen.get_terrain_height(x, z)
            if h > 2.0:  # Only record significant heights
                terrain_map.append({
                    "x": round(x),
                    "z": round(z),
                    "y": round(h, 1)
                })
    
    return terrain_map


def generate_height_map(terrain_gen, resolution=2.0):
    """
    Generate complete height map for trajectory planning
    Uses 2D array format for compact storage and fast access
    
    Returns:
        dict with:
            - heightMap: 2D array of heights (row-major, [z][x])
            - origin: [x_min, z_min]
            - step: resolution in meters
            - size: [width, depth] in grid cells
    """
    half = int(MAP_HALF_SIZE)
    step = float(resolution)
    
    # Calculate grid dimensions
    x_min = -half
    z_min = -half
    x_max = half
    z_max = half
    
    # Generate grid coordinates
    x_coords = []
    x = x_min
    while x <= x_max:
        x_coords.append(round(x, 1))
        x += step
    
    z_coords = []
    z = z_min
    while z <= z_max:
        z_coords.append(round(z, 1))
        z += step
    
    width = len(x_coords)
    depth = len(z_coords)
    
    print(f"Generating complete height map ({resolution}m resolution, {width}x{depth} = {width*depth} points)...")
    
    # Generate height map (row-major: heightMap[z_index][x_index])
    height_map = []
    total_points = width * depth
    progress_step = max(1, total_points // 20)  # Show progress every 5%
    
    for zi, z in enumerate(z_coords):
        row = []
        for xi, x in enumerate(x_coords):
            h = terrain_gen.get_terrain_height(x, z)
            row.append(round(h, 2))  # 2 decimal places for precision
            
            # Show progress
            idx = zi * width + xi
            if idx % progress_step == 0:
                progress = (idx + 1) / total_points * 100
                print(f"  Progress: {progress:.1f}% ({idx+1}/{total_points})", end='\r')
        
        height_map.append(row)
    
    print(f"  Progress: 100.0% ({total_points}/{total_points})")
    
    return {
        "heightMap": height_map,
        "origin": [x_min, z_min],
        "step": step,
        "size": [width, depth],
        "xCoords": x_coords,  # For easy lookup
        "zCoords": z_coords   # For easy lookup
    }


def generate_mountain_regions(terrain_gen):
    """
    Generate mountain region information by analyzing terrain heights
    Identifies areas that are clearly mountain regions (high elevation, outside city)
    """
    from config import MAP_HALF_SIZE, TERRAIN_SAMPLE_STEP
    
    mountain_regions = []
    half = int(MAP_HALF_SIZE)
    step = int(TERRAIN_SAMPLE_STEP)
    
    # Use a larger step for region analysis to reduce data size
    region_step = step * 5  # 50 meters for region analysis
    
    # Thresholds for mountain classification
    MOUNTAIN_MIN_HEIGHT = 30.0  # Minimum height to be considered mountain
    MOUNTAIN_MIN_AREA = 4  # Minimum number of adjacent high points
    
    # Grid to track visited points
    visited = set()
    
    for x in range(-half, half + 1, region_step):
        for z in range(-half, half + 1, region_step):
            if (x, z) in visited:
                continue
                
            h = terrain_gen.get_terrain_height(x, z)
            
            # Check if this is a mountain region
            if h >= MOUNTAIN_MIN_HEIGHT:
                # Check distance from city center to confirm it's outside city
                dist = (x * x + z * z) ** 0.5
                angle = math.atan2(z, x)
                city_limit = terrain_gen.get_city_limit(angle)
                
                # If clearly outside city boundary, it's a mountain
                if dist > city_limit + 20:  # 20m buffer beyond city limit
                    # Find connected mountain region
                    region_points = []
                    stack = [(x, z)]
                    
                    while stack:
                        cx, cz = stack.pop()
                        if (cx, cz) in visited:
                            continue
                        visited.add((cx, cz))
                        
                        ch = terrain_gen.get_terrain_height(cx, cz)
                        if ch >= MOUNTAIN_MIN_HEIGHT:
                            region_points.append({
                                "x": round(cx),
                                "z": round(cz),
                                "height": round(ch, 1)
                            })
                            
                            # Check neighbors
                            for dx, dz in [(-region_step, 0), (region_step, 0), 
                                          (0, -region_step), (0, region_step)]:
                                nx, nz = cx + dx, cz + dz
                                if -half <= nx <= half and -half <= nz <= half:
                                    if (nx, nz) not in visited:
                                        nh = terrain_gen.get_terrain_height(nx, nz)
                                        if nh >= MOUNTAIN_MIN_HEIGHT:
                                            stack.append((nx, nz))
                    
                    # Only add region if it has minimum area
                    if len(region_points) >= MOUNTAIN_MIN_AREA:
                        heights = [p["height"] for p in region_points]
                        # Calculate bounding box
                        xs = [p["x"] for p in region_points]
                        zs = [p["z"] for p in region_points]
                        
                        mountain_regions.append({
                            "minX": min(xs),
                            "maxX": max(xs),
                            "minZ": min(zs),
                            "maxZ": max(zs),
                            "minHeight": round(min(heights), 1),
                            "maxHeight": round(max(heights), 1),
                            "avgHeight": round(sum(heights) / len(heights), 1),
                            "pointCount": len(region_points),
                            "centerX": round(sum(xs) / len(xs)),
                            "centerZ": round(sum(zs) / len(zs)),
                            "centerHeight": round(sum(heights) / len(heights), 1)
                        })
    
    return mountain_regions


def main():
    """Main generation function"""
    # Generate or use provided seed
    if len(sys.argv) > 1:
        try:
            seed = float(sys.argv[1])
        except ValueError:
            print(f"Invalid seed: {sys.argv[1]}, using random seed")
            seed = time.time() * 1000
    else:
        seed = time.time() * 1000
    
    print(f"Generating terrain with seed: {seed}")
    
    # Initialize random number generator
    rng = Mulberry32Random(int(seed))
    
    # Initialize SimplexNoise (OpenSimplex uses seed directly)
    # Note: OpenSimplex seed must be integer
    # OpenSimplex uses noise2() method, not noise2d()
    simplex = OpenSimplex(int(seed))
    
    # Create terrain generator
    terrain_gen = TerrainGenerator(simplex, LAKES, rng)
    
    # Generate buildings
    print("Generating city buildings...")
    building_gen = BuildingGenerator(terrain_gen, rng)
    city_buildings = building_gen.generate_city_buildings()
    print(f"Generated {len(city_buildings)} city buildings")
    
    print("Generating mountain buildings...")
    mountain_buildings, cabin_positions = building_gen.generate_mountain_buildings()
    print(f"Generated {len(mountain_buildings)} mountain buildings")
    
    # Generate trees
    print("Generating trees...")
    tree_gen = TreeGenerator(terrain_gen, rng, cabin_positions)
    trees = tree_gen.generate_trees()
    print(f"Generated {len(trees)} trees")
    
    # Generate users
    print("Generating users...")
    user_gen = UserGenerator(rng)
    users = user_gen.generate_users(city_buildings, mountain_buildings)
    print(f"Generated {len(users)} users")
    
    # Generate terrain map (legacy format, 10m resolution)
    terrain_map = generate_terrain_map(terrain_gen)
    print(f"Generated {len(terrain_map)} terrain sample points (legacy format)")
    
    # Generate complete height map (2m resolution for trajectory planning)
    print()
    height_map_data = generate_height_map(terrain_gen, resolution=2.0)
    print(f"Generated complete height map: {height_map_data['size'][0]}x{height_map_data['size'][1]} grid")
    
    # Generate mountain regions information
    print("Analyzing mountain regions...")
    mountain_regions = generate_mountain_regions(terrain_gen)
    print(f"Identified {len(mountain_regions)} mountain regions")
    
    # Combine building colliders
    building_colliders = city_buildings.copy()
    building_colliders.extend(mountain_buildings)
    
    # Create world state
    world_state = {
        "seed": seed,
        "mapSize": MAP_SIZE,
        "maxAltitude": MAX_ALT,
        "lakes": LAKES,
        "cityBuildings": city_buildings,
        "mountainBuildings": mountain_buildings,
        "buildingColliders": building_colliders,
        "cabinPositions": cabin_positions,
        "trees": trees,
        "finalUsers": users,
        "terrainMap": terrain_map,  # Legacy format (backward compatibility)
        "heightMap": height_map_data,  # New format: complete height map for trajectory planning
        "mountainRegions": mountain_regions  # Mountain region information
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save to JSON
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(world_state, f, indent=2)
    
    # Calculate file size
    file_size = os.path.getsize(OUTPUT_FILE)
    
    print(f"✓ Terrain data saved to {OUTPUT_FILE}")
    print(f"  - File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"  - City buildings: {len(city_buildings)}")
    print(f"  - Mountain buildings: {len(mountain_buildings)}")
    print(f"  - Trees: {len(trees)}")
    print(f"  - Users: {len(users)}")
    print(f"  - Terrain samples (legacy): {len(terrain_map)}")
    print(f"  - Height map: {height_map_data['size'][0]}x{height_map_data['size'][1]} grid ({height_map_data['step']}m resolution)")
    print(f"  - Mountain regions: {len(mountain_regions)}")


if __name__ == "__main__":
    main()
