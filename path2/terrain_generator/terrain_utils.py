"""
Terrain generation utility functions
"""

import math
from opensimplex import OpenSimplex
from config import *


class Mulberry32Random:
    """
    Mulberry32 PRNG - compatible with JavaScript version
    Ensures same seed produces same random sequence
    """
    def __init__(self, seed=0):
        self.seed = seed
    
    def random(self):
        """Generate next random number [0, 1)"""
        # Mulberry32 algorithm
        t = self.seed + 0x6D2B79F5
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= t + ((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF
        t = ((t ^ (t >> 14)) >> 0) & 0xFFFFFFFF
        self.seed = t
        return (t / 4294967296.0)
    
    def set_seed(self, seed):
        """Set random seed"""
        self.seed = seed


class TerrainGenerator:
    """Terrain generation functions"""
    
    def __init__(self, simplex, lakes, rng):
        self.simplex = simplex
        self.lakes = lakes
        self.rng = rng
    
    def get_city_limit(self, angle):
        """Calculate city boundary at given angle"""
        noise = self.simplex.noise2(math.cos(angle), math.sin(angle))
        return 140 + noise * 60
    
    def get_lake_factor(self, x, z, lake):
        """Calculate lake factor for point (x, z)"""
        dx = x - lake["x"]
        dz = z - lake["z"]
        cos = math.cos(-lake["rot"])
        sin = math.sin(-lake["rot"])
        nx = dx * cos - dz * sin
        nz = dx * sin + dz * cos
        return (nx * nx) / (lake["rx"] * lake["rx"]) + (nz * nz) / (lake["rz"] * lake["rz"])
    
    def is_in_lake(self, x, z, buffer=0):
        """Check if point is in lake"""
        for lake in self.lakes:
            factor = self.get_lake_factor(x, z, lake)
            if factor < 1.0 + (buffer / lake["rx"]):
                return True
        return False
    
    def get_terrain_height(self, x, z):
        """Calculate terrain height at point (x, z)"""
        dist = math.sqrt(x * x + z * z)
        angle = math.atan2(z, x)
        city_limit = self.get_city_limit(angle)
        
        mountain_height = 0
        if dist > city_limit:
            noise = self.simplex.noise2(x * 0.005, z * 0.005) * 120
            noise += self.simplex.noise2(x * 0.015, z * 0.015) * 45
            noise += self.simplex.noise2(x * 0.05, z * 0.05) * 10
            factor = min(1, (dist - city_limit) / TRANSITION_ZONE)
            factor = factor * factor * (3 - 2 * factor)  # Smoothstep
            mountain_height = max(0, noise + 10) * factor
        
        lake_blend = 1.0
        water_level = -2
        for lake in self.lakes:
            factor_sq = self.get_lake_factor(x, z, lake)
            factor = math.sqrt(factor_sq)
            if factor < 1.0:
                return water_level
            bank_width = 0.4
            if factor < 1.0 + bank_width:
                t = (factor - 1.0) / bank_width
                t = t * t * (3 - 2 * t)  # Smoothstep
                lake_blend = min(lake_blend, t)
        
        return mountain_height * lake_blend
    
    def is_on_road(self, x, z):
        """Check if point is on road"""
        dist = math.sqrt(x * x + z * z)
        angle = math.atan2(z, x)
        limit = self.get_city_limit(angle)
        if dist > limit - 10:
            return False
        
        if abs(x) < 60 and abs(z) < 60:
            if abs(x % 30) < 4 or abs(z % 30) < 4:
                return True
        
        if abs(z - x) < 6 and dist > 50:
            return True
        
        if abs(x - math.sin(z * 0.05) * 20) < 6 and z < -50:
            return True
        
        if abs(z - math.sin(x * 0.03) * 30 - 20) < 6 and x > 50:
            return True
        
        ring_noise = self.simplex.noise2(x * 0.01, z * 0.01) * 20
        if abs(dist - (110 + ring_noise)) < 5:
            return True
        
        return False


class BuildingGenerator:
    """Building generation functions"""
    
    def __init__(self, terrain_gen, rng):
        self.terrain_gen = terrain_gen
        self.rng = rng
    
    def generate_city_buildings(self):
        """Generate city buildings"""
        buildings = []
        b_count = 0
        attempts = 0
        
        while b_count < CITY_BUILDING_COUNT and attempts < CITY_BUILDING_MAX_ATTEMPTS:
            attempts += 1
            angle = self.rng.random() * math.pi * 2
            limit = self.terrain_gen.get_city_limit(angle)
            r = math.sqrt(self.rng.random()) * limit
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            
            if self.terrain_gen.is_in_lake(x, z, 5):
                continue
            if r < PARK_RADIUS or self.terrain_gen.is_on_road(x, z):
                continue
            
            dist_factor = r / limit
            height = 6 + 50 * (1 - dist_factor * dist_factor) * self.rng.random()
            height += self.rng.random() * 40 * (1.5 if self.rng.random() > 0.8 else 0.5)
            if height > 90:
                height = 90
            
            w = 4 + self.rng.random() * 5
            d = 4 + self.rng.random() * 5
            
            buildings.append({
                "x": x,
                "z": z,
                "halfWidth": w / 2,
                "halfDepth": d / 2,
                "height": height
            })
            b_count += 1
        
        return buildings
    
    def generate_mountain_buildings(self):
        """Generate mountain buildings"""
        buildings = []
        cabin_positions = []
        b_count = 0
        attempts = 0
        
        while b_count < MOUNTAIN_BUILDING_COUNT and attempts < MOUNTAIN_BUILDING_MAX_ATTEMPTS:
            attempts += 1
            x = (self.rng.random() - 0.5) * 850
            z = (self.rng.random() - 0.5) * 850
            dist = math.sqrt(x * x + z * z)
            angle = math.atan2(z, x)
            
            if dist < self.terrain_gen.get_city_limit(angle) + 20:
                continue
            if self.terrain_gen.is_in_lake(x, z, 10):
                continue
            
            y = self.terrain_gen.get_terrain_height(x, z)
            if y > 5 and y < MOUNTAIN_BUILDING_CEILING:
                h = 3 + self.rng.random() * 3
                w = 3 + self.rng.random() * 2
                
                buildings.append({
                    "x": x,
                    "z": z,
                    "halfWidth": w / 2,
                    "halfDepth": w / 2,
                    "height": y + h,
                    "baseHeight": y - 10
                })
                
                cabin_positions.append({
                    "x": x,
                    "z": z,
                    "r": w + 2
                })
                b_count += 1
        
        return buildings, cabin_positions


class TreeGenerator:
    """Tree generation functions"""
    
    def __init__(self, terrain_gen, rng, cabin_positions):
        self.terrain_gen = terrain_gen
        self.rng = rng
        self.cabin_positions = cabin_positions
    
    def generate_trees(self):
        """Generate tree positions"""
        trees = []
        t_count = 0
        attempts = 0
        
        while t_count < TREE_COUNT and attempts < TREE_MAX_ATTEMPTS:
            attempts += 1
            x = (self.rng.random() - 0.5) * 1000
            z = (self.rng.random() - 0.5) * 1000
            dist = math.sqrt(x * x + z * z)
            angle = math.atan2(z, x)
            
            if self.terrain_gen.is_in_lake(x, z, 2):
                continue
            
            city_limit = self.terrain_gen.get_city_limit(angle)
            if dist < city_limit and dist > PARK_RADIUS and not self.terrain_gen.is_on_road(x, z):
                continue
            
            if self.terrain_gen.is_on_road(x, z):
                continue
            
            # Check if too close to cabin
            close_cabin = False
            for cabin in self.cabin_positions:
                if (x - cabin["x"]) ** 2 + (z - cabin["z"]) ** 2 < cabin["r"] ** 2:
                    close_cabin = True
                    break
            if close_cabin:
                continue
            
            h = self.terrain_gen.get_terrain_height(x, z)
            if dist < PARK_RADIUS or (h > 2 and h < MAX_TREE_HEIGHT):
                scale = (0.5 if dist < PARK_RADIUS else 1.0) + self.rng.random() * 0.5
                # Always use actual terrain height to ensure trees sit on ground
                trees.append({
                    "x": x,
                    "y": h,  # Use actual terrain height for all trees
                    "z": z,
                    "scale": scale,
                    "rotation": self.rng.random() * math.pi
                })
                t_count += 1
        
        return trees


class UserGenerator:
    """User position generation functions"""
    
    def __init__(self, rng):
        self.rng = rng
    
    def get_surface_point(self, building):
        """Get a random point on building surface"""
        is_roof = self.rng.random() < 0.3
        if is_roof:
            return {
                "x": building["x"],
                "y": building["height"] + 0.5,
                "z": building["z"]
            }
        
        # Random side
        side = math.floor(self.rng.random() * 4)
        wx = building["x"]
        wy = building["height"]
        wz = building["z"]
        base = building.get("baseHeight", 0)
        wy = base + (building["height"] - base) * (0.2 + self.rng.random() * 0.7)
        offset = 0.5
        
        if side == 0:  # +Z
            wz += building["halfDepth"] + offset
            wx += (self.rng.random() - 0.5) * building["halfWidth"]
        elif side == 1:  # -Z
            wz -= building["halfDepth"] + offset
            wx += (self.rng.random() - 0.5) * building["halfWidth"]
        elif side == 2:  # +X
            wx += building["halfWidth"] + offset
            wz += (self.rng.random() - 0.5) * building["halfDepth"]
        else:  # -X
            wx -= building["halfWidth"] + offset
            wz += (self.rng.random() - 0.5) * building["halfDepth"]
        
        return {"x": wx, "y": wy, "z": wz}
    
    def generate_users(self, city_buildings, mountain_buildings):
        """Generate user positions"""
        users = []
        
        # Shuffle buildings
        city_sorted = sorted(city_buildings, key=lambda _: self.rng.random())
        mountain_sorted = sorted(mountain_buildings, key=lambda _: self.rng.random())
        
        # City users
        for i in range(min(CITY_USER_COUNT, len(city_sorted))):
            users.append(self.get_surface_point(city_sorted[i]))
        
        # Mountain users
        for i in range(min(MOUNTAIN_USER_COUNT, len(mountain_sorted))):
            users.append(self.get_surface_point(mountain_sorted[i]))
        
        return users

