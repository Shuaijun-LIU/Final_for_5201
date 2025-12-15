"""
Height map utilities for reading terrain data from JSON
Provides fast lookup of terrain height using the heightMap format
"""

import math


class HeightMap:
    """
    Fast terrain height lookup using height map data from JSON
    
    Usage:
        with open('data/terrain_data.json') as f:
            data = json.load(f)
        
        height_map = HeightMap(data['heightMap'])
        height = height_map.get_height(x, z)
    """
    
    def __init__(self, height_map_data):
        """
        Initialize height map from JSON data
        
        Args:
            height_map_data: dict with keys:
                - heightMap: 2D array [z][x] of heights
                - origin: [x_min, z_min]
                - step: resolution in meters
                - size: [width, depth] in grid cells
                - xCoords: list of x coordinates (optional, for faster lookup)
                - zCoords: list of z coordinates (optional, for faster lookup)
        """
        self.height_map = height_map_data['heightMap']
        self.origin = height_map_data['origin']
        self.step = height_map_data['step']
        self.size = height_map_data['size']  # [width, depth]
        
        # Optional: use pre-computed coordinates for faster lookup
        self.x_coords = height_map_data.get('xCoords', None)
        self.z_coords = height_map_data.get('zCoords', None)
        
        # Cache for coordinate calculations
        self._x_min = self.origin[0]
        self._z_min = self.origin[1]
        self._width = self.size[0]
        self._depth = self.size[1]
    
    def get_height(self, x, z):
        """
        Get terrain height at point (x, z)
        
        Args:
            x: X coordinate in meters
            z: Z coordinate in meters
        
        Returns:
            float: Terrain height in meters
        """
        # Calculate grid indices
        x_idx = self._coord_to_index(x, self._x_min, self.step, self._width)
        z_idx = self._coord_to_index(z, self._z_min, self.step, self._depth)
        
        # Clamp to valid range
        x_idx = max(0, min(self._width - 1, x_idx))
        z_idx = max(0, min(self._depth - 1, z_idx))
        
        # Return height (heightMap is row-major: [z][x])
        return self.height_map[z_idx][x_idx]
    
    def get_height_interpolated(self, x, z):
        """
        Get terrain height with bilinear interpolation for smoother results
        
        Args:
            x: X coordinate in meters
            z: Z coordinate in meters
        
        Returns:
            float: Interpolated terrain height in meters
        """
        # Calculate grid indices (as floats)
        x_idx_f = (x - self._x_min) / self.step
        z_idx_f = (z - self._z_min) / self.step
        
        # Get integer indices
        x_idx0 = int(math.floor(x_idx_f))
        z_idx0 = int(math.floor(z_idx_f))
        x_idx1 = min(self._width - 1, x_idx0 + 1)
        z_idx1 = min(self._depth - 1, z_idx0 + 1)
        
        # Clamp to valid range
        x_idx0 = max(0, x_idx0)
        z_idx0 = max(0, z_idx0)
        
        # Get fractional parts
        fx = x_idx_f - x_idx0
        fz = z_idx_f - z_idx0
        
        # Bilinear interpolation
        h00 = self.height_map[z_idx0][x_idx0]
        h10 = self.height_map[z_idx0][x_idx1]
        h01 = self.height_map[z_idx1][x_idx0]
        h11 = self.height_map[z_idx1][x_idx1]
        
        # Interpolate in x direction
        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        
        # Interpolate in z direction
        return h0 * (1 - fz) + h1 * fz
    
    def _coord_to_index(self, coord, origin, step, size):
        """Convert coordinate to grid index"""
        idx = (coord - origin) / step
        return int(round(idx))
    
    def get_bounds(self):
        """
        Get map bounds
        
        Returns:
            dict with keys: x_min, x_max, z_min, z_max
        """
        return {
            'x_min': self._x_min,
            'x_max': self._x_min + (self._width - 1) * self.step,
            'z_min': self._z_min,
            'z_max': self._z_min + (self._depth - 1) * self.step
        }
    
    def is_in_bounds(self, x, z):
        """Check if point (x, z) is within map bounds"""
        bounds = self.get_bounds()
        return (bounds['x_min'] <= x <= bounds['x_max'] and
                bounds['z_min'] <= z <= bounds['z_max'])

