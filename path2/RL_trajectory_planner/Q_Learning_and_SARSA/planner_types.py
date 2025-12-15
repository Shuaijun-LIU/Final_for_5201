from typing import Hashable, Tuple

# Grid coordinates in meters (multiple of grid_step)
Coord = Tuple[int, int]  # (gx, gz)

# Generic RL state type (Coord, or (Coord, mask), etc.)
State = Hashable


