"""
Generate a safety-aware path through all user points with horizontal obstacle avoidance.
- Path goes through each user in order (User1 -> ...).
- Avoids buildings, lakes, trees, mountain regions; keeps inside map bounds.
- Altitude: max(user_y + SAFETY, terrain + SAFETY, building top + SAFETY).
- Outputs 120-point path to path2/data/paths.json.
"""

import json
import math
from pathlib import Path
import sys
import heapq

ROOT = Path(__file__).resolve().parents[3]  # repo root
PATH2 = ROOT / "path2"

# Ensure imports for terrain_generator
if str(PATH2) not in sys.path:
    sys.path.append(str(PATH2))
if str(PATH2 / "terrain_generator") not in sys.path:
    sys.path.append(str(PATH2 / "terrain_generator"))

from terrain_generator.height_map_utils import HeightMap  # noqa: E402

SAFETY = 10.0
TOTAL_POINTS = 120
GRID_STEP = 10.0  # horizontal planning resolution (meters)
X_MIN = -500
X_MAX = 500
Z_MIN = -500
Z_MAX = 500


def load_data():
    data_path = PATH2 / "data" / "terrain_data.json"
    with data_path.open() as f:
        data = json.load(f)
    hm = HeightMap(data["heightMap"])
    return data, hm


def in_bounds(x, z):
    return X_MIN <= x <= X_MAX and Z_MIN <= z <= Z_MAX


def lake_collision(lake, x, z, margin):
    dx, dz = x - lake["x"], z - lake["z"]
    rx, rz = lake["rx"] + margin, lake["rz"] + margin
    if rx <= 0 or rz <= 0:
        return False
    return (dx / rx) ** 2 + (dz / rz) ** 2 <= 1.0


def building_collision(b, x, z, margin):
    return (abs(x - b["x"]) <= b["halfWidth"] + margin) and (abs(z - b["z"]) <= b["halfDepth"] + margin)


def tree_collision(t, x, z, margin):
    r = t.get("scale", 1.0) * 0.8 + margin
    return math.hypot(x - t["x"], z - t["z"]) <= r


def mountain_collision(reg, x, z, margin):
    return (reg["minX"] - margin) <= x <= (reg["maxX"] + margin) and (reg["minZ"] - margin) <= z <= (reg["maxZ"] + margin)


def is_blocked(x, z, data, margin):
    if not in_bounds(x, z):
        return True
    # Lakes
    for lake in data.get("lakes", []):
        if lake_collision(lake, x, z, margin):
            return True
    # Buildings
    for b in data.get("buildingColliders", []):
        if building_collision(b, x, z, margin):
            return True
    # Trees
    for t in data.get("trees", []):
        if tree_collision(t, x, z, margin):
            return True
    # Mountain regions
    for reg in data.get("mountainRegions", []):
        if mountain_collision(reg, x, z, margin):
            return True
    return False


def neighbors(node):
    x, z = node
    step = GRID_STEP
    for dx, dz in [(step, 0), (-step, 0), (0, step), (0, -step), (step, step), (step, -step), (-step, step), (-step, -step)]:
        yield (x + dx, z + dz)


def a_star(start, goal, data):
    start = snap_to_grid(start)
    goal = snap_to_grid(goal)
    if is_blocked(*start, data=data, margin=SAFETY):
        return None
    if is_blocked(*goal, data=data, margin=SAFETY):
        return None

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if dist(current, goal) <= GRID_STEP:
            return reconstruct_path(came_from, current) + [goal]
        for n in neighbors(current):
            if is_blocked(*n, data=data, margin=SAFETY):
                continue
            tentative = g[current] + dist(current, n)
            if tentative < g.get(n, float("inf")):
                came_from[n] = current
                g[n] = tentative
                f = tentative + dist(n, goal)
                heapq.heappush(open_set, (f, n))
    return None


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def snap_to_grid(p):
    x = round(p[0] / GRID_STEP) * GRID_STEP
    z = round(p[1] / GRID_STEP) * GRID_STEP
    return (x, z)


def interpolate_path(points, target_n):
    """Resample path to target_n using cumulative distance (upsample or downsample)."""
    if len(points) < 2:
        return points
    # cumulative lengths
    d = [0.0]
    for i in range(1, len(points)):
        d.append(d[-1] + dist(points[i - 1], points[i]))
    total = d[-1]
    if total == 0:
        return [points[0]] * target_n
    step = total / (target_n - 1)
    res = []
    j = 1
    for k in range(target_n):
        target = k * step
        while j < len(points) and d[j] < target:
            j += 1
        if j >= len(points):
            res.append(points[-1])
        else:
            p0, p1 = points[j - 1], points[j]
            t = 0 if d[j] == d[j - 1] else (target - d[j - 1]) / (d[j] - d[j - 1])
            x = p0[0] + (p1[0] - p0[0]) * t
            z = p0[1] + (p1[1] - p0[1]) * t
            res.append((x, z))
    return res


def apply_altitude(path_xz, data, hm):
    buildings = data.get("buildingColliders", [])
    users = data.get("finalUsers", [])
    user_min_y = min(u["y"] for u in users)
    user_min_y = max(user_min_y + SAFETY, 0)

    pts = []
    for x, z in path_xz:
        terrain_h = hm.get_height_interpolated(x, z)
        y = max(user_min_y, terrain_h + SAFETY)
        # building top clearance
        for b in buildings:
            if building_collision(b, x, z, 0):
                base = b.get("baseHeight", 0.0)
                top = base + b["height"] + SAFETY
                y = max(y, top)
        pts.append({"x": x, "y": y, "z": z})
    return pts


def generate_path():
    data, hm = load_data()
    users = data["finalUsers"]
    segments = len(users) - 1
    raw_path = []

    for i in range(segments):
        a = users[i]
        b = users[i + 1]
        start = (a["x"], a["z"])
        goal = (b["x"], b["z"])
        path = a_star(start, goal, data)
        if path is None:
            # fallback straight line in xz
            path = [start, goal]
        # drop first node except for first segment to avoid duplication
        if i > 0 and path:
            path = path[1:]
        raw_path.extend(path)

    # Downsample or keep to target count before altitude
    raw_path = interpolate_path(raw_path, TOTAL_POINTS)
    pts3d = apply_altitude(raw_path, data, hm)

    out_path = PATH2 / "data" / "paths.json"
    out_path.write_text(json.dumps([{"name": "Path0", "points": pts3d}], indent=2))
    print(f"Wrote {len(pts3d)} points -> {out_path}")


if __name__ == "__main__":
    generate_path()

