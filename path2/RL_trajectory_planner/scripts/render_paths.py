"""
Placeholder script: save rollout paths to data/paths.json for Three.js viz.
"""

import json
from pathlib import Path
from typing import List, Dict

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "paths.json"


def save_paths(paths: List[List[Dict]], out_path: Path = DEFAULT_OUT):
    """
    paths: like [[{x,y,z}, ...], [...]]
    Output is compatible with terrain_generator visualization:
    [
      {"name": "Path0", "points": [...]},
      {"name": "Path1", "points": [...]}
    ]
    """
    payload = []
    for i, pts in enumerate(paths):
        payload.append({"name": f"Path{i}", "points": pts})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"saved {len(paths)} paths -> {out_path}")


if __name__ == "__main__":
    # Example: save one placeholder path
    demo_paths = [
        [{"x": 0, "y": 50, "z": 0}, {"x": 100, "y": 60, "z": 100}],
    ]
    save_paths(demo_paths)

