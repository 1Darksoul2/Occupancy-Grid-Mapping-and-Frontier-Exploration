"""Generate Webots worlds (worlds/<map>.wbt) from the rectangles in aisr_explore/maps.py.

    python tools/make_worlds.py                    # all maps, strategy utility_commit, start 0
    python tools/make_worlds.py --strategy nearest --start 1

The robot is a Pioneer 3-DX with, in its extension slot: a 360-degree 2-D lidar,
a GPS and an inertial unit (pose), and a Display that shows the live map.
"""

import argparse
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from aisr_explore.maps import MAPS  # noqa: E402

WEBOTS_VERSION = "R2025a"
PROTO = f"https://raw.githubusercontent.com/cyberbotics/webots/{WEBOTS_VERSION}/projects"

HEADER = f"""#VRML_SIM {WEBOTS_VERSION} utf8

EXTERNPROTO "{PROTO}/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "{PROTO}/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "{PROTO}/objects/floors/protos/RectangleArena.proto"
EXTERNPROTO "{PROTO}/robots/adept/pioneer3/protos/Pioneer3dx.proto"

WorldInfo {{
  info [ "AISR - occupancy-grid mapping and frontier exploration" "map: {{map}}" ]
  title "Frontier exploration - {{map}}"
  basicTimeStep 32
}}
Viewpoint {{
  orientation 0 1 0 1.5708
  position 0 0 17
}}
TexturedBackground {{
}}
TexturedBackgroundLight {{
}}
RectangleArena {{
  floorSize 10.6 10.6
  floorTileSize 1 1
  wallHeight 0.05
}}
"""

SOLID = """Solid {{
  translation {x:.3f} {y:.3f} {hz:.3f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {color}
        roughness 1
        metalness 0
      }}
      geometry DEF {defn} Box {{
        size {sx:.3f} {sy:.3f} {h:.3f}
      }}
    }}
  ]
  name "{name}"
  boundingObject USE {defn}
}}
"""

ROBOT = """Pioneer3dx {{
  translation {x:.3f} {y:.3f} 0.0975
  rotation 0 0 1 {th:.4f}
  controller "frontier_explorer"
  controllerArgs [
    "--map={map}"
    "--strategy={strategy}"
  ]
  extensionSlot [
    Lidar {{
      translation 0 0 0.25
      name "lidar"
      horizontalResolution 360
      fieldOfView 6.2832
      numberOfLayers 1
      verticalFieldOfView 0.05
      minRange 0.1
      maxRange 5
      noise 0.002
    }}
    GPS {{
      name "gps"
    }}
    InertialUnit {{
      name "imu"
    }}
    Display {{
      name "map_display"
      width 240
      height 240
    }}
  ]
}}
"""


def make_world(name, strategy, start):
    m = MAPS[name]
    out = [HEADER.replace("{map}", name)]
    for i, (cx, cy, sx, sy) in enumerate(m["walls"]):
        out.append(SOLID.format(x=cx, y=cy, hz=0.4, sx=sx, sy=sy, h=0.8,
                                color="0.82 0.78 0.72", name=f"wall_{i}", defn=f"WALL_{i}"))
    for i, (cx, cy, sx, sy) in enumerate(m["boxes"]):
        out.append(SOLID.format(x=cx, y=cy, hz=0.4, sx=sx, sy=sy, h=0.8,
                                color="0.55 0.35 0.2", name=f"box_{i}", defn=f"BOX_{i}"))
    x, y, th = m["starts"][start]
    out.append(ROBOT.format(x=x, y=y, th=th, map=name, strategy=strategy))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=list(MAPS))
    ap.add_argument("--strategy", default="utility_commit")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "worlds"), exist_ok=True)
    for name in a.maps:
        path = os.path.join(ROOT, "worlds", f"{name}.wbt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(make_world(name, a.strategy, a.start))
        print("wrote", path)


if __name__ == "__main__":
    main()
