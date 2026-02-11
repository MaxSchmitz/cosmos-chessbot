#!/usr/bin/env python3
"""Find the geometry child name for each piece USD."""
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom
from pathlib import Path

usd_dir = Path(__file__).parent.parent / "data" / "usd"

for usd_file in sorted(usd_dir.glob("*.usd")):
    stage = Usd.Stage.Open(str(usd_file))
    root = stage.GetDefaultPrim()
    geom_children = []
    for child in root.GetChildren():
        name = child.GetName()
        if name not in ("_materials", "env_light"):
            geom_children.append(name)
    print(f"{usd_file.stem}: root=/root, geom_child={geom_children}")

simulation_app.close()
