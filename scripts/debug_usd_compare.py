#!/usr/bin/env python3
"""Compare board.usd vs piece USDs to find rendering difference."""
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, Sdf

usd_dir = Path(__file__).parent.parent / "data" / "usd"

for name in ["board.usd", "pawn_w.usd", "king_b.usd"]:
    path = usd_dir / name
    stage = Usd.Stage.Open(str(path))
    print(f"\n{'=' * 50}")
    print(f"FILE: {name}")
    print(f"  defaultPrim: '{stage.GetDefaultPrim().GetName()}'")
    print(f"  upAxis: {UsdGeom.GetStageUpAxis(stage)}")
    print(f"  metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")

    root = stage.GetDefaultPrim()
    print(f"  root type: {root.GetTypeName()}")
    purpose = UsdGeom.Imageable(root).GetPurposeAttr().Get()
    vis = UsdGeom.Imageable(root).GetVisibilityAttr().Get()
    print(f"  root purpose: {purpose}")
    print(f"  root visibility: {vis}")

    xf = UsdGeom.Xformable(root)
    ops = [(op.GetOpName(), str(op.Get())) for op in xf.GetOrderedXformOps()]
    print(f"  root xformOps: {ops}")

    for child in root.GetChildren():
        cname = child.GetName()
        ctype = child.GetTypeName()
        print(f"  child: {cname} (type={ctype})")

        if ctype in ("Xform", "Mesh"):
            cxf = UsdGeom.Xformable(child)
            cops = [(op.GetOpName(), str(op.Get())) for op in cxf.GetOrderedXformOps()]
            print(f"    xformOps: {cops}")

            for gc in child.GetChildren():
                gctype = gc.GetTypeName()
                print(f"    gc: {gc.GetName()} (type={gctype})")
                if gctype == "Mesh":
                    mesh = UsdGeom.Mesh(gc)
                    extent = mesh.GetExtentAttr().Get()
                    points = mesh.GetPointsAttr().Get()
                    print(f"      extent: {extent}")
                    print(f"      points: {len(points) if points else 0}")
                    p = UsdGeom.Imageable(gc).GetPurposeAttr().Get()
                    v = UsdGeom.Imageable(gc).GetVisibilityAttr().Get()
                    print(f"      purpose: {p}, visibility: {v}")
    stage = None

simulation_app.close()
