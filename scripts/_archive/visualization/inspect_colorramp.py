#!/usr/bin/env python3
"""Inspect ColorRamp in table material"""

import bpy

table_mat = bpy.data.materials.get('TableTop')
if table_mat and table_mat.use_nodes:
    # Find ColorRamp that feeds into BSDF
    colorramp = table_mat.node_tree.nodes.get('ColorRamp')
    if colorramp:
        print("ColorRamp node found")
        print(f"  Color mode: {colorramp.color_ramp.color_mode}")
        print(f"  Interpolation: {colorramp.color_ramp.interpolation}")
        print(f"  Number of elements (stops): {len(colorramp.color_ramp.elements)}")
        print("\n  Color stops:")
        for i, elem in enumerate(colorramp.color_ramp.elements):
            print(f"    [{i}] Position: {elem.position:.3f}, Color: {elem.color[:]}")
