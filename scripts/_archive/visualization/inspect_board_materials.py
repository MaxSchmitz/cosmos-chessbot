#!/usr/bin/env python3
"""Inspect board materials in VALUE board.blend"""

import bpy

# Get board object
board = bpy.data.objects.get('chessBoard')
if board:
    print(f"\nBoard object: {board.name}")
    print(f"  Type: {board.type}")
    print(f"  Materials ({len(board.data.materials)}):")
    for i, mat in enumerate(board.data.materials):
        if mat:
            print(f"    [{i}] {mat.name}")
            if mat.use_nodes:
                print(f"        Has nodes: True")
                for node in mat.node_tree.nodes:
                    print(f"          - {node.type}: {node.name}")

# Check all objects with "board" in name
print("\nAll board-related objects:")
for obj in bpy.data.objects:
    if 'board' in obj.name.lower() or 'chess' in obj.name.lower():
        print(f"\n{obj.name}:")
        print(f"  Type: {obj.type}")
        if obj.type == 'MESH' and obj.data.materials:
            print(f"  Materials ({len(obj.data.materials)}):")
            for mat in obj.data.materials:
                if mat:
                    print(f"    - {mat.name}")

# List all materials with "board" in name
print("\nAll materials with 'board' or 'chess' in name:")
for mat in bpy.data.materials:
    if 'board' in mat.name.lower() or 'chess' in mat.name.lower():
        print(f"  - {mat.name}")
