#!/usr/bin/env python3
"""Detailed inspection of board material node setup"""

import bpy

def inspect_mix_node(mat_name):
    mat = bpy.data.materials.get(mat_name)
    if not mat or not mat.use_nodes:
        print(f"{mat_name} not found or no nodes")
        return

    print(f"\n=== {mat_name} ===")

    # Find Mix.001
    mix001 = mat.node_tree.nodes.get('Mix.001')
    if mix001:
        print("\nMix.001 node:")
        print(f"  Blend type: {mix001.blend_type if hasattr(mix001, 'blend_type') else 'N/A'}")
        print(f"  Data type: {mix001.data_type if hasattr(mix001, 'data_type') else 'N/A'}")

        # Check what feeds into Mix.001
        print(f"\n  Factor input:")
        if mix001.inputs['Factor'].is_linked:
            link = mix001.inputs['Factor'].links[0]
            print(f"    <- {link.from_node.name} ({link.from_node.type})")
        else:
            print(f"    Value: {mix001.inputs['Factor'].default_value}")

        print(f"\n  A input (color):")
        if mix001.inputs[6].is_linked:  # Color A
            link = mix001.inputs[6].links[0]
            print(f"    <- {link.from_node.name} ({link.from_node.type})")
        else:
            print(f"    Value: {mix001.inputs[6].default_value}")

        print(f"\n  B input (color):")
        if mix001.inputs[7].is_linked:  # Color B
            link = mix001.inputs[7].links[0]
            print(f"    <- {link.from_node.name} ({link.from_node.type})")
        else:
            print(f"    Value: {mix001.inputs[7].default_value}")

    # Check what Mix.001 feeds into
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        print(f"\n  Output goes to:")
        for link in mat.node_tree.links:
            if link.from_node == mix001:
                print(f"    -> {link.to_node.name}.{link.to_socket.name}")

# Check both materials
inspect_mix_node('BoardWhite')
inspect_mix_node('BoardBlack')

# List all texture nodes in board materials
print("\n\n=== All Texture Nodes ===")
for mat_name in ['BoardWhite', 'BoardBlack']:
    mat = bpy.data.materials.get(mat_name)
    if mat and mat.use_nodes:
        print(f"\n{mat_name}:")
        for node in mat.node_tree.nodes:
            if 'TEX' in node.type or 'VORONOI' in node.type or 'NOISE' in node.type:
                print(f"  - {node.type}: {node.name}")
