#!/usr/bin/env python3
"""Inspect board material node connections"""

import bpy

def print_node_inputs(node, indent="    "):
    """Print all inputs of a node and their connections."""
    print(f"{indent}Inputs:")
    for inp in node.inputs:
        if inp.is_linked:
            link = inp.links[0]
            from_node = link.from_node
            from_socket = link.from_socket
            print(f"{indent}  {inp.name} <- {from_node.name}.{from_socket.name}")
        else:
            # Show default value if available
            if hasattr(inp, 'default_value'):
                print(f"{indent}  {inp.name} = {inp.default_value}")

# Check BoardWhite material
board_white = bpy.data.materials.get('BoardWhite')
if board_white and board_white.use_nodes:
    print("\n=== BoardWhite Material ===")

    # Find the Principled BSDF
    bsdf = board_white.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        print(f"\nPrincipled BSDF:")
        print_node_inputs(bsdf)

    # Find Mix nodes
    for node in board_white.node_tree.nodes:
        if node.type == 'MIX':
            print(f"\n{node.name}:")
            print_node_inputs(node)

# Check BoardBlack material
board_black = bpy.data.materials.get('BoardBlack')
if board_black and board_black.use_nodes:
    print("\n=== BoardBlack Material ===")

    # Find the Principled BSDF
    bsdf = board_black.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        print(f"\nPrincipled BSDF:")
        print_node_inputs(bsdf)

    # Find Mix nodes
    for node in board_black.node_tree.nodes:
        if node.type == 'MIX':
            print(f"\n{node.name}:")
            print_node_inputs(node)
