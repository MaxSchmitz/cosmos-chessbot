#!/usr/bin/env python3
"""Inspect table material structure"""

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
                try:
                    print(f"{indent}  {inp.name} = {inp.default_value}")
                except:
                    print(f"{indent}  {inp.name} = <value>")

# Check TableTop material
table_mat = bpy.data.materials.get('TableTop')
if table_mat:
    print("\n=== TableTop Material ===")
    print(f"Has nodes: {table_mat.use_nodes}")

    if table_mat.use_nodes:
        print(f"\nAll nodes:")
        for node in table_mat.node_tree.nodes:
            print(f"  - {node.type}: {node.name}")

        # Find the Principled BSDF
        bsdf = table_mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            print(f"\nPrincipled BSDF:")
            print_node_inputs(bsdf)

        # Find any Mix nodes
        for node in table_mat.node_tree.nodes:
            if node.type == 'MIX':
                print(f"\n{node.name}:")
                print_node_inputs(node)
else:
    print("TableTop material not found!")

# List all materials with "table" in name
print("\n\nAll materials with 'table' in name:")
for mat in bpy.data.materials:
    if 'table' in mat.name.lower():
        print(f"  - {mat.name}")
