#!/usr/bin/env python3
"""List available piece sets in ChessR blend file."""

import bpy

# Get the Pieces sets collection
try:
    pieces_sets = bpy.data.collections.get('Pieces sets')
    if pieces_sets:
        print(f"\nFound {len(pieces_sets.children)} piece sets in ChessR:")
        print("=" * 60)
        for piece_set in pieces_sets.children:
            print(f"  - {piece_set.name} ({len(piece_set.all_objects)} objects)")
    else:
        print("No 'Pieces sets' collection found")
except Exception as e:
    print(f"Error: {e}")

# Also check for individual collections
print("\n\nAll collections:")
print("=" * 60)
for collection in bpy.data.collections:
    print(f"  - {collection.name} ({len(collection.all_objects)} objects)")
