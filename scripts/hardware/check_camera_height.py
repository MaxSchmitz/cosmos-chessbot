#!/usr/bin/env python3
"""Check original camera height in VALUE board.blend"""

import bpy

# Get camera
camera = bpy.data.objects.get('Camera')
if camera:
    print(f'Original Camera location: {camera.location}')
    print(f'Original Camera Z height: {camera.location.z:.4f}m')

# Get board
board = bpy.data.objects.get('chessBoard')
if board:
    print(f'Board location: {board.location}')
    print(f'Board Z: {board.location.z:.4f}m')

# Get a piece to see height
for obj in bpy.data.objects:
    if obj.name == 'K0':  # King piece
        print(f'King piece location: {obj.location}')
        print(f'King Z: {obj.location.z:.4f}m')
        break
