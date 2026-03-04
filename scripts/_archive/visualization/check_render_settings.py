#!/usr/bin/env python3
"""Check render settings"""

import bpy

scene = bpy.context.scene

print("Render Settings:")
print(f"  Resolution: {scene.render.resolution_x} x {scene.render.resolution_y}")
print(f"  Resolution %: {scene.render.resolution_percentage}%")
print(f"  Actual output: {int(scene.render.resolution_x * scene.render.resolution_percentage / 100)} x {int(scene.render.resolution_y * scene.render.resolution_percentage / 100)}")
print(f"  Samples: {scene.cycles.samples if scene.render.engine == 'CYCLES' else 'N/A'}")
print(f"  Render engine: {scene.render.engine}")
print(f"  File format: {scene.render.image_settings.file_format}")
print(f"  Color mode: {scene.render.image_settings.color_mode}")
print(f"  Quality: {scene.render.image_settings.quality}")
