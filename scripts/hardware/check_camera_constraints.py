#!/usr/bin/env python3
"""Check if camera has constraints or drivers"""

import bpy

# Check all cameras
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        print(f"\nCamera: {obj.name}")
        print(f"  Location: {obj.location}")
        print(f"  Is active render camera: {obj == bpy.context.scene.camera}")

        # Check for constraints
        if obj.constraints:
            print(f"  Constraints: {len(obj.constraints)}")
            for constraint in obj.constraints:
                print(f"    - {constraint.type}: {constraint.name} (enabled: {not constraint.mute})")
        else:
            print(f"  Constraints: None")

        # Check for drivers
        if obj.animation_data and obj.animation_data.drivers:
            print(f"  Drivers: {len(obj.animation_data.drivers)}")
            for driver in obj.animation_data.drivers:
                print(f"    - {driver.data_path}")
        else:
            print(f"  Drivers: None")

print(f"\nActive render camera: {bpy.context.scene.camera.name if bpy.context.scene.camera else 'None'}")
