import bpy
import subprocess


class ShrinkWrapper:
    def __init__(self, blender_path: str, script_path: str):
        self.blender_path = blender_path
        self.script_path = script_path

        self.command = [
            self.blender_path,
            "--background",
            "--python",
            self.script_path,
        ]
        
    def import_obj(self, obj_path: str):
        bpy.ops.import_mesh.obj(filepath=obj_path)

    def shrink_wrap(self):
        
        # Open Blender
        subprocess.run(self.command)

        # Delete all objects
        bpy.ops.object.delete()

        # Quit Blender
        bpy.ops.wm.quit_blender()

        pass