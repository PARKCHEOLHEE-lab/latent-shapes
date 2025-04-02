import bpy
import subprocess
import mathutils


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
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()
        
        bpy.ops.wm.obj_import(filepath=obj_path)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.join()
        
        return bpy.context.active_object

    def export_obj(self, output_path: str):
        bpy.ops.wm.obj_export(filepath=output_path)

    def shrink_wrap(self, obj: bpy.types.Object, subdivision: int):
        # Create cube and match it to object's bounds
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        
        # Get object's bounding box dimensions and center
        bound_box = obj.bound_box
        bbox_corners = [mathutils.Vector(corner) for corner in bound_box]
        bbox_center = sum(bbox_corners, mathutils.Vector()) / 8
        
        # Calculate bounding box dimensions
        min_corner = mathutils.Vector(map(min, zip(*bound_box)))
        max_corner = mathutils.Vector(map(max, zip(*bound_box)))
        dimensions = max_corner - min_corner
        
        # Set cube position and scale to match bounding box
        world_bbox_center = obj.matrix_world @ bbox_center
        cube.location = world_bbox_center
        cube.scale = dimensions / 2
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        
        subdiv_modifier = cube.modifiers.new(name="subdiv", type="SUBSURF")
        subdiv_modifier.levels = subdivision
        subdiv_modifier.render_levels = subdivision

        shrinkwrap_modifier = cube.modifiers.new(name="shrinkwrap", type="SHRINKWRAP")
        shrinkwrap_modifier.target = obj
        shrinkwrap_modifier.wrap_method = "PROJECT"
        shrinkwrap_modifier.cull_face = "OFF"
        shrinkwrap_modifier.use_positive_direction = True
        shrinkwrap_modifier.use_negative_direction = True
        shrinkwrap_modifier.offset = 0.0

        bpy.context.view_layer.objects.active = cube
        bpy.ops.object.modifier_apply(modifier="subdiv")
        bpy.ops.object.modifier_apply(modifier="shrinkwrap")
        
        return cube

    def run(self, obj_path: str, output_path: str, subdivision: int):
        
        # Open Blender
        subprocess.run(self.command)

        obj = self.import_obj(obj_path=obj_path)
        shrink_wrapped_obj = self.shrink_wrap(obj=obj, subdivision=subdivision)
        self.export_obj(output_path=output_path)

        # Quit Blender
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    shrink_wrapper = ShrinkWrapper(
        blender_path=r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        script_path=r"C:\Users\josep\Documents\GitHub\3D-Model-Processing\shrinkwrap.py"
    )
    shrink_wrapper.run(
        obj_path=r"data\03001627\1a6f615e8b1b5ae4dbbc9440457e303e\models\model_normalized.obj",
        output_path="test.obj",
        subdivision=5,
    )

