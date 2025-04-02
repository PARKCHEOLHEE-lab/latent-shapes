import bpy
import os
import sys
import bmesh
import math
import time
import random

def main():
    # Script start
    print("Blender Python script started")

    # Get variables from environment
    input_file = os.environ.get('BL_INPUT_FILE', '')
    target_faces = int(os.environ.get('BL_TARGET_FACES', '12324'))
    output_file = os.environ.get('BL_OUTPUT_FILE', '')
    exact_count = os.environ.get('BL_EXACT_COUNT', 'True').lower() == 'true'

    print(f"Input file: {input_file}")
    print(f"Target faces: {target_faces}")
    print(f"Output file: {output_file}")
    print(f"Exact count: {exact_count}")

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file does not exist: {input_file}")
        sys.exit(1)

    # Delete existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

    # Import mesh using built-in operators
    success = False

    # Try different import methods
    try:
        bpy.ops.wm.obj_import(filepath=input_file)
        print("Imported OBJ file using wm.obj_import")
        success = True
    except Exception as e:
        print(f"wm.obj_import error: {str(e)}")
        try:
            bpy.ops.import_scene.obj(filepath=input_file)
            print("Imported OBJ file using import_scene.obj")
            success = True
        except Exception as e2:
            print(f"import_scene.obj error: {str(e2)}")

    # Fall back to manual parsing if both import methods fail
    if not success:
        print("Creating mesh manually...")
        
        try:
            # Create a new mesh and object
            mesh = bpy.data.meshes.new("ImportedMesh")
            obj = bpy.data.objects.new("ImportedObject", mesh)
            
            # Link object to scene
            bpy.context.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Create bmesh
            bm = bmesh.new()
            
            # Parse OBJ file manually (basic implementation)
            vertices = []
            faces = []
            
            with open(input_file, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    elif line.startswith('f '):
                        parts = line.split()
                        # OBJ faces are 1-indexed
                        face = []
                        for p in parts[1:]:
                            # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                            idx = p.split('/')[0]
                            if idx:
                                face.append(int(idx) - 1)
                        if len(face) >= 3:
                            faces.append(face)
            
            # Create vertices
            for v in vertices:
                bm.verts.new(v)
            
            bm.verts.ensure_lookup_table()
            
            # Create faces
            for f in faces:
                try:
                    bm.faces.new([bm.verts[i] for i in f])
                except Exception as fe:
                    print(f"Error creating face {f}: {str(fe)}")
            
            # Update the mesh
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
            
            success = True
        except Exception as e3:
            print(f"Manual mesh creation failed: {str(e3)}")
            sys.exit(1)

    # Get the active object
    if len(bpy.context.selected_objects) == 0:
        # Find all mesh objects
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        
        if not mesh_objects:
            print("No mesh objects in scene.")
            sys.exit(1)
        
        # Select first mesh
        obj = mesh_objects[0]
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    else:
        obj = bpy.context.active_object

    print(f"Processing object: {obj.name}")
    original_vertices = len(obj.data.vertices)
    original_faces = len(obj.data.polygons)
    print(f"===== ORIGINAL MESH STATS =====")
    print(f"Vertices: {original_vertices}")
    print(f"Faces: {original_faces}")
    print(f"================================")

    # Save these counts to file for the parent process to read
    with open('mesh_stats.txt', 'w') as stats_file:
        stats_file.write(f"ORIGINAL_VERTICES={original_vertices}\n")
        stats_file.write(f"ORIGINAL_FACES={original_faces}\n")

    # Store a copy of the original mesh for shrinkwrap
    original_obj = obj.copy()
    original_obj.data = obj.data.copy()
    bpy.context.collection.objects.link(original_obj)
    original_obj.hide_set(True)  # Hide it from view
    print("Created a copy of the original mesh for shrinkwrap")

    # Make sure we're working with triangulated mesh for consistent face counting
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Check face count after triangulation
    current_faces = len(obj.data.polygons)
    print(f"Faces after triangulation: {current_faces}")

    # Clean up mesh first
    print("Performing initial mesh cleanup...")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)  # Remove duplicate vertices
    bpy.ops.mesh.normals_make_consistent(inside=False)  # Fix normals
    bpy.ops.object.mode_set(mode='OBJECT')

    # Remesh the object to get a clean topology first
    if current_faces > 100:  # Only apply remesh if we have enough faces
        try:
            print("Applying Voxel Remesh for clean topology...")
            # Calculate voxel size based on object dimensions
            dimensions = obj.dimensions
            max_dim = max(dimensions)
            
            # Estimate a reasonable voxel size
            # Higher value = less detail, fewer polygons
            voxel_size = max_dim / 50  
            
            remesh_mod = obj.modifiers.new(name="Remesh", type='REMESH')
            remesh_mod.mode = 'VOXEL'
            remesh_mod.voxel_size = voxel_size
            remesh_mod.use_smooth_shade = True
            bpy.ops.object.modifier_apply(modifier="Remesh")
            
            remeshed_faces = len(obj.data.polygons)
            print(f"After Voxel Remesh: {len(obj.data.vertices)} vertices, {remeshed_faces} faces")
        except Exception as e:
            print(f"Voxel Remesh failed: {str(e)}. Continuing without remesh.")

    # Apply Shrinkwrap to conform to the original mesh
    print("Applying Shrinkwrap modifier to maintain original shape...")
    shrinkwrap_mod = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_mod.target = original_obj
    shrinkwrap_mod.wrap_method = 'PROJECT'
    shrinkwrap_mod.use_project_z = True
    shrinkwrap_mod.use_negative_direction = True
    shrinkwrap_mod.use_positive_direction = True
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
    print("Shrinkwrap applied")

    # Now smooth the result a bit
    smooth_mod = obj.modifiers.new(name="Smooth", type='SMOOTH')
    smooth_mod.iterations = 2
    smooth_mod.factor = 0.5
    bpy.ops.object.modifier_apply(modifier="Smooth")

    # Apply another shrinkwrap to ensure we stick to the original shape after smoothing
    shrinkwrap_mod = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_mod.target = original_obj
    shrinkwrap_mod.wrap_method = 'NEAREST_SURFACEPOINT'
    shrinkwrap_mod.offset = 0.0001  # Small offset to prevent intersections
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

    # Now adjust the face count to the target value
    current_faces = len(obj.data.polygons)
    print(f"Faces after initial processing: {current_faces}")

    # Function to adjust face count using binary search if exact count is required
    def adjust_face_count_exact(obj, target_faces, max_iterations=20, tolerance=1):
        """
        Adjust face count to match target_faces exactly or within tolerance
        using binary search to refine the decimation ratio.
        """
        current_faces = len(obj.data.polygons)
        print(f"Attempting to get exactly {target_faces} faces (current: {current_faces})...")
        
        # If we're already at the target, nothing to do
        if abs(current_faces - target_faces) <= tolerance:
            print(f"Already within tolerance: {current_faces} faces")
            return
        
        # Make a copy of the mesh before we start iterating
        backup_mesh = obj.data.copy()
        
        # First, ensure we have more faces than needed for better control
        if current_faces < target_faces * 1.2:  # If we have less than 120% of target
            print("Subdividing first to ensure enough faces for precise control...")
            
            # Apply subdivision to get more faces
            modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            modifier.levels = 1  # One level of subdivision is usually enough
            bpy.ops.object.modifier_apply(modifier="Subdivision")
            
            # Update current face count
            current_faces = len(obj.data.polygons)
            print(f"After initial subdivision: {current_faces} faces")
            
            # Create a new backup after subdivision
            backup_mesh = obj.data.copy()
        
        # Determine initial range for binary search
        min_ratio = 0.01
        max_ratio = 0.99
        mid_ratio = target_faces / current_faces  # Initial guess
        
        # Track best result so far
        best_ratio = mid_ratio
        best_face_count = current_faces
        best_difference = abs(current_faces - target_faces)
        
        # Binary search phase
        for i in range(max_iterations):
            # Reset mesh to backup before each iteration
            if i > 0:
                obj.data = backup_mesh.copy()
            
            # Apply decimate modifier with current ratio
            modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            modifier.ratio = mid_ratio
            modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier="Decimate")
            
            # Check result
            new_face_count = len(obj.data.polygons)
            difference = abs(new_face_count - target_faces)
            print(f"  Iteration {i+1}: ratio {mid_ratio:.6f} gave {new_face_count} faces (diff: {difference})")
            
            # Keep track of best result
            if difference < best_difference:
                best_difference = difference
                best_ratio = mid_ratio
                best_face_count = new_face_count
            
            # If we're close enough, we're done
            if difference <= tolerance:
                print(f"  Success! Got {new_face_count} faces (target: {target_faces})")
                return
            
            # Otherwise, adjust the ratio using binary search
            if new_face_count > target_faces:
                # Too many faces, reduce ratio
                max_ratio = mid_ratio
                mid_ratio = (min_ratio + mid_ratio) / 2
            else:
                # Too few faces, increase ratio
                min_ratio = mid_ratio
                mid_ratio = (mid_ratio + max_ratio) / 2
        
        print(f"  Binary search complete. Best result: {best_face_count} faces (diff: {best_difference})")
        
        # If we didn't reach tolerance, use best result and try edge collapse or edge splits
        # Reset to best result first
        obj.data = backup_mesh.copy()
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        modifier.ratio = best_ratio
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier="Decimate")
        
        # Now try to add or remove individual faces to reach exact count
        current_faces = len(obj.data.polygons)
        if abs(current_faces - target_faces) > tolerance:
            print(f"  Attempting fine adjustment from {current_faces} to {target_faces} faces...")
            
            # Switch to Edit mode
            bpy.ops.object.mode_set(mode='EDIT')
            
            if current_faces > target_faces:
                # Need to reduce faces
                face_diff = current_faces - target_faces
                print(f"  Need to remove {face_diff} faces")
                
                # Select some faces to dissolve
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # Select specific faces to remove - try to select evenly distributed faces
                stride = current_faces // face_diff
                for i in range(0, face_diff):
                    if i * stride < len(obj.data.polygons):
                        obj.data.polygons[i * stride].select = True
                
                bpy.ops.object.mode_set(mode='EDIT')
                # Dissolve the selected faces
                bpy.ops.mesh.dissolve_faces()
                
            else:
                # Need to add faces
                # This is harder - we'll try to split some edges
                face_diff = target_faces - current_faces
                print(f"  Need to add {face_diff} faces")
                
                # Select some edges to split - we'll select random edges
                bpy.ops.mesh.select_all(action='DESELECT')
                bpy.ops.object.mode_set(mode='OBJECT')
                
                # Select random edges
                edge_count = len(obj.data.edges)
                edges_to_select = min(face_diff * 2, edge_count // 4)  # Don't select too many edges
                
                edges = list(range(edge_count))
                random.shuffle(edges)
                
                for i in range(edges_to_select):
                    if i < len(edges):
                        obj.data.edges[edges[i]].select = True
                
                bpy.ops.object.mode_set(mode='EDIT')
                # Subdivide the selected edges
                bpy.ops.mesh.subdivide(number_cuts=1)
            
            # Return to Object mode
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Check new count
            new_face_count = len(obj.data.polygons)
            print(f"  After fine adjustment: {new_face_count} faces (target: {target_faces}, diff: {abs(new_face_count - target_faces)})")

    # Adjust face count
    if exact_count:
        # Try to get exactly the target number of faces
        adjust_face_count_exact(obj, target_faces)
    else:
        # Use the standard method if exact count is not required
        if current_faces > target_faces:
            # Reduce face count
            print(f"Reducing faces: {current_faces} -> {target_faces}")
            ratio = target_faces / current_faces
            
            # Apply decimate modifier
            modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            modifier.ratio = ratio
            modifier.use_collapse_triangulate = True
            bpy.ops.object.modifier_apply(modifier="Decimate")
            
            print(f"After decimation: {len(obj.data.polygons)} faces, {len(obj.data.vertices)} vertices")
        else:
            # Increase face count
            print(f"Increasing faces: {current_faces} -> {target_faces}")
            
            # First subdivide
            modifier = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            # Calculate subdivision level needed (each level multiplies by ~4)
            target_ratio = target_faces / current_faces
            levels = 1
            while (4 ** levels) < target_ratio and levels < 5:  # Limit to 5 levels
                levels += 1
            
            modifier.levels = levels
            bpy.ops.object.modifier_apply(modifier="Subdivision")
            
            # Then decimate if needed
            current_faces = len(obj.data.polygons)
            if current_faces > target_faces:
                ratio = target_faces / current_faces
                modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
                modifier.ratio = ratio
                modifier.use_collapse_triangulate = True
                bpy.ops.object.modifier_apply(modifier="Decimate")
            
            print(f"After subdivision and decimation: {len(obj.data.polygons)} faces, {len(obj.data.vertices)} vertices")

    # One final shrinkwrap to ensure accuracy
    shrinkwrap_mod = obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_mod.target = original_obj
    shrinkwrap_mod.wrap_method = 'NEAREST_SURFACEPOINT'
    shrinkwrap_mod.offset = 0.0001
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")

    # Final cleanup
    bpy.ops.object.shade_smooth()  # Apply smooth shading
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)  # Remove any new duplicate vertices
    bpy.ops.mesh.normals_make_consistent(inside=False)  # Fix normals one last time
    bpy.ops.object.mode_set(mode='OBJECT')

    # Delete the original mesh copy
    bpy.data.objects.remove(original_obj)
    print("Removed original mesh copy")

    # Get final stats
    final_vertices = len(obj.data.vertices)
    final_faces = len(obj.data.polygons)
    print(f"===== FINAL MESH STATS =====")
    print(f"Vertices: {final_vertices}")
    print(f"Faces: {final_faces}")
    print(f"Target Faces: {target_faces}")
    print(f"Difference: {final_faces - target_faces} faces")
    print(f"=============================")

    # Append to the stats file
    with open('mesh_stats.txt', 'a') as stats_file:
        stats_file.write(f"FINAL_VERTICES={final_vertices}\n")
        stats_file.write(f"FINAL_FACES={final_faces}\n")
        stats_file.write(f"TARGET_FACES={target_faces}\n")
        stats_file.write(f"DIFFERENCE={final_faces - target_faces}\n")

    # Export mesh
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        print(f"Exporting to {output_file}")
        
        # Try built-in exporters first
        try:
            bpy.ops.export_scene.obj(filepath=output_file, use_selection=True, use_triangles=True)
            print("Exported using export_scene.obj")
        except Exception as e:
            print(f"export_scene.obj error: {str(e)}")
            try:
                bpy.ops.wm.obj_export(filepath=output_file, export_selected_objects=True)
                print("Exported using wm.obj_export")
            except Exception as e2:
                print(f"wm.obj_export error: {str(e2)}")
                
                # Manual export as last resort
                print("Using manual export method")
                # Get mesh data
                mesh = obj.data
                
                with open(output_file, 'w') as f:
                    # Write header
                    f.write("# OBJ file created by Blender script\n")
                    
                    # Write vertices
                    for v in mesh.vertices:
                        f.write(f"v {v.co.x:.6f} {v.co.y:.6f} {v.co.z:.6f}\n")
                    
                    # Write faces
                    for poly in mesh.polygons:
                        verts = [f"{i+1}" for i in poly.vertices]
                        f.write(f"f {' '.join(verts)}\n")
        
        print(f"Export successful: {output_file}")
    except Exception as e:
        print(f"All export methods failed: {str(e)}")
        
        # Try to save a Blender file as a last resort
        try:
            blend_file = output_file.replace(".obj", ".blend")
            bpy.ops.wm.save_as_mainfile(filepath=blend_file)
            print(f"Saved as Blender file instead: {blend_file}")
        except Exception as be:
            print(f"Blend file save also failed: {str(be)}")
            sys.exit(1)

    print("Blender Python script completed")

if __name__ == "__main__":
    main()