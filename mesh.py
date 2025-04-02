import subprocess
import os
import sys
import shutil

def remesh_with_blender(input_file, output_file, target_faces=12324, exact_count=True):
    # Blender path
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Check if blender_remesh.py exists
    script_path = "blender_remesh.py"
    if not os.path.exists(script_path):
        print(f"Error: Blender script not found: {script_path}")
        return False
    
    # Get absolute paths
    abs_input = os.path.abspath(input_file)
    abs_output = os.path.abspath(output_file)
    
    print(f"Input file: {abs_input}")
    print(f"Output file: {abs_output}")
    print(f"Target faces: {target_faces}")
    print(f"Exact count: {exact_count}")
    
    # Set environment variables for passing parameters
    env = os.environ.copy()
    env['BL_INPUT_FILE'] = abs_input
    env['BL_TARGET_FACES'] = str(target_faces)
    env['BL_OUTPUT_FILE'] = abs_output
    env['BL_EXACT_COUNT'] = str(exact_count)
    
    # Blender command
    cmd = [
        blender_exe,
        "--background",
        "--python", script_path
    ]
    
    # Run Blender
    print("Running Blender...")
    process = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Display all output
    print(process.stdout)
    if process.stderr:
        print("Error output:")
        print(process.stderr)
    
    # Read stats from the file
    mesh_stats = {}
    if os.path.exists('mesh_stats.txt'):
        with open('mesh_stats.txt', 'r') as stats_file:
            for line in stats_file:
                if '=' in line:
                    key, value = line.strip().split('=')
                    mesh_stats[key] = value
        
        # Show clear summary
        print("\n" + "="*50)
        print("MESH PROCESSING SUMMARY")
        print("="*50)
        if 'ORIGINAL_VERTICES' in mesh_stats:
            print(f"Original vertices: {mesh_stats['ORIGINAL_VERTICES']}")
        if 'ORIGINAL_FACES' in mesh_stats:
            print(f"Original faces:    {mesh_stats['ORIGINAL_FACES']}")
        print("-"*50)
        if 'FINAL_VERTICES' in mesh_stats:
            print(f"Final vertices:    {mesh_stats['FINAL_VERTICES']}")
        if 'FINAL_FACES' in mesh_stats:
            print(f"Final faces:       {mesh_stats['FINAL_FACES']} (Target: {target_faces})")
        if 'DIFFERENCE' in mesh_stats:
            diff = int(mesh_stats['DIFFERENCE'])
            print(f"Difference:        {diff} faces ({abs(diff)/int(target_faces)*100:.2f}%)")
        print("="*50)
        
        # Clean up
        os.remove('mesh_stats.txt')
    else:
        print("\nCould not find mesh stats file. Check the output above for face and vertex counts.")
    
    # Check if result file exists
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Result file created: {output_file} ({file_size} bytes)")
        if file_size == 0:
            print("WARNING: Output file is empty!")
            return False
        return True
    else:
        print(f"Error: Result file was not created: {output_file}")
        # Check for alternative files
        blend_file = output_file.replace(".obj", ".blend")
        if os.path.exists(blend_file):
            print(f"Found Blender file instead: {blend_file}")
            return True
        return False


# Calculate approximate target faces from desired vertices
# For triangular meshes, faces ≈ 2 * vertices
target_vertices = 6162
target_faces = target_vertices * 2

# Example usage
input_file = r"data\03001627\1a6f615e8b1b5ae4dbbc9440457e303e\models\model_normalized.obj"
output_file = "remeshed_model_exact.obj" 

print(f"Current working directory: {os.getcwd()}")
print(f"Target vertices: {target_vertices} (approximately {target_faces} faces)")

result = remesh_with_blender(input_file, output_file, target_faces, exact_count=True)

if result:
    print("Process completed successfully.")
else:
    print("Process failed.")