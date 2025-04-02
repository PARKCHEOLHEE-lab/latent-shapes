import open3d as o3d


obj_path = r"data\03001627\1a6f615e8b1b5ae4dbbc9440457e303e\models\model_normalized.obj"

mesh = o3d.io.read_triangle_mesh(obj_path)


mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(20000)

radii = [0.001, 0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([rec_mesh])


# with o3d.utility.VerbosityContextManager(
#     o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=9)
    
# print(mesh)
# o3d.visualization.draw_geometries([mesh])

