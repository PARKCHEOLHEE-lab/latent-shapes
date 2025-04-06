import time
import os
import trimesh
import numpy as np
import point_cloud_utils as pcu
from skimage import measure

class DataCreator:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def create(self, map_z_to_y=True):
        grid_size = 192
        min_bound = -1
        max_bound = 1
        x = np.linspace(min_bound, max_bound, grid_size)
        y = np.linspace(min_bound, max_bound, grid_size)
        z = np.linspace(min_bound, max_bound, grid_size)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


        class_number = 0
        while True:
            break

        count = 0

        data_list = sorted(os.listdir(self.data_path))
        for file in data_list:
            if count == 10:
                break

            obj_path = os.path.join(self.data_path, file, "models", "model_normalized.obj")

            mesh = trimesh.load(obj_path)
            if isinstance(mesh, trimesh.Scene):
                geo_list = []
                for g in mesh.geometry.values():
                    geo_list.append(g)
                mesh = trimesh.util.concatenate(geo_list)

            if not mesh.is_watertight:
                vertices, faces = pcu.make_mesh_watertight(mesh.vertices, mesh.faces, resolution=50000)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                if not mesh.is_watertight:
                    continue
            
            # map z to y
            if map_z_to_y:
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]
            
            # centralize the mesh
            mesh.vertices -= mesh.centroid
            assert np.allclose(mesh.centroid, 0)
            
            box_mesh = trimesh.creation.box(bounds=mesh.bounds)
            box_mesh.vertices = box_mesh.vertices @ np.array([[0.95, 0, 0], [0, 0.95, 0], [0, 0, 1]])
            box_mesh_subdivided = box_mesh.subdivide()
            
            (min_x, min_y, min_z), (max_x, max_y, max_z) = box_mesh.bounds
            
            nearest_indices = []
            for i, vertex in enumerate(box_mesh_subdivided.vertices):
                is_bottom_edge_midpoint = (
                    np.allclose(vertex, (np.array([min_x, min_y, min_z]) + np.array([max_x, min_y, min_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, max_y, min_z]) + np.array([max_x, max_y, min_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, min_y, min_z]) + np.array([max_x, max_y, min_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, min_y, min_z]) + np.array([min_x, max_y, min_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([max_x, min_y, min_z]) + np.array([max_x, max_y, min_z])) * 0.5)
                )
                
                is_top_edge_midpoint = (
                    np.allclose(vertex, (np.array([min_x, min_y, max_z]) + np.array([max_x, min_y, max_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, max_y, max_z]) + np.array([max_x, max_y, max_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, min_y, max_z]) + np.array([max_x, max_y, max_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([min_x, min_y, max_z]) + np.array([min_x, max_y, max_z])) * 0.5) 
                    or np.allclose(vertex, (np.array([max_x, min_y, max_z]) + np.array([max_x, max_y, max_z])) * 0.5)
                )
                
                if is_bottom_edge_midpoint or is_top_edge_midpoint:
                    ray_origin = vertex
                    ray_direction = np.array([0, 0, 1])
                    
                    # flip ray_direction
                    if is_top_edge_midpoint:
                        ray_direction *= -1
                    
                    locations, *_ = mesh.ray.intersects_location(
                        ray_origins=[ray_origin], 
                        ray_directions=[ray_direction]
                    )
                    
                    # if the ray intersects the mesh, relocate the vertex to the intersection point
                    if len(locations) > 0:
                        new_vertex = locations[0]
                        box_mesh_subdivided.vertices[i] = new_vertex
                    else:
                        # if not, relocate the vertex to the nearest point on the mesh
                        nearest_indices.append(i)

                else:
                    nearest_indices.append(i)
                
            nearest_vertices = mesh.nearest.on_surface(box_mesh_subdivided.vertices[nearest_indices])[0]
            box_mesh_subdivided.vertices[nearest_indices] = nearest_vertices

            # compute sdf values
            sdf, *_ = pcu.signed_distance_to_mesh(xyz, mesh.vertices, mesh.faces)
            
            # latent vector와 함께 저장될 (x, y, z, sdf) 데이터를 생성
            # data = np.hstack([xyz, sdf[:, np.newaxis]])
            

            # Save both the SDF data and the reconstructed mesh
            # np.save(f"{count}.npy", data)
            # reconstructed_mesh.export(f"{count}_reconstructed.obj")
            # mesh.export(f"{count}_mesh.obj")
            
            count += 1
            
    
    
if __name__ == "__main__":
    data_creator = DataCreator(data_path=os.path.join("data/03001627"))
    data_creator.create()
