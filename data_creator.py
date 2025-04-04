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
        grid_size = 192  # grid size 설정
        min_bound = -1
        max_bound = 1
        x = np.linspace(min_bound, max_bound, grid_size)
        y = np.linspace(min_bound, max_bound, grid_size)
        z = np.linspace(min_bound, max_bound, grid_size)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        xyz = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)  # (grid_size^3, 3)

        count = 0

        # 4. 각 점에 대해 Signed Distance 계산
        # trimesh의 signed_distance 함수를 사용 (mesh의 내부면 음수, 외부면 양수)
        for file in sorted(os.listdir(self.data_path)):
            if count == 10:  # 샘플 10개만
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
            
            # centralize the mesh
            mesh.vertices -= mesh.centroid
            assert np.allclose(mesh.centroid, 0)
            
            box_mesh = trimesh.creation.box(bounds=mesh.bounds)
            box_mesh = box_mesh.subdivide()
            box_mesh.vertices = mesh.nearest.on_surface(box_mesh.vertices)[0]
                
            
            if map_z_to_y:
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]  # z와 y를 교환

            # 5. 각 grid_point에 대해 SDF 값을 계산
            sdf, *_ = pcu.signed_distance_to_mesh(xyz, mesh.vertices, mesh.faces)
            
            # latent vector와 함께 저장될 (x, y, z, sdf) 데이터를 생성
            data = np.hstack([xyz, sdf[:, np.newaxis]])
            
            start = time.time()
            
            # Reshape SDF values back to 3D grid for marching cubes
            sdf_grid = sdf.reshape(grid_size, grid_size, grid_size)
            
            # Apply marching cubes to reconstruct the mesh
            vertices, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0)
            
            # Scale vertices back to original space
            vertices = vertices * (max_bound - min_bound) / (grid_size - 1) + min_bound
            
            # Create a new mesh from the marching cubes result
            reconstructed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            end = time.time()
            print(f"Time taken: {end - start} seconds")
            
            # Save both the SDF data and the reconstructed mesh
            # np.save(f"{count}.npy", data)
            reconstructed_mesh.export(f"{count}_reconstructed.obj")
            mesh.export(f"{count}_mesh.obj")
            
            count += 1
            
    
    
if __name__ == "__main__":
    data_creator = DataCreator(data_path=os.path.join("data/03001627"))
    data_creator.create()
