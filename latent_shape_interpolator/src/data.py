import os
import sys
import torch
import random
import shutil
import trimesh
import numpy as np
import multiprocessing
import point_cloud_utils as pcu

from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset, Subset

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration


class DataCreator:
    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def _map_mesh_z_to_y(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mapped_mesh = mesh.copy()
        mapped_mesh.vertices[:, [1, 2]] = mapped_mesh.vertices[:, [2, 1]]

        return mapped_mesh

    def _centralize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        centralized_mesh = mesh.copy()

        centralized_mesh.vertices -= centralized_mesh.vertices.mean(axis=0)
        assert np.allclose(centralized_mesh.vertices.mean(axis=0), 0)

        return centralized_mesh

    def _orient_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        oriented_mesh = mesh.copy()

        # compute obb information
        transform, (w, h) = trimesh.bounds.oriented_bounds_2D(oriented_mesh.vertices[:, :2])

        # define the 2d bounding box from the above information
        bounding_box_2d = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

        # apply the transformation to get the oriented bounding box
        oriented_bounding_box_2d = bounding_box_2d @ transform[:2, :2]

        # sort the obb vertices by angle
        center = np.mean(oriented_bounding_box_2d, axis=0)
        angles = np.arctan2(oriented_bounding_box_2d[:, 1] - center[1], oriented_bounding_box_2d[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        oriented_bounding_box_2d = oriented_bounding_box_2d[sorted_indices]

        # orient the mesh
        angle_to_orient = -np.arctan2(*(oriented_bounding_box_2d[2] - oriented_bounding_box_2d[1]))
        rotation_matrix = np.array(
            [
                [np.cos(angle_to_orient), -np.sin(angle_to_orient), 0],
                [np.sin(angle_to_orient), np.cos(angle_to_orient), 0],
                [0, 0, 1],
            ]
        )

        oriented_mesh.vertices = oriented_mesh.vertices @ rotation_matrix

        return oriented_mesh

    def _compute_latent_shape(self, mesh: trimesh.Trimesh) -> np.ndarray:
        box_mesh = trimesh.creation.box(bounds=mesh.bounds)
        box_mesh.vertices = box_mesh.vertices @ self.configuration.SCALE_MATRIX
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

                locations, *_ = mesh.ray.intersects_location(ray_origins=[ray_origin], ray_directions=[ray_direction])

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

        assert box_mesh_subdivided.vertices.shape == (self.configuration.NUM_LATENT_POINTS, 3)

        return box_mesh_subdivided.vertices, box_mesh_subdivided.faces

    def _sample_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        # sample surface points
        surface_points_sampled, _ = trimesh.sample.sample_surface(mesh, self.configuration.N_SURFACE_SAMPLING)

        # sample surface points with noise
        surface_points_noisy_sampled, _ = trimesh.sample.sample_surface(
            mesh, self.configuration.N_SURFACE_NOISY_SAMPLING
        )

        surface_points_noisy_sampled += np.random.uniform(
            -self.configuration.LATENT_POINTS_NOISE,
            self.configuration.LATENT_POINTS_NOISE,
            size=surface_points_noisy_sampled.shape,
        )

        # sample volume points
        volume_points_sampled = np.random.uniform(
            self.configuration.MIN_BOUND, self.configuration.MAX_BOUND, size=(self.configuration.N_VOLUME_SAMPLING, 3)
        )

        # check if the number of sampled points is same as N_TOTAL_SAMPLING
        assert (
            surface_points_sampled.shape[0] + surface_points_noisy_sampled.shape[0] + volume_points_sampled.shape[0]
            == self.configuration.N_TOTAL_SAMPLING
        )

        return np.concatenate([surface_points_sampled, surface_points_noisy_sampled, volume_points_sampled])

    def _create_one(self, file: str, map_z_to_y: bool) -> bool:
        print(f"processing {file}", flush=True)

        # load mesh
        mesh = trimesh.load(file)

        if isinstance(mesh, trimesh.Scene):
            geo_list = []
            for g in mesh.geometry.values():
                geo_list.append(g)
            mesh = trimesh.util.concatenate(geo_list)

        if not mesh.is_watertight:
            print(f"{file} is not watertight, making it watertight...", flush=True)
            vertices, faces = pcu.make_mesh_watertight(
                mesh.vertices, mesh.faces, resolution=self.configuration.WATERTIGHT_RESOLUTION
            )

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            if not mesh.is_watertight:
                print(f"{file} is not watertight, skipping...", flush=True)
                return False

        # map z to y
        if map_z_to_y:
            mesh = self._map_mesh_z_to_y(mesh)

        # centralize the mesh to the origin
        mesh = self._centralize_mesh(mesh)

        # orient the mesh
        mesh = self._orient_mesh(mesh)

        # sample points
        xyz = self._sample_points(mesh)

        # compute latent points
        latent_shape, faces = self._compute_latent_shape(mesh)

        # compute sdf values
        sdf, *_ = pcu.signed_distance_to_mesh(xyz, mesh.vertices, mesh.faces)

        np.savez(
            file.replace(self.configuration.DATA_NAME_OBJ, f"{file.split('/')[-3]}.npz"),
            xyz=xyz,
            sdf=sdf,
            latent_shape=latent_shape,
            faces=faces,
        )

        return True

    def create(
        self,
        map_z_to_y: bool = True,
        use_multiprocessing: bool = True,
        copy_obj: bool = False,
        slicer: Optional[int] = None,
    ) -> None:
        tasks: List[Tuple[str, bool]]
        tasks = []

        data_list = sorted(os.listdir(self.configuration.DATA_PATH))
        if isinstance(slicer, int):
            data_list = data_list[:slicer]

        data_list = [
            os.path.join(
                self.configuration.DATA_PATH, file, self.configuration.DATA_NAME, self.configuration.DATA_NAME_OBJ
            )
            for file in data_list
        ]

        for file in data_list:
            tasks.append((file, map_z_to_y))

        print("creating...", flush=True)

        if use_multiprocessing:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.starmap(self._create_one, tasks)
        else:
            results = []
            for task in tasks:
                results.append(self._create_one(*task))

        print("assigning class number...", flush=True)

        if os.path.exists(self.configuration.DATA_PATH_PROCESSED):
            shutil.rmtree(self.configuration.DATA_PATH_PROCESSED)

        os.makedirs(self.configuration.DATA_PATH_PROCESSED, exist_ok=False)

        class_number = 0
        for file, result in zip(data_list, results):
            if result:
                data_path = file.replace(self.configuration.DATA_NAME_OBJ, f"{file.split('/')[-3]}.npz")

                data = np.load(data_path)
                data_with_class_number = {**data, "class_number": class_number}
                np.savez(data_path, **data_with_class_number)

                class_number_dir = os.path.join(self.configuration.DATA_PATH_PROCESSED, f"{class_number}")

                os.makedirs(class_number_dir, exist_ok=True)

                shutil.move(data_path, os.path.join(class_number_dir, os.path.basename(data_path)))

                if copy_obj:
                    shutil.copy(file, os.path.join(class_number_dir, os.path.basename(data_path).replace("npz", "obj")))

                class_number += 1

        print("done!", flush=True)


class SDFDataset(Dataset):
    def __init__(self, data_dir: str, configuration: Configuration, data_slicer: Optional[int] = None):
        self.data_dir = data_dir
        self.configuration = configuration

        if not isinstance(data_slicer, int):
            self.data_slicer = len(os.listdir(data_dir))

        data_list = os.listdir(self.configuration.DATA_PATH_PROCESSED)
        data_list = sorted(data_list, key=lambda folder: int(folder))

        self.data_path = []
        for folder in data_list:
            each_data_dir = os.path.join(self.configuration.DATA_PATH_PROCESSED, folder)
            each_data_name, *_ = os.listdir(each_data_dir)
            assert each_data_name.endswith(".npz")

            each_data_path = os.path.join(each_data_dir, each_data_name)
            if os.path.exists(each_data_path):
                self.data_path.append(each_data_path)

            if len(self.data_path) == data_slicer:
                break

        self.num_classes = len(self.data_path)

        self.total_length = self.configuration.N_TOTAL_SAMPLING * len(self.data_path)
        self.cumulative_length = [0] + [
            self.configuration.N_TOTAL_SAMPLING * i for i in range(1, len(self.data_path) + 1)
        ]

        self.max_sdf = -np.inf
        self.min_sdf = np.inf
        for data in self.data_path:
            sdf = np.load(data)["sdf"]
            self.max_sdf = max(self.max_sdf, sdf.max())
            self.min_sdf = min(self.min_sdf, sdf.min())

        assert self.max_sdf != -np.inf
        assert self.min_sdf != np.inf

        self.train_dataset = None
        self.train_dataloader = None
        self.validation_dataset = None
        self.validation_dataloader = None

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx = None
        for file_idx, cumulative_length in enumerate(self.cumulative_length):
            if _idx < cumulative_length:
                file_idx -= 1
                break

        assert file_idx is not None

        data = np.load(self.data_path[file_idx])

        idx = _idx % self.configuration.N_TOTAL_SAMPLING

        xyz = torch.tensor(data["xyz"][idx], dtype=torch.float32)
        sdf = torch.tensor(data["sdf"][idx], dtype=torch.float32)
        class_number = torch.tensor(data["class_number"], dtype=torch.long)
        latent_shape = torch.tensor(data["latent_shape"], dtype=torch.float32)
        faces = torch.tensor(data["faces"], dtype=torch.long)

        xyz = xyz.to(self.configuration.DEVICE)
        sdf = sdf.to(self.configuration.DEVICE)
        class_number = class_number.to(self.configuration.DEVICE)
        latent_shape = latent_shape.to(self.configuration.DEVICE)
        faces = faces.to(self.configuration.DEVICE)

        return xyz, sdf, class_number, latent_shape, faces

    @staticmethod
    def create_dataset(
        data_dir: str,
        configuration: Configuration,
        data_slicer: Optional[int] = None,
    ) -> Dict:
        """"""

        sdf_dataset = SDFDataset(data_dir, configuration, data_slicer)

        train_size_per_class = int(configuration.N_TOTAL_SAMPLING * configuration.TRAIN_VALIDATION_RATIO[0])
        val_size_per_class = configuration.N_TOTAL_SAMPLING - train_size_per_class

        assert train_size_per_class + val_size_per_class == configuration.N_TOTAL_SAMPLING

        train_subsets = []
        validation_subsets = []
        latent_shapes = []

        for class_idx in range(sdf_dataset.num_classes):
            start_idx = class_idx * configuration.N_TOTAL_SAMPLING
            end_idx = (class_idx + 1) * configuration.N_TOTAL_SAMPLING

            class_indices = list(range(start_idx, end_idx))
            random.shuffle(class_indices)

            train_indices = class_indices[:train_size_per_class]
            val_indices = class_indices[train_size_per_class:]

            train_subsets.append(Subset(sdf_dataset, train_indices))
            validation_subsets.append(Subset(sdf_dataset, val_indices))
            latent_shapes.append(sdf_dataset[start_idx][3])

        assert len(latent_shapes) == sdf_dataset.num_classes
        assert len(sdf_dataset) == sdf_dataset.num_classes * configuration.N_TOTAL_SAMPLING

        train_dataset = torch.utils.data.ConcatDataset(train_subsets)
        validation_dataset = torch.utils.data.ConcatDataset(validation_subsets)

        train_dataloader = DataLoader(train_dataset, batch_size=configuration.BATCH_SIZE, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=configuration.BATCH_SIZE, shuffle=False)

        sdf_dataset.train_dataset = train_dataset
        sdf_dataset.train_dataloader = train_dataloader
        sdf_dataset.validation_dataset = validation_dataset
        sdf_dataset.validation_dataloader = validation_dataloader
        sdf_dataset.latent_shapes = torch.stack(latent_shapes)

        return sdf_dataset


if __name__ == "__main__":
    configuration = Configuration()
    data_creator = DataCreator(configuration=configuration)
    data_creator.create(map_z_to_y=True, use_multiprocessing=True, copy_obj=True, slicer=5)
