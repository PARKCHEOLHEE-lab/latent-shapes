import os
import trimesh
import traceback
import numpy as np
import multiprocessing

from .config import Configuration


def _compute_bounds(file_path: str):
    try:
        print(file_path)
        mesh = trimesh.load(file_path)
        bounds = mesh.bounds

        return bounds

    except Exception as e:
        print(f"{file_path}: {e}")
        traceback.print_exc()
        return None


def main():
    configuration = Configuration()
    data_path = configuration.DATA_PATH

    file_path_list = [
        os.path.join(data_path, path, os.path.join(configuration.DATA_NAME, configuration.DATA_NAME_OBJ))
        for path in os.listdir(data_path)
    ]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(_compute_bounds, file_path_list)

    valid_bounds = [bounds for bounds in results if bounds is not None]

    bounds = np.array(valid_bounds)

    min_x, min_y, min_z = np.min(bounds[:, 0, :], axis=0)
    max_x, max_y, max_z = np.max(bounds[:, 1, :], axis=0)

    print(
        f"""
        {"-" * 50}
        processed {len(valid_bounds)} files successfully"
        failed: {len(file_path_list) - len(valid_bounds)} files
        bounds:
            min: ({min_x:.10f}, {min_y:.10f}, {min_z:.10f})
            max: ({max_x:.10f}, {max_y:.10f}, {max_z:.10f})
        {"-" * 50}
        """
    )

    # --------------------------------------------------
    # processed 6778 files successfully"
    # failed: 0 files
    # bounds:
    #     min: (-0.6149350000, -0.7674950000, -0.6662140000)
    #     max: (0.6443440000, 0.7928820000, 0.5812350000)
    # --------------------------------------------------


if __name__ == "__main__":
    main()
