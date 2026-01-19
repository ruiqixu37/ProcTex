import numpy as np
import torch
import logging
from scipy.spatial import KDTree

def get_scale(points):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return np.sqrt(np.sum(max_bound - min_bound))

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm numpy array of points
        dst: Nxm numpy array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    tree = KDTree(dst)
    distances, indices = tree.query(src)
    return distances, indices

def best_fit_transform(source, target):
    assert source.shape == target.shape
    m = source.shape[1]

    # Translate points to their centroids
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    demeaned_source = source - centroid_source
    demeaned_target = target - centroid_target

    # Compute the rotation matrix
    H = np.dot(demeaned_target.T, demeaned_source)
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.identity(m + 1), np.full(m, -1), np.full((m, m), -1), np.full(m, -1)
    R = np.dot(U, Vt)

    # Special reflection case (handle improper rotation)
    if np.linalg.det(R) < 0:
        diag = np.identity(m)
        diag[m - 1, m - 1] = -1
        Vt = np.dot(Vt, diag)
        R = np.dot(U, Vt)

    # Compute translation
    t = centroid_target - np.dot(R, centroid_source)

    # Construct the homogeneous transformation matrix
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def icp(source_component, target_component, global_symmetry_axis, max_iterations=200, tolerance=0.001, max_scale=3, allow_scaling=False, return_rotation=False): 
    m = 3

    # Align scale with pca
    scale_factors = np.sqrt(target_component['pca_variance'] / source_component['pca_variance'])
    if np.any(scale_factors > 3) or np.any(scale_factors < 1/3):
        if return_rotation:
            return np.identity(4), np.inf, -1, np.full((m, m), -1)
        else:
            return np.identity(4), np.inf, -1

    source_vertices = source_component['vertices_ori'] - source_component['center_mass']
    target_vertices = target_component['vertices_ori'] - target_component['center_mass']
    pca_homogeneous = np.eye(4)
    if allow_scaling:
        pca_homogeneous[:3, :3] = source_component['pca_axis'].T @ np.diag(scale_factors) @ source_component['pca_axis']
        source_vertices = (source_vertices @ source_component['pca_axis'].T) @ np.diag(scale_factors) @ source_component['pca_axis']

    # Check if the component is fully on one side of symmetry
    reflection_matrix = np.eye(4)
    symmetric_axis_idx = np.where(global_symmetry_axis == -1)[0]
    for axis_idx in symmetric_axis_idx:
        if isinstance(source_component['vertices_ori'], torch.Tensor):
            source_v_np = source_component['vertices_ori'].cpu().numpy()
            target_v_np = target_component['vertices_ori'].cpu().numpy()
        else:
            source_v_np = source_component['vertices_ori']
            target_v_np = target_component['vertices_ori']
        if np.all(source_v_np[:, axis_idx] >= 0) and np.all(target_v_np[:, axis_idx] <= 0) or np.all(source_v_np[:, axis_idx] <= 0) and np.all(target_v_np[:, axis_idx] >= 0):
            source_vertices[:, axis_idx] *= -1
            reflection_matrix[axis_idx, axis_idx] = -1

    # Make points homogeneous
    src = np.ones((source_vertices.shape[0], m + 1))
    dst = np.ones((target_vertices.shape[0], m + 1))
    src[:, :m] = np.copy(source_vertices)
    dst[:, :m] = np.copy(target_vertices)

    target_scale = np.linalg.norm(np.max(source_vertices, axis=0) - np.min(source_vertices), axis=0)

    prev_error = np.inf
    for i in range(max_iterations):
        distances_src_to_dst, indices_src_to_dst = nearest_neighbor(src[:, :m], dst[:, :m]) 
        distances_dst_to_src, _ = nearest_neighbor(dst[:, :m], src[:, :m])

        T, R, t = best_fit_transform(src[:, :m], dst[indices_src_to_dst, :m])

        src = np.dot(T, src.T).T

        chamfer_distance = 0.5 * np.mean(distances_src_to_dst ** 2) + 0.5 * np.mean(distances_dst_to_src ** 2)
        chamfer_distance /= target_scale
        if np.abs(prev_error - chamfer_distance) < tolerance:
            break
        prev_error = chamfer_distance

    T, R, t = best_fit_transform(source_vertices, src[:, :m])

    T_final = reflection_matrix @ pca_homogeneous @ T
    T_final[:3, 3] += target_component['center_mass'] - ((T_final[:3, :3] @ source_component['center_mass']) + T_final[:3, 3])

    if return_rotation:
        return T_final, chamfer_distance, i, R
    else:
        return T_final, chamfer_distance, i

if __name__ == "__main__":
    import os
    import trimesh
    import numpy as np

    # Define file paths for the two meshes
    mesh_file_1 = "C:\\Users\\AAA\\Documents\\CompInpaint\\logs\\cakes\\out_component_0.obj"  # Replace with the actual path to the first mesh
    mesh_file_2 = "C:\\Users\\AAA\\Documents\\CompInpaint\\logs\\cakes\\out_component_9.obj"  # Replace with the actual path to the second mesh

    # Load the two meshes
    mesh1 = trimesh.load(mesh_file_1)
    mesh2 = trimesh.load(mesh_file_2)

    # Extract vertices from the meshes
    A = mesh1.vertices
    B = mesh2.vertices

    # Run the ICP with reflections function
    T, scale_factors, distances, iterations = icp_with_reflections(A, B, max_iterations=10000, tolerance=1e-20, allow_scaling=True)

    print("Transformation Matrix (T):")
    print(T)
    print("Scale Factors:", scale_factors)
    print("Mean Distance:", np.mean(distances))
    print("Number of Iterations:", iterations)

    # Transform mesh1 to align with mesh2
    new_A = np.dot(T, np.concatenate([A.T, np.ones((1, A.shape[0]))], axis=0)).T[:, :3]
    aligned_mesh1 = trimesh.Trimesh(vertices=new_A, faces=mesh1.faces)

    # Visualize both meshes in the same scene
    scene = trimesh.Scene()
    scene.add_geometry(aligned_mesh1, node_name="Aligned Mesh 1", geom_name="Aligned Mesh 1")
    scene.add_geometry(mesh2, node_name="Mesh 2", geom_name="Mesh 2")

    # Display the scene
    scene.show()