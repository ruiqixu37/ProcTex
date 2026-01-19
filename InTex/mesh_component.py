import os
import cv2
import numpy as np
import trimesh
import xatlas
import pywavefront
import logging

import torch

from utils.icp import icp, nearest_neighbor
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from utils.mesh_io import parse_obj_file
from scipy.special import ellipkinc, ellipk

import open3d as o3d
def visualize_open3d(source_pts, target_pts):
    """
    Visualizes the transformed source (`source_pts`) and target (`target_pts`) using Open3D.
    """
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    source_pcd.paint_uniform_color([0, 0, 1])  # Blue for transformed source
    target_pcd.paint_uniform_color([1, 0, 0])  # Red for target
    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    
def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def compare_mesh_topology(vertices1: np.ndarray, faces1: np.ndarray, vertices2: np.ndarray, faces2: np.ndarray):
    
    if len(vertices1) != len(vertices2) or len(faces1) != len(faces2):
        return False

    faces1_sorted = np.sort(faces1, axis=1)
    faces2_sorted = np.sort(faces2, axis=1)
    faces1_sorted = np.lexsort(faces1_sorted.T)
    faces2_sorted = np.lexsort(faces2_sorted.T)

    return np.array_equal(faces1_sorted, faces2_sorted)

def sample_points_on_mesh(vertices, faces, num_points):
    triangles = vertices[faces]

    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]
    cross_product = np.cross(v0, v1)
    areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    total_area = areas.sum()
    
    if np.isclose(total_area, 0):
        return np.zeros((num_points, 3))
    cdf = np.cumsum(areas) / total_area

    random_vals = np.random.rand(num_points)

    triangle_indices = np.searchsorted(cdf, random_vals)
    triangle_indices = np.clip(triangle_indices, 0, len(triangles) - 1) 

    sampled_triangles = triangles[triangle_indices]
    r1 = np.random.rand(num_points, 1)
    r2 = np.random.rand(num_points, 1)
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2

    sampled_points = (
        u * sampled_triangles[:, 0] +
        v * sampled_triangles[:, 1] +
        w * sampled_triangles[:, 2]
    )

    return sampled_points

class Mesh:
    def __init__(
        self,
        v=None,
        f=None,
        vn=None,
        fn=None,
        vt=None,
        ft=None,
        albedo=None,
        vc=None, # vertex color
        device=None,
    ):
        self.device = device
        self.v = v
        self.vn = vn
        self.vt = vt
        self.f = f
        self.fn = fn
        self.ft = ft

        self.ori_center = 0
        self.ori_scale = 1

        self.components = []
        self.albedo = None
        
        self.partial_groups = None # None means save all groups

    @classmethod
    def load(cls, path=None, remesh=False, resize=True, renormal=True, retex=False, device=None, **kwargs):

        if path.endswith(".obj"):
            mesh = cls.load_obj(path, device, **kwargs)

        # auto-normalize
        if resize:
            mesh.auto_size()

        # split mesh and generate sphere uv
        mesh.find_connected_components()
        # logging.debug(f"The mesh groups are: {mesh.groups}")

        # # auto-fix normal
        # if renormal or mesh.vn is None:
        #     mesh.auto_normal()
        #     print(f"[Mesh loading] vn: {mesh.vn.shape}, fn: {mesh.fn.shape}")

        # if retex or mesh.vt is None:
        #     mesh.auto_uv()

        return mesh

    # load from obj file
    @classmethod
    def load_obj(cls, path, device, albedo_path=None):
        assert os.path.splitext(path)[-1] == ".obj"

        mesh = cls()

        # device
        mesh.device = device
        if mesh.device is None:
            mesh.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj = parse_obj_file(path)
        mesh.v = np.array(obj['vertices'])
        mesh.v = mesh.v - np.mean(mesh.v, axis=0)
        if len(obj['faces'].shape) == 3:
            mesh.f = np.array(obj['faces'])[:, :, 0] - 1
        else:
            mesh.f = np.array(obj['faces']) - 1
        mesh.global_symmetry_axis = mesh.get_symmetry_axis(mesh.v)
        mesh.v = torch.tensor(mesh.v, dtype=torch.float32, device=mesh.device)
        mesh.f = torch.tensor(mesh.f, dtype=torch.int32, device=mesh.device)
        if obj['uvs'] is not None:
            mesh.vt = obj['uvs']
            mesh.vt = torch.tensor(mesh.vt, dtype=torch.float32, device=mesh.device)

        return mesh

    def get_symmetry_axis(self, vertices, tolerance=1e-4):
        symmetry_axis = np.array([1, 1, 1])
        vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0))

        for axis, idx in zip(['x', 'y', 'z'], range(3)):
            reflected = np.copy(vertices)
            reflected[:, idx] *= -1

            distances, _ = nearest_neighbor(reflected, vertices)

            if np.mean(distances) < tolerance:
                symmetry_axis[idx] = -1

        return symmetry_axis

    def find_connected_components(self):
        vertices = self.v.cpu().detach().numpy()
        faces = self.f.cpu().detach().numpy()
        uvs = self.vt.cpu().detach().numpy() if self.vt != None else self.vt
        face_components = self._find_connected_components(vertices, faces)
        pca = PCA(n_components=3)

        for idx, component_faces in enumerate(face_components):
            component_vertices, component_faces_new, component_uvs = self.create_submesh(vertices, faces, uvs, component_faces)
            
            # in procedural models, there will often be trivial components with ~0 volume (a single dot). Skip these.
            scale = np.linalg.norm(np.max(component_vertices, axis=0) - np.min(component_vertices, axis=0))
            if scale < 1e-3:
                continue

            center_mass = np.mean(component_vertices, axis=0) 
            vertices_centered = component_vertices - center_mass
            pca.fit(vertices_centered)

            component_mesh = {
                'vertices_ori': np.array(component_vertices),
                'faces_ori': np.array(component_faces_new),
                # 'sampled': sample_points_on_mesh(np.array(component_vertices), np.array(component_faces_new), 10000),
                'center_mass': center_mass,
                'pca_axis': pca.components_,
                "pca_variance": pca.explained_variance_,
                'uvs': None
            }
            self.components.append(component_mesh)

        self.groups = self.group_by_icp()
        
        logging.debug("Found %d connected components, and %d groups", len(self.components), len(self.groups))
                
    def _find_connected_components(self, vertices, faces):
        """parse the mesh into components based on connectivity"""
        adjacency_pairs = {}

        # Build adjacency list (vertices -> faces)
        for i, face in enumerate(faces):
            for j in range(3):
                vertex_idx = face[j]
                if vertex_idx not in adjacency_pairs:
                    adjacency_pairs[vertex_idx] = []
                # adjacency_pairs[edge].append(i)  # Store all faces sharing this edge
                adjacency_pairs[vertex_idx].append(i) # store all faces sharing this vertex

        visited = np.zeros(len(faces), dtype=bool)
        components = []

        # Traverse the faces to identify connected components
        for i in range(len(faces)):
            if visited[i]:
                continue

            stack = [i]
            component_faces = []

            while stack:
                face_index = stack.pop()
                if visited[face_index]:
                    continue

                visited[face_index] = True
                component_faces.append(face_index)

                for j in range(3):
                    # edge = tuple(sorted((faces[face_index][j], faces[face_index][(j + 1) % 3])))
                    vertex_idx = faces[face_index][j]
                    adjacent_faces = adjacency_pairs.get(vertex_idx, [])
                    for adj_face in adjacent_faces:
                        if not visited[adj_face]:
                            stack.append(adj_face)

            components.append(component_faces)

        return components

    def create_submesh(self, vertices, faces, uvs, component_faces):
        submesh_vertices = set()
        submesh_faces = []

        for face_index in component_faces:
            face = faces[face_index]
            submesh_faces.append(face)
            for vertex in face:
                submesh_vertices.add(vertex)

        sorted_vertices = sorted(submesh_vertices)
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_vertices)}

        new_vertices = [vertices[old_idx] for old_idx in sorted_vertices]
        new_faces = [[vertex_map[vertex] for vertex in face] for face in submesh_faces]
        if uvs is not None:
            new_uvs = [uvs[old_idx] for old_idx in sorted_vertices]
        else:
            new_uvs = None

        return np.array(new_vertices), np.array(new_faces), np.array(new_uvs) if new_uvs is not None else None

    def group_by_icp(self):
        """group the components based on ICP"""
        num_components = len(self.components)
        icp_T = {}
        groups = {}
        
        representatives = []
        
        for i in range(num_components):
            matched_group = None
            direct_assign = True
            for rep in representatives:
                if self.components[i]['vertices_ori'].shape[0] != self.components[rep]['vertices_ori'].shape[0] and self.components[i]['faces_ori'].shape[0] != self.components[rep]['faces_ori'].shape[0]:
                    continue
                # elif np.any((self.components[i]['pca_variance'] - self.components[rep]['pca_variance']) > 1e-3):
                #     print(f"ignoring {i} and {rep} due to PCA variance")
                #     continue
                elif compare_mesh_topology(self.components[i]['vertices_ori'], self.components[i]['faces_ori'], self.components[rep]['vertices_ori'], self.components[rep]['faces_ori']):
                    matched_group = rep
                    break
                else:
                    T, distances, iter = icp(
                        self.components[i],
                        self.components[rep],
                        self.global_symmetry_axis,
                        max_iterations=1000,
                        tolerance=1e-20,
                        allow_scaling=False
                    )

                    uv_vertices = (T[:3, :3] @ self.components[i]['vertices_ori'].T).T + T[:3, 3]
                    tree = cKDTree(uv_vertices)
                    _, indices = tree.query(self.components[rep]['vertices_ori'], k=1)
                    if len(np.unique(indices)) == len(indices) and np.min(indices) == 0 and np.max(indices) == len(indices) - 1:
                        self.components[i]['vertices_ori'] = self.components[i]['vertices_ori'][indices]
                        self.components[i]['faces_ori'] = self.components[rep]['faces_ori'][:, [0, 2, 1]]
                        matched_group = rep
                        direct_assign = False
                        break

            if matched_group is None:
                representatives.append(i)
                groups[i] = [i]
                self.components[i]['vertices'], self.components[i]['faces'], self.components[i]['uvs'], _, self.components[i]['vmapping'] = self.auto_uv(self.components[i]['vertices_ori'], self.components[i]['faces_ori'])
                self.components[i]['vertices'] = torch.tensor(self.components[i]['vertices'], dtype=torch.float32, device=self.device)
                self.components[i]['faces'] = torch.tensor(self.components[i]['faces'], dtype=torch.int32, device=self.device)
                self.components[i]['uvs'] = torch.tensor(self.components[i]['uvs'], dtype=torch.float32, device=self.device)
                self.components[i]['vn'] = self.auto_normal_component(self.components[i]['vertices'], self.components[i]['faces'])
            else:
                groups[matched_group].append(i)
                self.components[i]['vertices'] = torch.tensor(self.components[i]['vertices_ori'][self.components[matched_group]['vmapping']], dtype=torch.float32, device=self.device)
                if direct_assign:
                    self.components[i]['faces'] = self.components[matched_group]['faces'].clone()
                else:
                    self.components[i]['faces'] = self.components[matched_group]['faces'][:, [0, 2, 1]].clone()
                self.components[i]['uvs'] = self.components[matched_group]['uvs'].clone()
                self.components[i]['vn'] = self.auto_normal_component(self.components[i]['vertices'], self.components[i]['faces'])
            self.components[i]['sampled'] = sample_points_from_meshes(Meshes(verts=self.components[i]['vertices'].unsqueeze(0), faces=self.components[i]['faces'].unsqueeze(0)), 2000).squeeze(0).to(self.device)
        return groups

    # aabb
    def aabb(self):
        return torch.min(self.v, dim=0).values, torch.max(self.v, dim=0).values

    # unit size
    @torch.no_grad()
    def auto_size(self, size=1.6):
        vmin, vmax = self.aabb()
        self.ori_center = (vmax + vmin) / 2
        self.ori_scale = size / torch.max(vmax - vmin).item()
        self.v = (self.v - self.ori_center) * self.ori_scale

    def auto_normal_component(self, vertices, faces):
        i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
        v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to corresponding vertices
        vn = torch.zeros_like(vertices)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize vertex normals
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device)
        )
        vn = safe_normalize(vn)

        return vn

    def auto_normal(self):
        i0, i1, i2 = self.f[:, 0].long(), self.f[:, 1].long(), self.f[:, 2].long()
        v0, v1, v2 = self.v[i0, :], self.v[i1, :], self.v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        vn = torch.zeros_like(self.v)
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        vn = torch.where(
            dot(vn, vn) > 1e-20,
            vn,
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
        )
        vn = safe_normalize(vn)

        self.vn = vn
        self.fn = self.f

    
    def auto_uv(self, v_np, f_np):
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        atlas.generate(chart_options=chart_options)
        vmapping_np, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        remap_v_np = v_np[vmapping_np]
        remap_f_np = ft_np.astype(np.int32)

        return remap_v_np, remap_f_np, vt_np, ft_np, vmapping_np

    def to(self, device):
        self.device = device
        for name in ["v", "f", "vn", "fn", "vt", "ft", "albedo"]:
            tensor = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    def write(self, path, write_texture=True):
        if path.endswith(".obj"):
            self.write_obj(path, write_texture=write_texture)
        else:
            raise NotImplementedError(f"format {path} not supported!")

    def write_obj(self, base_path, write_texture=False, texture_only=False, group_correspondence=None):
        """
        Write each connected component to a separate .obj file along with its corresponding texture and material.
        
        Args:
            base_path (str): The base path for saving OBJ, MTL, and texture files.
        """
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        for reference_idx in self.groups:
            target_group_idx = group_correspondence[reference_idx] if group_correspondence is not None else reference_idx
            if self.partial_groups is not None and reference_idx not in self.partial_groups:
                    continue
            for component_idx in self.groups[reference_idx]:
                if self.partial_groups:
                    obj_path = base_path.replace(".obj", f"part_add_component_{component_idx}.obj")
                    mtl_path = base_path.replace(".obj", f"part_add_group{reference_idx}.mtl")
                    texture_filename = f"part_add_{base_name}_texture_group{reference_idx}.png"
                else:
                    obj_path = base_path.replace(".obj", f"_component_{component_idx}.obj")
                    mtl_path = base_path.replace(".obj", f"_group{target_group_idx}.mtl")
                    texture_filename = f"{base_name}_texture_group{target_group_idx}.png"
                if component_idx == reference_idx:
                    if write_texture:
                        # Write component texture
                        albedo_np = self.albedo[reference_idx].cpu().detach().numpy()
                        albedo_np = (albedo_np.clip(0, 1) * 255).astype(np.uint8)
                        albedo_bgr = cv2.cvtColor(albedo_np, cv2.COLOR_RGB2BGR)
                        texture_filepath = os.path.join(os.path.dirname(base_path), texture_filename)
                        cv2.imwrite(texture_filepath, albedo_bgr)
                    with open(mtl_path, "w") as mtl_fp:                        
                        # Write the material for this component to the .mtl file
                        mtl_fp.write(f"newmtl material_group{target_group_idx} \n")
                        mtl_fp.write(f"Ka 1.0 1.0 1.0 \n")
                        mtl_fp.write(f"Kd 1.0 1.0 1.0 \n")
                        mtl_fp.write(f"Ks 0.0 0.0 0.0 \n")
                        mtl_fp.write(f"Ns 0.0 \n")
                        mtl_fp.write(f"illum 1 \n")
                        mtl_fp.write(f"map_Kd {texture_filename} \n")

                if texture_only:
                    continue

                with open(obj_path, "w") as obj_fp:
                    obj_fp.write(f"mtllib {os.path.basename(mtl_path)} \n") # (TODO)

                    # Get vertices, texture coordinates, and normals for this component
                    v_np = self.components[component_idx]['vertices'].cpu().detach().numpy()
                    vt_np = self.components[component_idx]['uvs'].cpu().detach().numpy()
                    vn_np = self.components[component_idx]['vn'].cpu().detach().numpy()
                    f_np = self.components[component_idx]['faces'].cpu().detach().numpy()

                    # Write vertices to .obj file
                    for v in v_np:
                        obj_fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

                    # Write texture coordinates (if available) to .obj file
                    if vt_np is not None:
                        for vt in vt_np:
                            obj_fp.write(f"vt {vt[0]} {1 - vt[1]} \n")

                    # Assign the material to this component
                    obj_fp.write(f"usemtl material_group{reference_idx} \n")

                    # Write faces
                    for face_idx in range(f_np.shape[0]):
                        f = f_np[face_idx]
                        face_str = f"f"
                        for vertex_idx in range(3):
                            v_idx = f[vertex_idx] + 1
                            vt_idx = v_idx if vt_np is not None else ""
                            vn_idx = v_idx if vn_np is not None else ""
                            face_str += f" {v_idx}/{vt_idx}"
                        obj_fp.write(face_str + "\n")


    # visulization methods (FOR DEBUG)
    def create_trimesh_for_components(self, mesh):
        trimesh_components = []

        for component in mesh.components:
            vertices = component['vertices'].cpu().numpy()
            faces = component['faces'].cpu().numpy()
            uvs = component['uvs'].cpu().numpy()
            vn = component['vn'].cpu().numpy()


            #### uv
            vertex_colors = np.zeros((vertices.shape[0], 3))
            vertex_colors[:, 0] = uvs[:, 0]  # Red channel (U)
            vertex_colors[:, 1] = uvs[:, 1]  # Green channel (V)
            vertex_colors[:, 2] = 1.0  # Blue channel (constant 1.0

            # Create a trimesh object for the current component
            trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False, visual=trimesh.visual.TextureVisuals(uv=uvs))
            trimesh_components.append(trimesh_mesh)

        return trimesh_components


if __name__ == '__main__':
    input_folder = "C:\\Users\\AAA\Documents\\blender\\procedural_cakes"

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.obj'):
            obj_path = os.path.join(input_folder, file_name)
            mesh = Mesh.load(obj_path)  

            print(f"File: {file_name}")
            print(f"Groups: {mesh.groups}")

            for group_id, indices in mesh.groups.items():
                print(f"Visualizing Group {group_id} with indices {indices}")

                group_vertices = []
                group_faces = []
                face_offset = 0

                for idx in indices:
                    component = mesh.components[idx]
                    group_vertices.append(component['vertices'].detach().cpu().numpy())

                    group_faces.append(component['faces'].detach().cpu().numpy() + face_offset)
                    face_offset += component['vertices'].shape[0]

                group_mesh = trimesh.Trimesh(
                    vertices=np.vstack(group_vertices),
                    faces=np.vstack(group_faces)
                )

                group_mesh.show()