from network.network import PointNet
from InTex.mesh_component import Mesh, compare_mesh_topology
from utils.mesh_io import normalize_shape
from utils.bake_texture import texture_transfer, inverse_map_batched
from utils.icp import icp
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import torch
import time
import logging
import numpy as np
import json
import os

TARGET_MESH_FP = "data/procedural_sofa/train/41.obj" # template mesh used in train.py
TRAIN_RESULT_FP = "logs/sofa/" # outputs from train.py
INFERENCE_DATA_DIR = "data/procedural_sofa/inference" # input data for inference
PARAM_COUNT = 4

def write_mesh_components_with_mtl(mesh, mesh_idx, correspondence, savedir):
    for group_idx, group in mesh.groups.items():
        target_group_idx = correspondence[group_idx]
        for component_idx in group:
            component = mesh.components[component_idx]

            mtl_name = f"mesh{mesh_idx}_group{group_idx}.mtl"
            tex_name = f"mesh{mesh_idx}_group{group_idx}.png"
            obj_path = os.path.join(savedir, f"mesh{mesh_idx}_component{component_idx}.obj")
            mtl_path = os.path.join(savedir, mtl_name)

            with open(mtl_path, "w") as mtl_fp:
                mtl_fp.write(f"newmtl material_group{target_group_idx}\n")
                mtl_fp.write("Ka 1.0 1.0 1.0\n")  # ambient color
                mtl_fp.write("Kd 1.0 1.0 1.0\n")  # diffuse color
                mtl_fp.write("Ks 0.0 0.0 0.0\n")  # specular
                mtl_fp.write("d 1.0\n")           # transparency
                mtl_fp.write("illum 1\n")
                mtl_fp.write(f"map_Kd {tex_name}\n")  # link to the texture

            # === Write OBJ file ===
            with open(obj_path, "w") as obj_fp:
                obj_fp.write(f"mtllib {mtl_name}\n")

                # Write vertices
                for v in component['vertices']:
                    v = v / mesh.ori_scale
                    obj_fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

                # Determine UVs
                vt_np = component['uvs']

                for vt in vt_np:
                    obj_fp.write(f"vt {vt[0]} {1 - vt[1]}\n")  # flip Y axis

                # Use material
                obj_fp.write(f"usemtl material_group{target_group_idx}\n")

                # Write faces
                for face in component['faces']:
                    face_str = "f"
                    for vertex_idx in range(3):
                        v_idx = face[vertex_idx] + 1
                        vt_idx = v_idx if 'uvs' in component else ""
                        face_str += f" {v_idx}/{vt_idx}"
                    obj_fp.write(face_str + "\n")


def write_mesh_components_with_mtl_single_file(mesh, mesh_idx, correspondence, savedir):
    obj_path = os.path.join(savedir, f"mesh{mesh_idx}.obj")
    mtl_name = f"mesh{mesh_idx}.mtl"
    mtl_path = os.path.join(savedir, mtl_name)

    all_vertices = []
    all_uvs = []
    all_faces = []

    material_faces = {}

    with open(mtl_path, "w") as mtl_fp:
        for group_idx in mesh.groups:
            target_group_idx = correspondence[group_idx]
            tex_name = f"mesh{mesh_idx}_group{group_idx}.png"

            mtl_fp.write(f"newmtl material_group{target_group_idx}\n")
            mtl_fp.write("Ka 1.0 1.0 1.0\n")
            mtl_fp.write("Kd 1.0 1.0 1.0\n")
            mtl_fp.write("Ks 0.0 0.0 0.0\n")
            mtl_fp.write("d 1.0\n")
            mtl_fp.write("illum 1\n")
            mtl_fp.write(f"map_Kd {tex_name}\n\n")

    vertex_offset = 0
    for group_idx, group in mesh.groups.items():
        target_group_idx = correspondence[group_idx]
        for component_idx in group:
            component = mesh.components[component_idx]
            v = component['vertices'] / mesh.ori_scale
            vt = component['uvs']
            f = component['faces']

            remapped_faces = []
            for face in f:
                remap = []
                for vid in face:
                    remap.append(vertex_offset + vid)
                remapped_faces.append(remap)
            material_faces.setdefault(target_group_idx, []).extend(remapped_faces)

            all_vertices.extend(v)
            all_uvs.extend(vt)
            vertex_offset += len(v)

    with open(obj_path, "w") as obj_fp:
        obj_fp.write(f"mtllib {mtl_name}\n")

        for v in all_vertices:
            obj_fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for vt in all_uvs:
            obj_fp.write(f"vt {vt[0]} {1 - vt[1]}\n")

        for mat_idx, faces in material_faces.items():
            obj_fp.write(f"usemtl material_group{mat_idx}\n")
            for face in faces:
                face_str = "f"
                for idx in face:
                    i = idx + 1
                    face_str += f" {i}/{i}"
                obj_fp.write(face_str + "\n")
                
def compute_correspondence(source_mesh, target_mesh):
    # Calculate icp distance between each component
    icp_dist = np.full((len(source_mesh.groups), len(target_mesh.groups)), np.inf)

    for i, source_group_idx in enumerate(source_mesh.groups.keys()):
        source_component = source_mesh.components[source_group_idx]
        # Check if the same topology to avioid running icp
        for j, target_group_idx in enumerate(target_mesh.groups.keys()):
            target_component = target_mesh.components[target_group_idx]
            if compare_mesh_topology(source_component['vertices_ori'], source_component['faces_ori'], target_component['vertices_ori'], target_component['faces_ori']):
                icp_dist[i, j] = -1
                break
            else:
                T, distances, itr, R = icp(source_component, target_component, target_mesh.global_symmetry_axis, max_iterations=200, tolerance=1e-8, allow_scaling=True, return_rotation=True)
                if np.isclose(np.mean(distances), -1):
                    icp_dist[i, j] = np.inf
                else:
                    angles = Rot.from_matrix(R).as_euler('xyz', degrees=True)
                    x_deg, y_deg, z_deg = angles
                    if np.abs(x_deg) > 30 or np.abs(z_deg) > 30:
                        icp_dist[i, j] = np.inf                
                    else:
                        icp_dist[i, j] = np.mean(distances)
    
    correspondence = np.argmin(icp_dist, axis=1)
    
    source_groups = list(source_mesh.groups.keys())
    target_groups = list(target_mesh.groups.keys())
    correspondence = {source_groups[i]:target_groups[correspondence[i]] for i in range(len(source_groups))}

    return correspondence

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f'{TRAIN_RESULT_FP}/hue', exist_ok=True)
        
    # Load the target mesh and trained model
    target_mesh = Mesh.load(TARGET_MESH_FP, resize=True, device=device) 
    networks = {}
    for group_idx in target_mesh.groups.keys():
        ckpt_fp = os.path.join(TRAIN_RESULT_FP, "ckpt", f"component_{group_idx}.ckpt")
        
        if not os.path.exists(ckpt_fp):
            logging.info(f"Group {group_idx} does not have a trained model. UVs will be directly assigned.")
            continue
        else:
            logging.info(f"Loading model for group {group_idx}")
            net = PointNet(feature_dim=256, num_mlp1_layers=2, num_mlp2_layers=2)
            net.load_state_dict(torch.load(ckpt_fp))  
            networks[group_idx] = net.to(device)
    
    component_correspondence = {}
    start = time.time()
    # Load the inference data
    for file in os.listdir(INFERENCE_DATA_DIR):
        if not file.endswith(".obj"):
            continue
        
        logging.info(f"Processing {file}")
        mesh_fp = os.path.join(INFERENCE_DATA_DIR, file)
        json_fp = os.path.join(INFERENCE_DATA_DIR, file.replace(".obj", ".json"))
        mesh = Mesh.load(mesh_fp, resize=True, device=device)
        mesh_idx = file.replace(".obj", "")
        component_correspondence[mesh_idx] = {}
        with open(json_fp, "r") as f:
            data = json.load(f)
        data = torch.tensor(list(data.values())).float().to(device)
        
        if len(target_mesh.components) == len(mesh.components):
            # easy case. assume the order of components is the same
            correspondence = {group_idx:group_idx for group_idx in target_mesh.groups.keys()}
        else:
            # hard case. number of components is different. need to do matching
            correspondence = compute_correspondence(mesh, target_mesh)
        
        # run inference    
        for group_idx, component_indices in mesh.groups.items():
            target_group_idx = correspondence[group_idx]
            for component_idx in component_indices:
                component_correspondence[mesh_idx][component_idx] = target_group_idx
                if target_group_idx in networks:
                    if component_idx == group_idx:
                        component = mesh.components[component_idx]
                        target_component = target_mesh.components[target_group_idx]
                        if compare_mesh_topology(component['vertices_ori'], component['faces_ori'], target_component['vertices_ori'], target_component['faces_ori']):
                            # directly assign
                            texture_image = Image.open(f"{TRAIN_RESULT_FP}/out_texture_group{correspondence[group_idx]}.png").convert("RGB")
                            output_texture_file = f"{TRAIN_RESULT_FP}/hue/mesh{mesh_idx}_group{group_idx}.png"
                            texture_image.save(output_texture_file)
                            for component_idx in mesh.groups[group_idx]:
                                mesh.components[component_idx]['vertices'] = torch.tensor(mesh.components[component_idx]['vertices_ori'][target_mesh.components[correspondence[group_idx]]['vmapping']], dtype=torch.float32, device=device)
                                mesh.components[component_idx]['faces'] = target_mesh.components[correspondence[group_idx]]['faces'].clone()
                                mesh.components[component_idx]['uvs'] = target_mesh.components[correspondence[group_idx]]['uvs']
                            break                 
                        source_triangles_uvs = component['uvs'][component['faces']]
                        target_triangles_uvs = 1 - target_component['uvs'][target_component['faces']]
                        # source_faces = component['faces_ori']
                        target_vertices = target_component['vertices']
                        target_faces = target_component['faces']
                        
                        surface_points, texel_indices = inverse_map_batched(
                            triangles=source_triangles_uvs,
                            n=512,
                            mesh=type('Mesh', (object,), {'faces': component['faces'], 'vertices': component['vertices']})(),
                            batch_size=5000,
                            device=device
                        )
                        _, center, scale = normalize_shape(component['vertices'])
                        surface_points = (surface_points - center)/ scale
                        with torch.no_grad():
                            # networks[target_group_idx].update_procedural_parameters(data)
                            out = networks[target_group_idx](surface_points)
                        _, center, scale = normalize_shape(target_vertices)
                        out = out * scale + center
                        
                        texture_transfer(TRAIN_RESULT_FP, f'{TRAIN_RESULT_FP}/hue', correspondence, group_idx, target_vertices, target_faces, target_triangles_uvs, out, texel_indices, mesh_idx, device)
                else:
                    # directly assign
                    texture_image = Image.open(f"{TRAIN_RESULT_FP}/out_texture_group{correspondence[group_idx]}.png").convert("RGB")
                    output_texture_file = f"{TRAIN_RESULT_FP}/hue/mesh{mesh_idx}_group{group_idx}.png"
                    texture_image.save(output_texture_file)
                    for component_idx in mesh.groups[group_idx]:
                        mesh.components[component_idx]['vertices'] = torch.tensor(mesh.components[component_idx]['vertices_ori'][target_mesh.components[correspondence[group_idx]]['vmapping']], dtype=torch.float32, device=device)
                        mesh.components[component_idx]['faces'] = target_mesh.components[correspondence[group_idx]]['faces'].clone()
                        mesh.components[component_idx]['uvs'] = target_mesh.components[correspondence[group_idx]]['uvs']

        # save the mesh
        idx = file.replace(".obj", "")
        write_mesh_components_with_mtl_single_file(mesh, idx, correspondence, f'{TRAIN_RESULT_FP}/hue')
        logging.info(f"Mesh {idx} saved.")
    
    with open(os.path.join(f'{TRAIN_RESULT_FP}/hue', "component_correspondence.json"), "w") as f:
        json.dump(component_correspondence, f)
    logging.info(f"Component correspondence saved.")
    
    end = time.time()
    logging.info(f"Inference completed in {end - start:.2f} seconds.")