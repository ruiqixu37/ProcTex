import argparse
import json
import os
import re
import logging
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from InTex.generate_texture_map import generate_texture_map
from InTex.mesh_component import Mesh, compare_mesh_topology
from network.network import PointNet
from utils.bake_texture import texture_transfer, inverse_map_batched, compute_surface_projections_from_3d
from utils.nonrigid_registration import compute_cage_aabb, optimize_cage_mvc, deform_with_MVC, icp_umeyama
from utils.icp import icp
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import point_mesh_face_distance
from utils.mesh_io import normalize_shape

def find_mean_mesh_fp(mesh_dir):
    """find the mesh file path that has the median parameters"""
    files = os.listdir(mesh_dir)
    params_fp = []
    params_list = []
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(mesh_dir, file)) as f:
                data = json.load(f)
            params_fp.append(os.path.join(mesh_dir, file))
            params_list.append(list(data.values()))
    params_list = np.array(params_list)

    # find the parameters index that is closest to the mean
    mean_params = np.mean(params_list, axis=0)
    diff = np.abs(params_list - mean_params)
    diff = np.sum(diff, axis=1)
    idx = np.argmin(diff)
    
    json_fp = params_fp[idx]
    mesh_fp = json_fp.replace('.json', '.obj')
    return mesh_fp

def generate_component_meshes(opt):
    """generate mesh components for all training meshes"""
    mesh_dict = {}
    for file in os.listdir(opt.mesh):
        if file.endswith('.obj'):    
            logging.debug(f"Processing {file}")
            mesh = Mesh.load(os.path.join(opt.mesh, file), resize=True, device='cuda')
            mesh_dict[file.replace('.obj', '')] = mesh
    
    return mesh_dict

def compute_group_correspondences(target_mesh_idx, mesh_dict, opt_InTex, gui_template, threshold=0.1):
    d = {}
    train_displacement_groups = set()
    target_mesh = mesh_dict[target_mesh_idx]
    target_group_indices = list(target_mesh.groups.keys())
    target_albedo, target_cnt, target_viewcos_cache = gui_template.renderer.mesh.albedo.copy(), gui_template.renderer.mesh.cnt.copy(), gui_template.renderer.mesh.viewcos_cache.copy()
    original_albedo = gui_template.albedo.copy()
    
    for mesh_idx, mesh in tqdm(mesh_dict.items(), desc="Processing training meshes", total=len(mesh_dict)):
        correspondence = {}
        group_indices = list(mesh.groups.keys())
        
        # Calculate icp distance between groups of the current mesh and the target mesh
        icp_dist = np.full((len(mesh.groups), len(target_mesh.groups)), np.inf)
        icp_T  = [[None for j in range(len(target_mesh.groups))] for i in range(len(mesh.groups))]
        
        for idx, group_idx in enumerate(group_indices):
            component = mesh.components[group_idx]
            # Check if the same topology to avioid running icp
            for target_idx, target_group_idx in enumerate(target_group_indices):
                target_component = target_mesh.components[target_group_idx]
                if compare_mesh_topology(component['vertices_ori'], component['faces_ori'], target_component['vertices_ori'], target_component['faces_ori']):
                    icp_dist[idx, target_idx] = -1
                    break
                else:
                    # use sample vertices whose number of vertices is the same, which is required by icp
                    T, distances, itr, R = icp(component, target_component, target_mesh.global_symmetry_axis, max_iterations=200, tolerance=1e-8, allow_scaling=True, return_rotation=True)
                    if np.isclose(np.mean(distances), -1):
                        icp_dist[idx, target_idx] = np.inf
                    else:
                        angles = Rot.from_matrix(R).as_euler('xyz', degrees=True)
                        x_deg, y_deg, z_deg = angles
                        if np.abs(x_deg) > 45 or np.abs(z_deg) > 45:
                            icp_dist[idx, target_idx] = np.inf                
                        else:
                            icp_dist[idx, target_idx] = np.mean(distances)
                    icp_T[idx][target_idx] = T
        
        # if the part addition can be explained by direct assignment or ICP, then we do not need to run re-texturing
        non_affine_groups = len(mesh.groups) - np.sum(np.min(icp_dist, axis=1) <= threshold)
        unmatched_target = np.sum(np.all(icp_dist > threshold, axis=0))
        extra_groups = max(non_affine_groups - unmatched_target, 0)

        # identify the extra groups if needed. The extra groups are the ones with largest min distance to any target group
        extra_groups_indices = []
        if extra_groups > 0:
            for i in range(len(mesh.groups)):
                # if icp_dist is infinity for all entries, then this component is an extra component
                if np.all(icp_dist[i] == np.inf):
                    extra_groups_indices.append(i)
            extra_groups -= len(extra_groups_indices)
            
            # find the extra components with the largest min distance to any target component
            icp_dist_nan = np.where(icp_dist == np.inf, np.nan, icp_dist)
            row_mins = np.nanmin(icp_dist_nan, axis=1)
            sorted_indices = np.argsort(-row_mins)
            extra_groups_indices.extend(sorted_indices[:extra_groups])

        # For part addition, rerun the InTex pipeline, and only update the textures for the new components
        if extra_groups_indices != []:
            new_mesh_fp = opt_InTex.mesh.replace(f"{target_mesh_idx}.obj", f"{mesh_idx}.obj")
            opt_InTex.mesh = new_mesh_fp
            opt_InTex.partial_groups = [group_indices[i] for i in extra_groups_indices]
            logging.info(f"Running retexturing (part addition case) for mesh {mesh_idx}...")
            generate_texture_map(opt_InTex)
            

        for i, group_idx in enumerate(group_indices):
            if group_idx in extra_groups_indices: # Part addition
                correspondence[group_idx] = None
                continue
            # Find current component's correspondence to the target_mesh
            # correspondence[group_idx] = target_group_indices[np.argmin(icp_dist[i])]
            correspondence[group_idx] = group_idx

            source_triangles_uvs = mesh.components[group_idx]['uvs'][mesh.components[group_idx]['faces']]
            target_triangles_uvs = 1 - target_mesh.components[correspondence[group_idx]]['uvs'][target_mesh.components[correspondence[group_idx]]['faces']]
            source_faces = mesh.components[group_idx]['faces_ori']
            target_vertices = target_mesh.components[correspondence[group_idx]]['vertices']
            target_faces = target_mesh.components[correspondence[group_idx]]['faces']
            icp_target_index = target_group_indices.index(correspondence[group_idx])
            if np.min(icp_dist[i]) == -1: # Direct assign
                logging.debug("mesh" + str(mesh_idx) + ", group" + str(group_idx) + " direct assign")  # Keep consistent so make a copy
                texture_image = Image.open(f"{opt_InTex.outdir}/out_texture_group{correspondence[group_idx]}.png").convert("RGB")
                output_texture_file = f"{opt_InTex.outdir}/mesh{mesh_idx}_group{group_idx}.png"
                texture_image.save(output_texture_file)
                for component_idx in mesh.groups[group_idx]:
                    mesh.components[component_idx]['vertices'] = torch.tensor(mesh.components[component_idx]['vertices_ori'][target_mesh.components[correspondence[group_idx]]['vmapping']], dtype=torch.float32, device=device)
                    mesh.components[component_idx]['faces'] = target_mesh.components[correspondence[group_idx]]['faces'].clone()
                    mesh.components[component_idx]['uvs'] = target_mesh.components[correspondence[group_idx]]['uvs']
            else:
                surface_points, texel_indices = inverse_map_batched(
                    triangles=source_triangles_uvs,
                    n=opt_InTex.texture_size,
                    mesh=type('Mesh', (object,), {'faces': mesh.components[group_idx]['faces'], 'vertices': mesh.components[group_idx]['vertices']})(),
                    batch_size=5000,
                    device=device
                )
                logging.debug("mesh" + str(mesh_idx) + ", group" + str(group_idx) + " running icp")
                T = torch.tensor(icp_T[i][icp_target_index], dtype=surface_points.dtype, device=surface_points.device)  # (4, 4)
                # direct output
                uv_vertices = (T[:3, :3] @ surface_points.T).T + T[:3, 3]
                # prepare for training
                deformed = (T[:3, :3] @ mesh.components[group_idx]['sampled'].T).T + T[:3, 3]
                distance = point_mesh_face_distance(Meshes(target_vertices.unsqueeze(0), target_faces.unsqueeze(0)), Pointclouds(uv_vertices.unsqueeze(0)))
                
                if distance > 0: # Run cage deform
                    logging.debug("mesh" + str(mesh_idx) + ", group" + str(group_idx) + " running cage deformation")
                    aligned, s_final, R_final, t_final, history = icp_umeyama(
                        mesh.components[group_idx]['vertices_ori'], target_mesh.components[correspondence[group_idx]]['vertices_ori'],
                        max_iterations=100,
                        tol=1e-8
                    )
                    s_final = torch.from_numpy(np.array([s_final])).float().to(device)
                    R_final = torch.from_numpy(R_final).float().to(device)
                    t_final = torch.from_numpy(t_final).float().to(device)
                    
                    # use dummy s R T
                    s_final = torch.tensor([1.0], dtype=torch.float32, device=device)
                    R_final = torch.eye(3, dtype=torch.float32, device=device)
                    t_final = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
                    aligned = mesh.components[group_idx]['vertices_ori']
                    
                    # visualize_open3d(mesh.components[group_idx]['vertices_ori'], target_mesh.components[correspondence[group_idx]]['vertices_ori'])
                    # visualize_open3d(aligned, target_mesh.components[correspondence[group_idx]]['vertices_ori'])

                    if 'cage_vertices' not in mesh.components[group_idx]:
                        splits = np.linspace(0, 1, num=1)[1:-1].tolist()
                        mesh.components[group_idx]['cage_vertices'], mesh.components[group_idx]['cage_faces'] = compute_cage_aabb(torch.from_numpy(aligned).float(), [], 0.1)

                    registration_points = (s_final * (mesh.components[group_idx]['sampled'] @ R_final)) + t_final
                    cage_v, cage_f, loss = optimize_cage_mvc(
                        cage_init = mesh.components[group_idx]['cage_vertices'].to(device),
                        cage_faces = mesh.components[group_idx]['cage_faces'].to(device).to(torch.int64),
                        source_points = registration_points.unsqueeze(0).to(device),
                        target_points = target_mesh.components[correspondence[group_idx]]['sampled'].unsqueeze(0).to(device),
                        target_normals = None,
                        lr = 1e-3,
                        plateau_threshold = 1e-5,
                        plateau_period = 100,
                        convergence_threshold = 1e-5,
                        num_epochs = 1000,
                        lap_weight = 0,
                        mvc_weight = .05,
                        shape_preservation_weight = 0.,
                        normal_preservation_weight = 5.,
                        align_loss_weight = 10.,
                        p2f_loss_weight = 1.,
                        x_symmetry_loss_weight = 0.,
                        y_symmetry_loss_weight = 0.,
                        z_symmetry_loss_weight = 0.,
                        neighborhood_size = 1,
                        surface_penalty_weight = 1.0,
                        surface_mesh = Meshes(target_mesh.components[correspondence[group_idx]]['vertices'].unsqueeze(0), target_mesh.components[correspondence[group_idx]]['faces'].unsqueeze(0)),
                        device = device
                    )

                    # direct output
                    uv_vertices = []
                    for j in range(0, len(surface_points), 5000):
                        sp = (s_final * (surface_points[j:j+5000] @ R_final)) + t_final
                        batch_uv_vertices = deform_with_MVC(
                            mesh.components[group_idx]['cage_vertices'].to(device),
                            cage_v.to(device),
                            mesh.components[group_idx]['cage_faces'].to(device).to(torch.int64),
                            sp.unsqueeze(0),
                            verbose=False
                        )
                        uv_vertices.append(batch_uv_vertices.squeeze(0))
                    uv_vertices = torch.cat(uv_vertices, dim=0).to(torch.float32)
                    # prepare for training
                    deformed = deform_with_MVC(
                        mesh.components[group_idx]['cage_vertices'].to(device),
                        cage_v.to(device),
                        mesh.components[group_idx]['cage_faces'].to(device).to(torch.int64),
                        registration_points.unsqueeze(0),
                        verbose=False
                    )
                    
                mesh.components[group_idx]['targets'] = compute_surface_projections_from_3d(target_vertices[target_faces], deformed.squeeze(0)) 
                # visualize_open3d(mesh.components[group_idx]['targets'].cpu().detach().numpy(), target_vertices.cpu().detach().numpy())
                # uv_vertices is the 3d position mapping onto target component                 
                texture_transfer(opt_InTex.outdir, opt_InTex.outdir, correspondence, group_idx, target_vertices, target_faces, target_triangles_uvs, uv_vertices, texel_indices, mesh_idx, device)
                train_displacement_groups.add(correspondence[group_idx])
                
        logging.debug(f"Mesh {mesh_idx} group correspondence: {correspondence}")
        
        d[mesh_idx] = correspondence
        
        # # Check if there's disocclusion of existing components
        # if mesh_idx != target_mesh_idx:
        #     gui_template.renderer.mesh = mesh
        #     gui_template.renderer.mesh.albedo = {source_grp: target_albedo[target_grp] for source_grp, target_grp in correspondence.items()}
        #     gui_template.albedo = {source_grp: original_albedo[target_grp] for source_grp, target_grp in correspondence.items()}
        #     gui_template.renderer.mesh.cnt = {source_grp: target_cnt[target_grp] for source_grp, target_grp in correspondence.items()}
        #     gui_template.renderer.mesh.viewcos_cache = {source_grp: target_viewcos_cache[target_grp] for source_grp, target_grp in correspondence.items()}
        #     gui_template.generate(retexturing=True)
        #     gui_template.renderer.mesh.write_obj(os.path.join(opt_InTex.outdir, opt_InTex.save_path), write_texture=True, texture_only=True, group_correspondence=correspondence)        
        #     target_albedo = {target_grp: gui_template.renderer.mesh.albedo[source_grp] for source_grp, target_grp in correspondence.items()}
        #     target_cnt = {target_grp: gui_template.renderer.mesh.cnt[source_grp] for source_grp, target_grp in correspondence.items()}
        #     target_viewcos_cache = {target_grp: gui_template.renderer.mesh.viewcos_cache[source_grp] for source_grp, target_grp in correspondence.items()}
        #     original_albedo = {target_grp: gui_template.albedo[source_grp] for source_grp, target_grp in correspondence.items()}
        
    # gui_template.dilate_texture()
    # gui_template.deblur()
    
    return d, train_displacement_groups
    
def _read_procedural_parameters(mesh_fp, device):
    '''Parse procedural parameters from the json files in the mesh_fp directory'''
    params_dict = {}
    for file in os.listdir(mesh_fp):
        if file.endswith('.json'):
            with open(os.path.join(mesh_fp, file)) as f:
                data = json.load(f)
            data = torch.tensor(list(data.values()), dtype=torch.float32, device=device)
            
            params_dict[file.replace('.json', '')] = data
    
    param_count = len(params_dict[list(params_dict.keys())[0]])
    
    return params_dict, param_count

def train_single_group_model(opt, group_idx, target_mesh_idx, group_correspondence, network, params_dict, mesh_dict, global_step, device=torch.device('cuda:0')):
    '''Train a uv displacement model for each group'''
    network = network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.ProcCorrNet.lr)
    max_per_point_loss = np.inf
    num_train_meshes = len(params_dict) # should also be equal to len(mesh_dict.keys())
    
    if num_train_meshes != len(mesh_dict):
        logging.error("Number of training meshes does not match the number of parameters.")
    
    revert_correspondence = {}
    for mesh_idx, correspondence in group_correspondence.items():
        for source_group_idx, target_group_idx in correspondence.items():
            if target_group_idx == group_idx:
                if source_group_idx in revert_correspondence:
                    logging.error(f"Multiple groups in mesh {mesh_idx} are assigned to group {group_idx}. This is not supported for training displacement network.")
                revert_correspondence[mesh_idx] = source_group_idx
         
    target_vertices = mesh_dict[str(target_mesh_idx)].components[group_idx]['vertices']
    target_faces = mesh_dict[str(target_mesh_idx)].components[group_idx]['faces']       
    best_max_per_point_loss = np.inf
    
    # for iter in range(opt.ProcCorrNet.iterations):
    for iter in tqdm(range(opt.ProcCorrNet.iterations), desc=f"Training group {group_idx}", initial=global_step):
        mesh_idx = np.random.randint(num_train_meshes)
        while mesh_idx == target_mesh_idx or 'targets' not in mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]:
            mesh_idx = np.random.randint(num_train_meshes)
        source_vertices = mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]['sampled']
        mapped_vertices = mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]['targets']
        _, center, scale = normalize_shape(mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]['vertices'])
        source_vertices = (source_vertices - center) / scale
        _, center, scale = normalize_shape(target_vertices)
        mapped_vertices = (mapped_vertices - center) / scale
    
        # get the procedural parameters
        procedural_parameters = params_dict[str(mesh_idx)]
        # network.update_procedural_parameters(procedural_parameters)
        
        optimizer.zero_grad()    
        
        out = network(source_vertices)
        diff = torch.abs(out - mapped_vertices)  # (N, 3)
    
        per_point_loss = diff.norm(dim=1)
        loss = per_point_loss.mean()
        max_loss = per_point_loss.max()
        
        if max_loss.item() < best_max_per_point_loss:
            best_max_per_point_loss = max_loss.item()
        
        loss.backward()
        optimizer.step()  

        # if iter % opt.ProcCorrNet.verbose_interval == 0:
        #     logging.info(f"Epoch {iter}, Loss Avg: {loss.item():.6f}, Max Point Loss: {max_loss.item():.6f}")
        
    torch.save(network.state_dict(), os.path.join(f'{opt.InTex.outdir}', 'ckpt', f'component_{group_idx}.ckpt'))
    # print(f"Training for group {group_idx} finished. Best Max-Point loss: {best_max_per_point_loss}. Average loss: {loss.item()}")
    
    for mesh_idx in range(num_train_meshes):
        if mesh_idx == target_mesh_idx or 'targets' not in mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]:
            continue
        component = mesh_dict[str(mesh_idx)].components[revert_correspondence[str(mesh_idx)]]
        target_component = mesh_dict[str(target_mesh_idx)].components[group_idx]
        source_triangles_uvs = component['uvs'][component['faces']]
        target_triangles_uvs = 1 - target_component['uvs'][target_component['faces']]
        source_faces = component['faces_ori']
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
            # procedural_parameters = params_dict[str(mesh_idx)]
            # network.update_procedural_parameters(procedural_parameters)
            out = network(surface_points)
        _, center, scale = normalize_shape(target_vertices)
        out = out * scale + center
        # visualize_open3d(out.cpu().detach().numpy(), target_vertices.cpu().detach().numpy())
        
        # out is the 3d position mapping onto target component
        texture_transfer(opt_InTex.outdir, opt_InTex.outdir, group_correspondence[str(mesh_idx)], revert_correspondence[str(mesh_idx)], target_vertices, target_faces, target_triangles_uvs, out, texel_indices, mesh_idx, device)


def write_mesh_components_with_mtl(mesh_dict, target_mesh_idx, correspondence, base_path):
    component_correspondence = {}

    for mesh_idx, mesh in mesh_dict.items():
        if mesh_idx == target_mesh_idx:
            continue
        component_correspondence[mesh_idx] = {}
        logging.debug(f"Processing mesh {mesh_idx}...")

        for group_idx, group in mesh.groups.items():
            target_group_idx = correspondence[mesh_idx][group_idx]
            if target_group_idx is None:
                logging.debug(f"No correspondence found for group {group_idx}, skipping.")
                continue

            for component_idx in group:
                component_correspondence[mesh_idx][component_idx] = target_group_idx
                component = mesh.components[component_idx]

                mtl_name = f"mesh{mesh_idx}_group{group_idx}.mtl"
                tex_name = f"mesh{mesh_idx}_group{group_idx}.png"
                obj_path = os.path.join(base_path, f"mesh{mesh_idx}_component{component_idx}.obj")
                mtl_path = os.path.join(base_path, mtl_name)

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

        logging.debug(f"Finished writing mesh {mesh_idx}.")

    with open(os.path.join(base_path, "component_correspondence.json"), "w") as f:
        json.dump(component_correspondence, f)


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging.")
    args, extras = parser.parse_known_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    logging.getLogger('pywavefront').setLevel(logging.CRITICAL)
    
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    
    # Step 1: find the average mesh file
    target_mesh_fp = find_mean_mesh_fp(opt.mesh)
    # target_mesh_fp = "procedural_data/procedural_drawer_dual/1.obj"
    target_mesh_idx = re.split(r'[\\/]', target_mesh_fp)[-1].replace('.obj', '') # works for both windows and linux
    opt_InTex = OmegaConf.merge(opt.InTex, OmegaConf.from_cli(extras))
    opt_InTex.mesh = target_mesh_fp
    logging.info(f"Using {opt_InTex.mesh} as the template mesh.")
    
    # Step 2: generate texture maps for the target mesh
    gui_template = generate_texture_map(opt_InTex)
    # # Step 3: generate component meshes for all training meshes
    mesh_dict = generate_component_meshes(opt)    
    
    # Step 4: compute group correspondences across all meshes
    group_correspondence, train_displacement_groups = compute_group_correspondences(target_mesh_idx, mesh_dict, opt_InTex, gui_template)

    # Step 5: train the displacement networks and save the models
    if not os.path.exists(os.path.join(opt_InTex.outdir, 'ckpt')):
        os.makedirs(os.path.join(opt_InTex.outdir, 'ckpt'))
    params_dict, param_count = _read_procedural_parameters(opt.mesh, device)
    logging.info(f"train_displacement_groups: {train_displacement_groups}")
    for i, group_idx in enumerate(train_displacement_groups):
        network = PointNet(feature_dim=256, num_mlp1_layers=2, num_mlp2_layers=2)
        train_single_group_model(opt, group_idx, int(target_mesh_idx), group_correspondence, network, params_dict, mesh_dict, i * opt.ProcCorrNet.iterations, device=device)
    
    write_mesh_components_with_mtl(mesh_dict, target_mesh_idx, group_correspondence, opt_InTex.outdir)
    
    logging.info(f"Training finished. For inference purpose, the median mesh is {target_mesh_idx}.obj.")