import nvdiffrast.torch as dr
import torch
import os
import glob
import numpy as np
import trimesh
import torch.nn.functional as F
import cv2
import json
from InTex.mesh_component import Mesh
from InTex.cam_utils import orbit_camera, OrbitCamera
from tqdm import tqdm

def render(glctx, pose, proj, meshes, resolution, enable_mip, max_mip_level):
    renders = []
    global_depth_min, global_depth_max = np.inf, -np.inf
    
     # Convert pose and projection matrices to torch tensors
    pose = torch.from_numpy(pose.astype(np.float32)).cuda()
    proj = torch.from_numpy(proj.astype(np.float32)).cuda()
    
    # get global min depth and max depth
    for (pos, pos_idx, uv, tex) in meshes:
        v_cam = torch.matmul(
            torch.functional.F.pad(pos, (0, 1), "constant", 1.0),
            torch.inverse(pose).T,
        ).float().unsqueeze(0)
        depth = -1 / (v_cam[..., [2]] + 1e-20)
        global_depth_min = min(global_depth_min, torch.min(depth).item())
        global_depth_max = max(global_depth_max, torch.max(depth).item())
    
    for (pos, pos_idx, uv, tex) in meshes:
        uv_idx = pos_idx
        v_cam_component = torch.matmul(F.pad(pos, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip_component = v_cam_component @ proj.T
        rast_out, rast_out_db = dr.rasterize(glctx, v_clip_component, pos_idx, resolution=[resolution, resolution])
        if enable_mip:
            texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
            color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
        else:
            texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
            color = dr.texture(tex[None, ...], texc, filter_mode='linear')
        color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
        
        disp = -1 / (v_cam_component[..., [2]] + 1e-20)
        disp = (disp - global_depth_min) / (global_depth_max - global_depth_min + 1e-20)
        depth, _ = dr.interpolate(disp, rast_out, pos_idx)
        depth = depth.clamp(0, 1).squeeze(0)
        
        renders.append((color, depth))
    
    # Combine the renders by comparing the depth
    color = torch.ones(1, resolution, resolution, 3, dtype=torch.float32, device="cuda")
    depth = torch.full((1, resolution, resolution, 1), -float('inf'), dtype=torch.float32, device="cuda")

    for c, d in renders:
        mask = (d != 0) & (d > depth)
        color_mask = torch.cat([mask, mask, mask], axis=-1)
        color = torch.where(color_mask, c, color)
        depth = torch.where(mask, d, depth) 

    return color

def read_procedural_parameters(param_dir, device):
    '''Parse procedural parameters from the json files in the mesh_fp directory'''
    params_dict = {}
    params_list = []
    for file in os.listdir(param_dir):
        if file.endswith('.json'):
            with open(os.path.join(param_dir, file)) as f:
                data = json.load(f)
            params_list.append(list(data.values()))
            params_dict[file.replace('.json', '')] = data
    
    params_list = np.array(params_list)
    mean_params = np.mean(params_list, axis=0)
    diff = np.abs(params_list - mean_params)
    diff = np.sum(diff, axis=1)
    target_idx = np.argmin(diff)
    
    return params_dict, target_idx

if __name__ == "__main__":
    
    PARAM_DIR = "data/procedural_fork/inference" # directory containing procedural parameter json files
    RESULT_DIR = "logs/fork/" # outputs from train.py or inference.py
    OUTPUT_PATH = "fork.mp4" # where to save the final video
    # component_correspondence_fp = os.path.join(RESULT_DIR, "component_correspondence.json") # do not need this because we produced texture maps for each geometry
    
    # produce turntable video or static pose video
    TURNTABLE_VIDEO = True
    
    # with open(component_correspondence_fp) as f:
    #     component_correspondence = json.load(f)
    param_dict, target_idx = read_procedural_parameters(PARAM_DIR, "cuda")

    # camera settings 
    h = 512
    w = 512
    radius = 1
    fovy = 50
    cam = OrbitCamera(w, h, r=radius, fovy=fovy)
    glctx = dr.RasterizeGLContext()
    rot_angle = 0 # start with 0 azimuth angle
    
    renderings = []
    
    rot_step_size = (360 // len(param_dict))
    
    # read meshes and textures
    print(f"Producing video with {len(param_dict)} meshes.")
    for mesh_idx in tqdm(range(len(param_dict))):
        meshes = []
        pattern = os.path.join(RESULT_DIR, f"mesh{mesh_idx}_component*.obj")
        matching_files = glob.glob(pattern)
        # print(f"Found {len(matching_files)} matching files for mesh{mesh_idx}.")
        # print(matching_files)
        whole_mesh = Mesh.load(os.path.join(PARAM_DIR, f"{mesh_idx}.obj"), resize=False, device="cuda")
        groups = whole_mesh.groups
        inverse_groups = {}
        for group_idx, component_indices in groups.items():
            for component_idx in component_indices:
                inverse_groups[component_idx] = group_idx
        for i, component_path in enumerate(matching_files):
            mesh = trimesh.load(component_path)
            component_idx = component_path.split("component")[-1].split(".obj")[0]
            corresponding_tex_idx = inverse_groups[int(component_idx)]
            tex_path = os.path.join(RESULT_DIR, f"mesh{mesh_idx}_group{corresponding_tex_idx}.png")
            tex = cv2.imread(tex_path)
            tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
            tex = torch.from_numpy(tex.astype(np.float32) / 255).cuda()
            v = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            f = torch.tensor(mesh.faces, dtype=torch.int32).cuda()
            vt = torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
            vt[:, 1] = 1 - vt[:, 1]
            meshes.append((v, f, vt, tex))

        if TURNTABLE_VIDEO:
            pose = orbit_camera(-30, rot_angle, radius=radius, is_degree=True)
        else:
            pose = orbit_camera(-30, 0, radius=radius, is_degree=True)
        proj = cam.perspective
        color = render(
            glctx, pose, proj, meshes, 512, False, 0
        )
        color = (color.cpu().numpy()[0] * 255).astype(np.uint8) # [h, w, 3]
        for i, (param_name, param_value) in enumerate(param_dict[str(mesh_idx)].items()):
            color = cv2.putText(color, f"{param_name}: {param_value}", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        renderings.append(color)
        
        rot_angle = (rot_angle + rot_step_size) % 360
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps=5, frameSize=(w, h))
    for frame in renderings:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(frame)
    video.release()
    print(f"Video saved to {OUTPUT_PATH}.")