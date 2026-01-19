import os
import math
import cv2
import trimesh
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from .mesh_component import Mesh, safe_normalize

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def trunc_rev_sigmoid(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))

def make_divisible(x, m=8):
    return int(math.ceil(x / m) * m)

class Renderer(nn.Module):
    def __init__(self, device, opt):
        
        super().__init__()

        self.device = device
        self.opt = opt

        self.mesh = None

        if opt.bg_image is not None and os.path.exists(opt.bg_image):
            # load an image as the background
            bg_image = cv2.imread(opt.bg_image)
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
            bg_image = torch.from_numpy(bg_image.astype(np.float32) / 255).to(self.device)
            self.bg = F.interpolate(bg_image.permute(2, 0, 1).unsqueeze(0), (opt.render_resolution, opt.render_resolution), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous()
        else:
            # default as blender grey
            # self.bg = 0.807 * torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
            self.bg = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.bg_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.device)

        if not self.opt.gui or os.name == 'nt':
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()

    @torch.no_grad()
    def load_mesh(self, path):
        if not os.path.exists(path):
            # try downloading from objaverse (treat path as uid)
            import objaverse
            objects = objaverse.load_objects(uids=[path], download_processes=1)
            path = objects[path]
            print(f'[INFO] load Objaverse from {path}')

        self.mesh = Mesh.load(path, retex=self.opt.retex, remesh=self.opt.remesh, device=self.device)

    @torch.no_grad()
    def export_mesh(self, path):
        self.mesh.write(path)
        
    def render(self, pose, proj, h, w):
        results = {}
        self.mesh.device = "cuda" if self.device == torch.device('cuda') else "cpu"

        # Convert pose and projection matrices to torch tensors
        pose = torch.from_numpy(pose.astype(np.float32)).to(self.mesh.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(self.mesh.device)

        v_cam = torch.matmul(F.pad(self.mesh.v.to(self.mesh.device), pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        global_depth = -1 / (v_cam[..., [2]] + 1e-20)
        depth_min = global_depth.min()
        depth_max = global_depth.max()

        # Initialize final aggregated buffers
        final_rgb = torch.zeros((h, w, 3), dtype=torch.float32, device=self.mesh.device)
        final_alpha = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)
        final_depth = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)
        final_normal = torch.zeros((h, w, 3), dtype=torch.float32, device=self.mesh.device)
        final_rot_normal = torch.zeros((h, w, 3), dtype=torch.float32, device=self.mesh.device)
        final_viewcos = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)
        final_texc = torch.zeros((h, w, 2), dtype=torch.float32, device=self.mesh.device) 
        final_group = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)

        final_cnt = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)
        final_viewcos_cache = torch.zeros((h, w, 1), dtype=torch.float32, device=self.mesh.device)

        per_component_depths = {}

        # Loop through each component of the mesh
        components = self.mesh.components
        for reference_idx in self.mesh.groups:
            component_albedo = self.mesh.albedo[reference_idx]
            component_cnt = self.mesh.cnt[reference_idx]
            component_viewcos_cache = self.mesh.viewcos_cache[reference_idx]

            for component_idx in self.mesh.groups[reference_idx]:
                component_v = components[component_idx]['vertices'].to(self.mesh.device)
                component_f = components[component_idx]['faces'].to(self.mesh.device)
                component_vt = components[component_idx]['uvs'].to(self.mesh.device)
                component_vn = components[component_idx]['vn'].to(self.mesh.device)

                v_cam_component = torch.matmul(F.pad(component_v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip_component = v_cam_component @ proj.T

                rast, rast_db = dr.rasterize(self.glctx, v_clip_component, component_f, (h, w))

                texc, texc_db = dr.interpolate(component_vt.unsqueeze(0).contiguous(), rast, component_f, rast_db=rast_db, diff_attrs='all')

                albedo = dr.texture(component_albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')  # [1, H, W, 3]
                albedo = dr.antialias(albedo, rast, v_clip_component, component_f).squeeze(0).clamp(0, 1)  # [H, W, 3]

                alpha = (rast[..., 3:] > 0).float()
                alpha = dr.antialias(alpha, rast, v_clip_component, component_f).squeeze(0).clamp(0, 1)  # [H, W, 1]

                disp = -1 / (v_cam_component[..., [2]] + 1e-20)
                disp = (disp - depth_min) / (depth_max - depth_min + 1e-20) # pre-normalize
                depth, _ = dr.interpolate(disp, rast, component_f) # [1, H, W, 1]
                depth = depth.clamp(0, 1).squeeze(0) # [H, W, 1]
                per_component_depths[component_idx] = depth

                # Compute normals for this component
                normal, _ = dr.interpolate(component_vn.unsqueeze(0).contiguous(), rast, component_f)
                normal = safe_normalize(normal[0])  # [1, H, W, 3]

                # Rotate normals to align with the camera (to compute `viewcos`)
                rot_normal = normal @ pose[:3, :3]

                # Calculate viewcos (dot product between view direction and surface normal)
                viewcos = rot_normal[..., [2]].abs()  # Only care about the z-axis for viewcos

                cnt = dr.texture(component_cnt.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                cnt = dr.antialias(cnt, rast, v_clip_component, component_f).squeeze(0)
                cnt = alpha * cnt + (1 - alpha) * 1 

                viewcos_cache = dr.texture(component_viewcos_cache.unsqueeze(0), texc, uv_da=texc_db, filter_mode='linear-mipmap-linear')
                viewcos_cache = dr.antialias(viewcos_cache, rast, v_clip_component, component_f).squeeze(0)

                closer_mask = (depth > final_depth).squeeze(-1) 

                # Update the final buffer with the closer components
                final_rgb[closer_mask] = albedo[closer_mask]
                final_alpha[closer_mask] = alpha[closer_mask]
                final_depth[closer_mask] = depth[closer_mask]
                final_normal[closer_mask] = normal[closer_mask]
                final_rot_normal[closer_mask] = rot_normal[closer_mask]
                final_viewcos[closer_mask] = viewcos[closer_mask]
                final_texc[closer_mask] = texc.squeeze(0)[closer_mask]
                final_cnt[closer_mask] = cnt[closer_mask]
                final_viewcos_cache[closer_mask] = viewcos_cache[closer_mask]
                final_group[closer_mask] = reference_idx

        # Replace background for areas not covered by components
        final_rgb = final_alpha * final_rgb + (1 - final_alpha) * self.bg
        final_normal = final_alpha * final_normal + (1 - final_alpha) * self.bg_normal
        final_rot_normal = final_alpha * final_rot_normal + (1 - final_alpha) * self.bg_normal

        # Store the results
        results['image'] = final_rgb
        results['alpha'] = final_alpha
        results['depth'] = final_depth
        results['normal'] = final_normal
        results['rot_normal'] = final_rot_normal
        results['viewcos'] = final_viewcos
        results['uvs'] = final_texc
        results['per_component_depths'] = per_component_depths
        results['final_group'] = final_group
        results['cnt'] = final_cnt
        results['viewcos_cache'] = final_viewcos_cache

        return results