import os
import cv2
import time
import kiui.vis
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from .cam_utils import orbit_camera, undo_orbit_camera, OrbitCamera
from .mesh_renderer import Renderer

from scipy import ndimage
from kornia.morphology import dilation
from .grid_put import mipmap_linear_grid_put_2d, linear_grid_put_2d, nearest_grid_put_2d

import kiui

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion

from patchmatch import patch_match

def dilate_image(image, mask, iterations):
    # image: [H, W, C], current image
    # mask: [H, W], region with content (~mask is the region to inpaint)
    # iterations: int

    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    
    if mask.dtype != bool:
        mask = mask > 0.5

    inpaint_region = binary_dilation(mask, iterations=iterations)
    inpaint_region[mask] = 0

    search_region = mask.copy()
    not_search_region = binary_erosion(search_region, iterations=3)
    search_region[not_search_region] = 0

    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)

    image[tuple(inpaint_coords.T)] = image[tuple(search_coords[indices[:, 0]].T)]
    return image


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = opt.seed
        self.save_path = opt.save_path

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.buffer_overlay = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.buffer_out = None  # for 2D to 3D projection

        self.need_update = True  # update buffer_image
        self.need_update_overlay = True  # update buffer_overlay

        self.mouse_loc = np.array([0, 0])
        self.draw_mask = False
        self.draw_radius = 20
        self.mask_2d = np.zeros((self.W, self.H, 1), dtype=np.float32)

        # models
        self.device = torch.device("cuda")

        self.guidance = None
        self.guidance_embeds = None

        # renderer
        self.renderer = Renderer(self.device, opt)

        # input mesh
        if self.opt.mesh is not None:
            self.renderer.load_mesh(self.opt.mesh)
            
        # decide if only some components need to be saved, for part addition case
        if opt.partial_components is not None:
            self.renderer.mesh.partial_components = opt.partial_components

        # input text
        self.prompt = self.opt.posi_prompt + ', ' + self.opt.prompt
        self.negative_prompt = self.opt.nega_prompt

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)
        
        print(f'[INFO] seed = {seed}')

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed
    
    def prepare_guidance(self):
        
        if self.guidance is None:
            print(f'[INFO] loading guidance model...')
            if self.opt.enable_lcm:
                from .guidance.sd_lcm_utils import StableDiffusion
            else:
                from .guidance.sd_utils import StableDiffusion
            self.guidance = StableDiffusion(self.device, control_mode=self.opt.control_mode, model_key=self.opt.model_key, lora_keys=self.opt.lora_keys)
            print(f'[INFO] loaded guidance model!')

        print(f'[INFO] encoding prompt...')
        nega = self.guidance.get_text_embeds([self.negative_prompt])

        if not self.opt.text_dir:
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds = torch.cat([nega, posi], dim=0)
        else:
            self.guidance_embeds = {}
            posi = self.guidance.get_text_embeds([self.prompt])
            self.guidance_embeds['default'] = torch.cat([nega, posi], dim=0)
            for d in ['front', 'side', 'back', 'top', 'bottom']:
                posi = self.guidance.get_text_embeds([self.prompt + f', {d} view'])
                self.guidance_embeds[d] = torch.cat([nega, posi], dim=0)
        
    
    @torch.no_grad()
    def inpaint_view(self, pose, retexturing=False):

        h = w = int(self.opt.texture_size)
        H = W = int(self.opt.render_resolution)

        out = self.renderer.render(pose, self.cam.perspective, H, W)

        # valid crop region with fixed aspect ratio
        valid_pixels = out['alpha'].squeeze(-1).nonzero() # [N, 2]
        min_h, max_h = valid_pixels[:, 0].min().item(), valid_pixels[:, 0].max().item()
        min_w, max_w = valid_pixels[:, 1].min().item(), valid_pixels[:, 1].max().item()
        
        size = max(max_h - min_h + 1, max_w - min_w + 1) * 1.1
        h_start = min(min_h, max_h) - (size - (max_h - min_h + 1)) / 2
        w_start = min(min_w, max_w) - (size - (max_w - min_w + 1)) / 2

        min_h = int(h_start)
        min_w = int(w_start)
        max_h = int(min_h + size)
        max_w = int(min_w + size)

        # crop region is outside rendered image: do not crop at all.
        if min_h < 0 or min_w < 0 or max_h > H or max_w > W:
            min_h = 0
            min_w = 0
            max_h = H
            max_w = W
        
        # TODO: investiagate zooming bug for retexturing    
        min_h = 0
        min_w = 0
        max_h = H
        max_w = W

        def _zoom(x, mode='bilinear', size=(H, W)):
            return F.interpolate(x[..., min_h:max_h+1, min_w:max_w+1], size, mode=mode)

        image = _zoom(out['image'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 3, H, W]

        # trimap: generate, refine, keep
        mask_generate = _zoom(out['cnt'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') < 0.1 # [1, 1, H, W] #TODO: chekc cnt and viewcos_cache need to revise or not

        # print(out['cnt'])

        viewcos_old = _zoom(out['viewcos_cache'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        viewcos = _zoom(out['viewcos'].permute(2, 0, 1).unsqueeze(0).contiguous()) # [1, 1, H, W]
        group = _zoom(out['final_group'].permute(2, 0, 1).unsqueeze(0).contiguous(), mode='nearest') # [1, 1, H, W]
        # print(f"plotting viewcos")
        # kiui.vis.plot_image(viewcos.squeeze(-1).detach().cpu().numpy())
        # print("plotting viewcos_old")
        # kiui.vis.plot_image(viewcos_old.squeeze(-1).detach().cpu().numpy())
        mask_refine = ((viewcos_old < viewcos) & ~mask_generate)

        mask_keep = (~mask_generate & ~mask_refine)

        mask_generate = mask_generate.float()
        mask_refine = mask_refine.float()
        mask_keep = mask_keep.float()

        # dilate and blur mask
        # blur_size = 9
        # mask_generate_blur = dilation(mask_generate, kernel=torch.ones(blur_size, blur_size, device=mask_generate.device))
        # mask_generate_blur = gaussian_blur(mask_generate_blur, kernel_size=blur_size, sigma=5) # [1, 1, H, W]
        # mask_generate[mask_generate > 0.5] = 1 # do not mix any inpaint region
        mask_generate_blur = mask_generate

        # weight map for mask_generate
        # mask_weight = (mask_generate > 0.5).float().cpu().numpy().squeeze(0).squeeze(0)
        # mask_weight = ndimage.distance_transform_edt(mask_weight)#.clip(0, 30) # max pixel dist hardcoded...
        # mask_weight = (mask_weight - mask_weight.min()) / (mask_weight.max() - mask_weight.min() + 1e-20)
        # mask_weight = torch.from_numpy(mask_weight).to(self.device).unsqueeze(0).unsqueeze(0)

        # kiui.vis.plot_matrix(mask_generate, mask_refine, mask_keep)

        if not (mask_generate > 0.5).any():
            return

        if not retexturing:
            control_images = {}

            # construct normal control
            if 'normal' in self.opt.control_mode:
                rot_normal = out['rot_normal'] # [H, W, 3]
                rot_normal[..., 0] *= -1 # align with normalbae: blue = front, red = left, green = top
                control_images['normal'] = _zoom(rot_normal.permute(2, 0, 1).unsqueeze(0).contiguous() * 0.5 + 0.5, size=(512, 512)) # [1, 3, H, W]
            
            # construct depth control
            if 'depth' in self.opt.control_mode:
                depth = out['depth']
                control_images['depth'] = _zoom(depth.view(1, 1, H, W), size=(512, 512)).repeat(1, 3, 1, 1) # [1, 3, H, W]
            
            # construct ip2p control
            if 'ip2p' in self.opt.control_mode:
                ori_image = _zoom(out['ori_image'].permute(2, 0, 1).unsqueeze(0).contiguous(), size=(512, 512)) # [1, 3, H, W]
                control_images['ip2p'] = ori_image

            # construct inpaint control
            if 'inpaint' in self.opt.control_mode:
                image_generate = image.clone()
                image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
                control_images['inpaint'] = image_generate

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep
                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
            
            # construct depth-aware-inpaint control
            if 'depth_inpaint' in self.opt.control_mode:

                image_generate = image.clone()

                # image_generate[mask_generate.repeat(1, 3, 1, 1) > 0.5] = -1 # -1 is inpaint region
                image_generate[mask_keep.repeat(1, 3, 1, 1) < 0.5] = -1 # -1 is inpaint region

                image_generate = F.interpolate(image_generate, size=(512, 512), mode='bilinear', align_corners=False)
                depth = _zoom(out['depth'].view(1, 1, H, W), size=(512, 512)).clamp(0, 1).repeat(1, 3, 1, 1) # [1, 3, H, W]
                control_images['depth_inpaint'] = torch.cat([image_generate, depth], dim=1) # [1, 6, H, W]

                # mask blending to avoid changing non-inpaint region (ref: https://github.com/lllyasviel/ControlNet-v1-1-nightly/commit/181e1514d10310a9d49bb9edb88dfd10bcc903b1)
                latents_mask = F.interpolate(mask_generate_blur, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_refine = F.interpolate(mask_refine, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                latents_mask_keep = F.interpolate(mask_keep, size=(64, 64), mode='bilinear') # [1, 1, 64, 64]
                control_images['latents_mask'] = latents_mask
                control_images['latents_mask_refine'] = latents_mask_refine
                control_images['latents_mask_keep'] = latents_mask_keep

                # image_fill = image.clone()
                # image_fill = dilate_image(image_fill, mask_generate_blur, iterations=int(H*0.2))

                control_images['latents_original'] = self.guidance.encode_imgs(F.interpolate(image, (512, 512), mode='bilinear', align_corners=False).to(self.guidance.dtype)) # [1, 4, 64, 64]
            
            
            if not self.opt.text_dir:
                text_embeds = self.guidance_embeds
            else:
                # pose to view dir
                ver, hor, _ = undo_orbit_camera(pose)
                if ver <= -60: d = 'top'
                elif ver >= 60: d = 'bottom'
                else:
                    if abs(hor) < 30: d = 'front'
                    elif abs(hor) < 90: d = 'side'
                    else: d = 'back'
                text_embeds = self.guidance_embeds[d]

            # prompt to reject & regenerate
            rgbs = self.guidance(text_embeds, height=512, width=512, control_images=control_images, refine_strength=self.opt.refine_strength).float()
        else: # patchmatch
            # patch_match
            depth = out['depth']
            depth = _zoom(depth.view(1, 1, H, W), size=(H, W)).clamp(0, 1) # [1, 1, H, W]
            depth = depth.squeeze(0).squeeze(0).detach().cpu().numpy()
            
            image_generate = image.clone()
            image_generate = F.interpolate(image_generate, size=(H, W), mode='bilinear', align_corners=False)

            guidance_image = (image_generate.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy() * 255).astype(np.uint8)
            mask_generate_shrink = mask_generate.clone()
            mask_generate_shrink = F.interpolate(mask_generate_shrink, size=(H, W), mode='nearest')
            inpainting_mask = mask_generate_shrink.detach().cpu().squeeze(0).squeeze(0).contiguous().numpy().astype(np.uint8)

            # global map: each component should only get texture from the same group
            group_shrink = group.clone()
            group_shrink = F.interpolate(group_shrink, size=(H, W), mode='nearest')
            group_shrink = group_shrink.squeeze(0).squeeze(0).detach().cpu().numpy()
            inpainted_groups = np.unique(group_shrink[inpainting_mask > 0.5])
            
            inpainting_masks = {}
            for inpaint_group in inpainted_groups:
                inpainting_mask_curr = inpainting_mask.copy()
                inpainting_mask_curr[group_shrink != inpaint_group] = 0
                inpainting_mask_curr[depth == 0] = 0
                # dilate the inpainting mask
                binary_mask = (inpainting_mask_curr > 0.5).astype(np.uint8)
                
                mask_height, mask_width = binary_mask.shape
                pad_h, pad_w = int(0.2*mask_height), int(0.2*mask_width) # optional padding
                r0, r1 = pad_h, mask_height - pad_h
                c0, c1 = pad_w, mask_width - pad_w
                # dilation kernel
                radius  = 5
                kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
                # dilate only inside ROI
                dilated = binary_mask.copy()
                roi = binary_mask[r0:r1, c0:c1]
                roi_dilated = cv2.dilate(roi, kernel, iterations=1)
                dilated[r0:r1, c0:c1] = roi_dilated
                inpainting_mask_curr = dilated
                inpainting_mask_curr[group_shrink != inpaint_group] = 0
                
                inpainting_mask_curr_torch = torch.from_numpy(inpainting_mask_curr).unsqueeze(0).unsqueeze(0).contiguous().to(self.device) # [1, 1, H, W]
                inpainting_masks[inpaint_group] = _zoom(inpainting_mask_curr_torch, size=(H, W), mode="nearest").contiguous() # [1, 1, H, W]
                inpainting_masks[inpaint_group] = (inpainting_masks[inpaint_group] > 0).bool()
                
                # prepare global mask. the global mask should have 1 where the guidance_image differs the same group
                global_mask = np.zeros_like(group_shrink)
                global_mask[group_shrink != inpaint_group] = 1
                global_mask = global_mask.astype(np.uint8)
                inpainted = patch_match.inpaint(guidance_image, inpainting_mask_curr, global_mask=global_mask, patch_size=3)        
                guidance_image[inpainting_mask_curr > 0.5] = inpainted[inpainting_mask_curr > 0.5]

            rgbs = np.array(guidance_image)

            # turn rgbs into tensor
            rgbs = np.array(rgbs)
            rgbs = (rgbs / 255).astype(np.float32)
            rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        # performing upscaling (assume 2/4/8x)
        if rgbs.shape[-1] != W or rgbs.shape[-2] != H:
            scale = W // rgbs.shape[-1]
            rgbs = rgbs.detach().cpu().squeeze(0).permute(1, 2, 0).contiguous().numpy()
            rgbs = (rgbs * 255).astype(np.uint8)
            rgbs = kiui.sr.sr(rgbs, scale=scale)
            rgbs = rgbs.astype(np.float32) / 255
            rgbs = torch.from_numpy(rgbs).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.device)
        
        # apply mask to make sure non-inpaint region is not changed
        rgbs = rgbs * (1 - mask_keep) + image * mask_keep
        # rgbs = rgbs * mask_generate_blur + image * (1 - mask_generate_blur)

        # rgbs visualize
        alpha_np = rgbs.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        alpha_np = (alpha_np.clip(0, 1) * 255).astype(np.uint8)
        alpha_bgr = cv2.cvtColor(alpha_np, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"control_net.png", alpha_bgr)

        if self.opt.vis:
            if 'depth' in control_images:
                kiui.vis.plot_image(control_images['depth'])
            if 'normal' in control_images:
                kiui.vis.plot_image(control_images['normal'])
            if 'ip2p' in control_images:
                kiui.vis.plot_image(ori_image)
            # kiui.vis.plot_image(mask_generate)
            if 'inpaint' in control_images:
                kiui.vis.plot_image(control_images['inpaint'].clamp(0, 1))
                # kiui.vis.plot_image(control_images['inpaint_refine'].clamp(0, 1))
            if 'depth_inpaint' in control_images:
                kiui.vis.plot_image(control_images['depth_inpaint'][:, :3].clamp(0, 1))
                kiui.vis.plot_image(control_images['depth_inpaint'][:, 3:].clamp(0, 1))
            kiui.vis.plot_image(rgbs)


        # project-texture mask
        proj_mask = (out['alpha'] > 0) & (out['viewcos'] > self.opt.cos_thresh)  # [H, W, 1]

        # kiui.vis.plot_image(out['viewcos'].squeeze(-1).detach().cpu().numpy())
        proj_mask = _zoom(proj_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
        uvs = _zoom(out['uvs'].permute(2, 0, 1).unsqueeze(0).contiguous(), 'nearest')
        # # visualize uv
        # print(f"plotting uvs")
        # kiui.vis.plot_image(uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2).detach().cpu().numpy())
        # print(f"uv max and min: {uvs.max()}, {uvs.min()}")
        # # kiui.vis.plot_image(proj_mask.view(H, W).detach().cpu().numpy())

        uvs = uvs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 2)
        rgbs = rgbs.squeeze(0).permute(1, 2, 0).contiguous().view(-1, 3)

        mesh = self.renderer.mesh
        for reference_idx in self.renderer.mesh.groups:
            component_viewcos_cache = self.renderer.mesh.viewcos_cache[reference_idx]

            final_cur_albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            final_cur_cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)
            final_cur_viewcos = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            for component_idx in self.renderer.mesh.groups[reference_idx]:
                component_uvs = mesh.components[component_idx]['uvs']

                # Determine which UVs correspond to this component based on the per-component depth comparison
                depth_mask = (out['per_component_depths'][component_idx] == out['depth']) * out['alpha']
                depth_mask = _zoom(depth_mask.view(1, 1, H, W).float(), 'nearest').view(-1).bool()
                visible_mask = depth_mask.view(-1) & proj_mask

                if retexturing:
                    visible_mask = visible_mask & mask_generate.view(-1).bool()
                    if reference_idx not in inpainting_masks: # the group does not need re-texturing, and thus we don't need to update the cnt and albedos
                        # logging.debug(f"group {reference_idx} does not need re-texturing")
                        break
                if visible_mask.sum() > 0:
                    visible_uvs = uvs[visible_mask]
                    visible_rgbs = rgbs[visible_mask]

                    # Project the RGBs back onto the component's texture using the visible UVs
                    cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                        h, w, 
                        visible_uvs[..., [1, 0]] * 2 - 1,
                        visible_rgbs, 
                        min_resolution=128, 
                        return_count=True
                    )
                    # viewcos_values = out['viewcos'].view(-1, 1)[visible_mask]
                    viewcos_values = viewcos.view(-1, 1)[visible_mask]
                    cur_viewcos = mipmap_linear_grid_put_2d(
                        h, w, 
                        visible_uvs[..., [1, 0]] * 2 - 1, 
                        viewcos_values,
                        min_resolution=256
                    )

                    mask_current = cur_cnt.squeeze(-1) > 0
                    mask_final = final_cur_cnt.squeeze(-1) > 0
                    overlap_mask = mask_current & mask_final

                    if overlap_mask.sum() > 0:
                        better_viewcos_mask = cur_viewcos[overlap_mask] > final_cur_viewcos[overlap_mask]

                        # Handle overlap according to viewcos comparision
                        final_cur_albedo[overlap_mask][~better_viewcos_mask.expand(-1, 3)] = cur_albedo[overlap_mask][~better_viewcos_mask.expand(-1, 3)]
                        final_cur_cnt[overlap_mask][~better_viewcos_mask] = cur_cnt[overlap_mask][~better_viewcos_mask]

                    # Handle non_overlap part by keeping
                    non_overlap_mask = mask_current & ~mask_final
                    final_cur_albedo[non_overlap_mask] = cur_albedo[non_overlap_mask]
                    final_cur_cnt[non_overlap_mask] = cur_cnt[non_overlap_mask]

                    mesh.viewcos_cache[reference_idx] = torch.maximum(mesh.viewcos_cache[reference_idx], cur_viewcos)

            if retexturing: # keep the original texture
                mask = (final_cur_cnt.squeeze(-1) > 0) & (mesh.cnt[reference_idx].squeeze(-1) < 0.1)
                # mask = final_cur_cnt.squeeze(-1) > 0 // completely generate new texture, same as the original pipeline
                # print(f"updating {mask.sum()} pixels")
            else:
                mask = final_cur_cnt.squeeze(-1) > 0
            self.albedo[reference_idx][mask] += final_cur_albedo[mask]
            mesh.cnt[reference_idx][mask] += final_cur_cnt[mask]

            mask = mesh.cnt[reference_idx].squeeze(-1) > 0
            cur_albedo = self.albedo[reference_idx].clone()
            cur_albedo[mask] /= mesh.cnt[reference_idx][mask].repeat(1, 3)
            mesh.albedo[reference_idx] = cur_albedo

        # print(f'[INFO] processing {ver} - {hor}, {rgbs.shape}')
        # cur_albedo, cur_cnt = linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, rgbs, return_count=True)
        
        # albedo += cur_albedo
        # cnt += cur_cnt

        # mask = cnt.squeeze(-1) < 0.1
        # albedo[mask] += cur_albedo[mask]
        # cnt[mask] += cur_cnt[mask]

        # self.backup()

        # mask = cur_cnt.squeeze(-1) > 0
        # self.albedo[mask] += cur_albedo[mask]
        # self.cnt[mask] += cur_cnt[mask]

        # # update mesh texture for rendering
        # self.update_mesh_albedo()

        # # kiui.vis.plot_image(cur_albedo.detach().cpu().numpy())

        # # update viewcos cache
        # viewcos = viewcos.view(-1, 1)[proj_mask]
        # cur_viewcos = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, viewcos, min_resolution=256)
        # self.renderer.mesh.viewcos_cache = torch.maximum(self.renderer.mesh.viewcos_cache, cur_viewcos)
    
    @torch.no_grad()
    def backup(self):
        self.backup_albedo = self.albedo.clone()
        self.backup_cnt = self.cnt.clone()
        self.backup_viewcos_cache = self.renderer.mesh.viewcos_cache.clone()
    
    @torch.no_grad()
    def restore(self):
        self.albedo = self.backup_albedo.clone()
        self.cnt = self.backup_cnt.clone()
        self.renderer.mesh.cnt = self.cnt
        self.renderer.mesh.viewcos_cache = self.backup_viewcos_cache.clone()
        self.update_mesh_albedo()
    
    @torch.no_grad()
    def update_mesh_albedo(self):
        mask = self.cnt.squeeze(-1) > 0
        cur_albedo = self.albedo.clone()
        cur_albedo[mask] /= self.cnt[mask].repeat(1, 3)
        self.renderer.mesh.albedo = cur_albedo

    @torch.no_grad()
    def dilate_texture(self):
        h = w = int(self.opt.texture_size)

        # self.backup()
        mesh = self.renderer.mesh
        for reference_idx in self.renderer.mesh.groups:
            mask = mesh.cnt[reference_idx].squeeze(-1) > 0
            
            # rare case when there's no empty region (the full texture canvas is painted)
            if mask.all():
                continue
            
            ## dilate texture
            mask = mask.view(h, w)
            mask = mask.detach().cpu().numpy()

            mesh.albedo[reference_idx] = dilate_image(mesh.albedo[reference_idx], mask, iterations=int(h*0.2))
            mesh.cnt[reference_idx] = dilate_image(mesh.cnt[reference_idx], mask, iterations=int(h*0.2))
        
            # self.update_mesh_albedo()
            mask = mesh.cnt[reference_idx].squeeze(-1) > 0
            cur_albedo = self.albedo[reference_idx].clone()
            cur_albedo[mask] /= mesh.cnt[reference_idx][mask].repeat(1, 3)
            mesh.albedo[reference_idx] = cur_albedo
    
    @torch.no_grad()
    def deblur(self, ratio=2):
        h = w = int(self.opt.texture_size)

        # self.backup()
        mesh = self.renderer.mesh
        for reference_idx in self.renderer.mesh.groups:
            # overall deblur by LR then SR
            # kiui.vis.plot_image(self.albedo)
            cur_albedo = mesh.albedo[reference_idx].detach().cpu().numpy()
            cur_albedo = (cur_albedo * 255).astype(np.uint8)
            cur_albedo = cv2.resize(cur_albedo, (w // ratio, h // ratio), interpolation=cv2.INTER_CUBIC)
            cur_albedo = kiui.sr.sr(cur_albedo, scale=ratio)
            cur_albedo = cur_albedo.astype(np.float32) / 255
            # kiui.vis.plot_image(cur_albedo)
            cur_albedo = torch.from_numpy(cur_albedo).to(self.device)
            
            # enhance quality by SD refine...
            # kiui.vis.plot_image(albedo.detach().cpu().numpy())
            # text_embeds = self.guidance_embeds if not self.opt.text_dir else self.guidance_embeds['default']
            # albedo = self.guidance.refine(text_embeds, albedo.permute(2,0,1).unsqueeze(0).contiguous(), strength=0.8).squeeze(0).permute(1,2,0).contiguous()
            # kiui.vis.plot_image(albedo.detach().cpu().numpy())
            
            mesh.albedo[reference_idx] = cur_albedo

    @torch.no_grad()
    def initialize(self, keep_ori_albedo=False):

        self.prepare_guidance()
        
        h = w = int(self.opt.texture_size)

        self.albedo = {}

        if self.renderer.mesh.albedo == None:
            self.renderer.mesh.albedo = {}
        self.renderer.mesh.cnt = {}
        self.renderer.mesh.viewcos_cache = {}

        for reference_idx in self.renderer.mesh.groups:
            # TODO: Need to add existance check after adding repaint
            self.renderer.mesh.albedo[reference_idx] = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            self.albedo[reference_idx] = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)

            self.renderer.mesh.cnt[reference_idx] = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            self.renderer.mesh.viewcos_cache[reference_idx] = -torch.ones((h, w, 1), device=self.device, dtype=torch.float32)

        # keep original texture if using ip2p
        if 'ip2p' in self.opt.control_mode:
            self.renderer.mesh.ori_albedo = self.renderer.mesh.albedo.clone()

        if keep_ori_albedo:
            self.albedo = self.renderer.mesh.albedo.clone()
            self.cnt += 1 # set to 1
            self.viewcos_cache *= -1 # set to 1

     
    @torch.no_grad()
    def generate(self, retexturing=False):

        if not retexturing:
            self.initialize(keep_ori_albedo=False)

        # vers = [0,]
        # hors = [0,]

        if self.opt.camera_path == 'default':
            vers = [-15] * 8 + [-89.9, 89.9] + [45]
            hors = [45, 0, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.opt.camera_path == 'front':
            vers = [0] * 8 + [-89.9, 89.9] + [45]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] + [0, 0] + [0]
        elif self.opt.camera_path == 'top':
            vers = [0, -45, 45, -89.9, 89.9] + [0] + [0] * 6
            hors = [0] * 5 + [180] + [45, -45, 90, -90, 135, -135]
        elif self.opt.camera_path == 'side':
            vers = [0, 0, 0, 0, 0] + [-45, 45, -89.9, 89.9] + [-45, 0]
            hors = [0, 45, -45, 90, -90] + [0, 0, 0, 0] + [180, 180]
        else:
            raise NotImplementedError(f'camera path {self.opt.camera_path} not implemented!')
        # vers = [0, 0]
        # hors = [0, 45]
        # better to generate a top-back-view earlier
        # vers = [0, -45, -45,  0,   0, -89.9,  0,   0, 89.9,   0,    0]
        # hors = [0, 180,   0, 45, -45,     0, 90, -90,    0, 135, -135]

        start_t = time.time()

        print(f'[INFO] start generation...')
        for ver, hor in tqdm.tqdm(zip(vers, hors), total=len(vers)):
            # render image
            pose = orbit_camera(ver, hor, self.cam.radius)
            self.inpaint_view(pose, retexturing=retexturing)

            # preview
            self.need_update = True
            self.test_step()

        # self.dilate_texture()
        # self.deblur()

        torch.cuda.synchronize()
        end_t = time.time()
        print(f'[INFO] finished generation in {end_t - start_t:.3f}s!')

        self.need_update = True

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update and not self.need_update_overlay:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha', 'viewcos', 'viewcos_cache', 'cnt']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)
            
            if self.mode in ['normal', 'rot_normal']:
                buffer_image = (buffer_image + 1) / 2

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()

            self.buffer_out = out

            self.need_update = False
        
        # should update overlay
        if self.need_update_overlay:
            buffer_overlay = np.zeros_like(self.buffer_overlay)

            # draw mask 2d
            buffer_overlay += self.mask_2d * 0.5
            
            self.buffer_overlay = buffer_overlay
            self.need_update_overlay = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:

            # mix image and overlay
            buffer = np.clip(
                self.buffer_image + self.buffer_overlay, 0, 1
            )  # mix mode, sometimes unclear

            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", buffer
            )  # buffer must be contiguous, else seg fault!


    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.save_path)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")
        return path

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Inpaint", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # prompt stuff
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )
            
                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # generate texture
                with dpg.group(horizontal=True):
                    dpg.add_text("Generate: ")

                    def callback_generate(sender, app_data):
                        self.generate()
                        self.need_update = True

                    dpg.add_button(
                        label="auto",
                        tag="_button_generate",
                        callback=callback_generate,
                    )
                    dpg.bind_item_theme("_button_generate", theme_button)

                    def callback_init(sender, app_data, user_data):
                        self.initialize(keep_ori_albedo=user_data)
                        self.need_update = True

                    dpg.add_button(
                        label="init",
                        tag="_button_init",
                        callback=callback_init,
                        user_data=False,
                    )
                    dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="init_ori",
                        tag="_button_init_ori",
                        callback=callback_init,
                        user_data=True,
                    )
                    dpg.bind_item_theme("_button_init_ori", theme_button)

                    def callback_encode(sender, app_data):
                        self.prepare_guidance()

                    dpg.add_button(
                        label="encode",
                        tag="_button_encode",
                        callback=callback_encode,
                    )
                    dpg.bind_item_theme("_button_encode", theme_button)

                    def callback_inpaint(sender, app_data):
                        # inpaint current view
                        self.inpaint_view(self.cam.pose)
                        self.need_update = True

                    dpg.add_button(
                        label="inpaint",
                        tag="_button_inpaint",
                        callback=callback_inpaint,
                    )
                    dpg.bind_item_theme("_button_inpaint", theme_button)

                    def callback_undo(sender, app_data):
                        # inpaint current view
                        self.restore()
                        self.need_update = True

                    dpg.add_button(
                        label="undo",
                        tag="_button_undo",
                        callback=callback_undo,
                    )
                    dpg.bind_item_theme("_button_undo", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("Post-process: ")

                    def callback_dilate(sender, app_data):
                        self.dilate_texture()
                        self.need_update = True

                    dpg.add_button(
                        label="dilate",
                        tag="_button_dilate",
                        callback=callback_dilate,
                    )
                    dpg.bind_item_theme("_button_dilate", theme_button)

                    def callback_deblur(sender, app_data):
                        self.deblur()
                        self.need_update = True

                    dpg.add_button(
                        label="deblur",
                        tag="_button_deblur",
                        callback=callback_deblur,
                    )
                    dpg.bind_item_theme("_button_deblur", theme_button)
                
            
                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save_model(sender, app_data):
                        self.save_model()

                    dpg.add_button(
                        label="mesh",
                        tag="_button_save_model",
                        callback=callback_save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            
            # draw mask 
            with dpg.collapsing_header(label="Repaint", default_open=True):
                with dpg.group(horizontal=True):

                    def callback_toggle_draw_mask(sender, app_data):
                        self.draw_mask = not self.draw_mask
                        self.need_update_overlay = True
                    
                    def callback_reset_mask(sender, app_data):
                        self.mask_2d *= 0
                        self.need_update_overlay = True
                    
                    def callback_erase_mask(sender, app_data):
                        out = self.buffer_out
                        h = w = int(self.opt.texture_size)

                        proj_mask = (out['alpha'] > 0.1).view(-1).bool()
                        uvs = out['uvs'].view(-1, 2)[proj_mask]
                        mask_2d = torch.from_numpy(self.mask_2d).to(self.device).permute(2, 0, 1).contiguous()
                        mask_2d = F.interpolate(mask_2d.unsqueeze(0), size=out['alpha'].shape[:2], mode='nearest').squeeze(0)
                        mask_2d = mask_2d.view(-1, 1)[proj_mask]
                        # mask_2d = mipmap_linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, mask_2d, min_resolution=128)
                        mask_2d = linear_grid_put_2d(h, w, uvs[..., [1, 0]] * 2 - 1, mask_2d)
                        
                        # reset albedo and cnt
                        self.backup()
                        
                        mask = mask_2d.squeeze(-1) > 0.1
                        self.albedo[mask] = 0
                        self.cnt[mask] = 0
                        self.renderer.mesh.viewcos_cache[mask] = -1

                        # update mesh texture for rendering
                        self.update_mesh_albedo()
                        
                        # reset mask_2d too
                        self.mask_2d *= 0
                        self.need_update = True
                        self.need_update_overlay = True

                    dpg.add_checkbox(
                        label="draw",
                        default_value=self.draw_mask,
                        callback=callback_toggle_draw_mask,
                    )

                    dpg.add_button(
                        label="reset",
                        tag="_button_reset_mask",
                        callback=callback_reset_mask,
                    )
                    dpg.bind_item_theme("_button_reset_mask", theme_button)

                    dpg.add_button(
                        label="erase",
                        tag="_button_erase_mask",
                        callback=callback_erase_mask,
                    )
                    dpg.bind_item_theme("_button_erase_mask", theme_button)
                
                dpg.add_slider_int(
                    label="draw radius",
                    min_value=1,
                    max_value=100,
                    format="%d",
                    default_value=self.draw_radius,
                    callback=callback_setattr,
                    user_data="draw_radius",
                )

            # rendering options
            with dpg.collapsing_header(label="Render", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal", "rot_normal", "viewcos", "viewcos_cache", "cnt"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )


        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            if self.draw_mask:
                self.mask_2d[
                    int(self.mouse_loc[1])
                    - self.draw_radius : int(self.mouse_loc[1])
                    + self.draw_radius,
                    int(self.mouse_loc[0])
                    - self.draw_radius : int(self.mouse_loc[0])
                    + self.draw_radius,
                ] = 1

            else:
                self.cam.orbit(dx, dy)
                self.need_update = True

            self.need_update_overlay = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )
            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="InteX",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def run(self, save_model=True):
        self.generate()
        if save_model:
            self.save_model()

def generate_texture_map(opt, save_model=True):
    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.run(save_model=save_model)
        
    return gui

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/base.yaml', help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    
    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    print(opt)

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.run()
