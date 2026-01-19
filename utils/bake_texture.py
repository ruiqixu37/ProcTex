import torch
import torch.nn.functional as F
import numpy as np
import cv2

from tqdm import tqdm
from loguru import logger
from PIL import Image


anti_alias_avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=3)
anti_alias_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)

def get_texels(n, device=torch.device("cpu")):
    # Create a grid of pixel coordinates
    x = torch.linspace(0, n-1, steps=n).to(device)
    y = torch.linspace(0, n-1, steps=n).to(device)
    px, py = torch.meshgrid(x, y)

    # Create pixel point tensor
    p = torch.stack((px.flatten(), py.flatten()), dim=1)

    return p

def compute_barycentric_coords(triangles, points):
    a, b, c = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    v0, v1 = b - a, c - a

    v2 = points[:, None, :] - a[None, :, :]
    
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - (v + w)
    
    return torch.stack([u, v, w], dim=-1)

def closest_point_on_segment(p, a, b):
    ab = b - a
    ab_norm_sq = torch.sum(ab * ab, dim=-1)
    t = torch.sum((p - a) * ab, dim=-1) / ab_norm_sq
    t_clamped = torch.clamp(t, 0.0, 1.0)
    closest = a + t_clamped.unsqueeze(-1) * ab
    return closest, t_clamped

def compute_barycentric_coords_v2(triangles, points):
    a, b, c = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    v0, v1 = b - a, c - a
    v2 = points[:, None, :] - a[None, :, :]
    
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    proj = u.unsqueeze(-1) * a.unsqueeze(0) + v.unsqueeze(-1) * b.unsqueeze(0) + w.unsqueeze(-1) * c.unsqueeze(0)
    is_inside = (u >= 0) & (v >= 0) & (w >= 0)
    dist_inside = (points[:, None, :] - proj).norm(dim=-1)

    pa = points[:, None, :]
    edge_ab, t_ab = closest_point_on_segment(pa, a.unsqueeze(0), b.unsqueeze(0))
    edge_bc, t_bc = closest_point_on_segment(pa, b.unsqueeze(0), c.unsqueeze(0))
    edge_ca, t_ca = closest_point_on_segment(pa, c.unsqueeze(0), a.unsqueeze(0))

    dist_ab = (pa - edge_ab).norm(dim=-1)
    dist_bc = (pa - edge_bc).norm(dim=-1)
    dist_ca = (pa - edge_ca).norm(dim=-1)

    bary_ab = torch.stack([1 - t_ab, t_ab, torch.zeros_like(t_ab)], dim=-1)
    bary_bc = torch.stack([torch.zeros_like(t_bc), 1 - t_bc, t_bc], dim=-1)
    bary_ca = torch.stack([t_ca, torch.zeros_like(t_ca), 1 - t_ca], dim=-1)

    dists = torch.stack([dist_ab, dist_bc, dist_ca], dim=-1)
    bary_outside = torch.stack([bary_ab, bary_bc, bary_ca], dim=-2)

    min_dists, min_idxs = torch.min(dists, dim=-1)
    gather_idx = min_idxs.unsqueeze(-1).unsqueeze(-1)
    closest_bary_outside = torch.gather(bary_outside, -2, gather_idx.expand(-1, -1, 1, 3)).squeeze(-2)

    final_bary = torch.where(is_inside.unsqueeze(-1), torch.stack([u, v, w], dim=-1), closest_bary_outside)
    final_dist = torch.where(is_inside, dist_inside, min_dists)

    return final_bary, final_dist
    
def compute_uvs_from_3d(traingles_3d, triangles_uvs, points_3d, batch_size=5000, tolerance = 1e-6):
    num_points = points_3d.shape[0]
    batched_uvs = []

    for i in range(0, num_points, batch_size):
        batch_points = points_3d[i:i+batch_size]
        batch_barycentric_coords, dist = compute_barycentric_coords_v2(traingles_3d, batch_points)

        sorted_tri_idx = torch.argsort(dist, dim=1)
        first_valid_tri = sorted_tri_idx[:, 0]
        first_valid_bary_coords = batch_barycentric_coords[torch.arange(sorted_tri_idx.shape[0]), first_valid_tri]
        
        uvs = triangles_uvs[first_valid_tri]
        batch_uvs = (
            first_valid_bary_coords[:, 0:1] * uvs[:, 0, :] +
            first_valid_bary_coords[:, 1:2] * uvs[:, 1, :] +
            first_valid_bary_coords[:, 2:3] * uvs[:, 2, :]
        )  # Shape: (B, 2)
        batched_uvs.append(batch_uvs)
        
    uvs = torch.cat(batched_uvs, dim=0)
    return uvs

def compute_surface_projections_from_3d(triangles_3d, points_3d, batch_size=5000, tolerance=1e-6):
    num_points = points_3d.shape[0]
    batched_surface_points = []

    for i in range(0, num_points, batch_size):
        batch_points = points_3d[i:i+batch_size]
        batch_bary_coords, dist = compute_barycentric_coords_v2(triangles_3d, batch_points)

        sorted_tri_idx = torch.argsort(dist, dim=1)
        first_valid_tri = sorted_tri_idx[:, 0]
        first_valid_bary_coords = batch_bary_coords[torch.arange(sorted_tri_idx.shape[0]), first_valid_tri]

        tri_verts = triangles_3d[first_valid_tri]

        projected_points = (
            first_valid_bary_coords[:, 0:1] * tri_verts[:, 0, :] +
            first_valid_bary_coords[:, 1:2] * tri_verts[:, 1, :] +
            first_valid_bary_coords[:, 2:3] * tri_verts[:, 2, :]
        )  # (B, 3)

        batched_surface_points.append(projected_points)

    surface_points = torch.cat(batched_surface_points, dim=0)
    return surface_points

def inverse_map_batched(
    triangles: torch.FloatTensor,
    n: int,
    mesh,
    tolerance: float = 1e-6,
    batch_size: int = 5000,
    anti_aliasing: bool = False,
    bake_anti_aliasing: bool = False,
    aa_scale: int = 3,
    bake_aa_scale: int = 3,
    pooling_operation=anti_alias_maxpool,
    device: torch.device = torch.device("cpu"),
):
    if anti_aliasing:
        n *= aa_scale
    if bake_anti_aliasing:
        n *= bake_aa_scale

    # Get texel points
    p = get_texels(n, device=device)

    # scale triangles to pixel coordinates
    scaled_triangles = triangles * n

    with torch.no_grad():
        surface_points_full = []
        pt_idx_full = []
        tri_idx_full = []
        # for i in tqdm(range(0, p.shape[0], batch_size)):
        for i in range(0, p.shape[0], batch_size):
            batch_p = p[i:i+batch_size]
            batch_barycentric_coords = compute_barycentric_coords(scaled_triangles, batch_p)

            # Determinte which points are inside a triangle
            valid_points = torch.all((batch_barycentric_coords >= -tolerance) & (batch_barycentric_coords <= 1+tolerance), dim=2)

            # Get indices of valid points and corresponding triangles
            pt_idx, tri_idx = valid_points.nonzero(as_tuple=True)

            # Select valid barycentric coordinates
            valid_barycentric_coords = batch_barycentric_coords[pt_idx, tri_idx]

            # Select corresponding triangle vertices for each valid point
            vt_idx = mesh.faces[tri_idx].long()

            # Compute 3D coordinates for valid points
            surface_points = torch.einsum("ij,ijk->ik", valid_barycentric_coords, mesh.vertices[vt_idx])

            surface_points_full.append(surface_points)
            pt_idx_full.append(pt_idx+i)
            if anti_aliasing: # and not bake_anti_aliasing:
                # save triangle indices
                tri_idx_full.append(tri_idx)
        surface_points = torch.cat(surface_points_full, dim=0)
        pt_idx = torch.cat(pt_idx_full, dim=0)

        if anti_aliasing: # and not bake_anti_aliasing:
            # get anti-aliased valid points
            tri_idx = torch.cat(tri_idx_full, dim=0)
            texture_flat = torch.zeros((n**2), dtype=torch.float32).to(device)
            texture_flat[pt_idx] = 1
            texture = texture_flat.reshape(n, n)
            aa_texture, max_indices = pooling_operation(texture.unsqueeze(0).unsqueeze(0))
            flat_aa_texture = aa_texture.flatten()
            aa_pt_idx = torch.where(flat_aa_texture)[0]

            triangles_flat = -1 * torch.ones((n**2), dtype=torch.long).to(device)
            triangles_flat[pt_idx] = tri_idx

            # get maxpooled indices
            max_indices = max_indices.flatten()
            flat_aa_triangles = triangles_flat[max_indices]

            # get triangles
            aa_triangle_idx = flat_aa_triangles[torch.where(flat_aa_triangles!=-1)[0]]

            # rescale n
            n = n//aa_scale
            # Get texel points
            p = get_texels(n, device=device)
            # scale triangles to pixel coordinates
            scaled_triangles = triangles * n

            # get aa surface points
            aa_valid_points = p[aa_pt_idx]
            aa_surface_points_full = []

            # for i in tqdm(range(0, aa_valid_points.shape[0], batch_size)):
            for i in range(0, aa_valid_points.shape[0], batch_size):
                batch_points = aa_valid_points[i:i+batch_size]
                batch_tri_idx = aa_triangle_idx[i:i+batch_size]
                batch_tris = scaled_triangles[batch_tri_idx]

                # Compute the barycentric coordinates for valid points+triangles
                batch_barycentric_coords = compute_barycentric_coords(batch_tris, batch_points)

                # Select valid barycentric coordinates
                first_dim_idx = torch.arange(batch_barycentric_coords.shape[0])
                second_dim_idx = torch.arange(batch_barycentric_coords.shape[1])
                aa_valid_barycentric_coords = batch_barycentric_coords[first_dim_idx, second_dim_idx]

                # Select corresponding triangle vertices for each valid point
                vt_idx = mesh.faces[batch_tri_idx].long()

                # Compute 3D coordinates for valid points
                aa_surface_points = torch.einsum("ij,ijk->ik", aa_valid_barycentric_coords, mesh.vertices[vt_idx])

                aa_surface_points_full.append(aa_surface_points)
            aa_surface_points = torch.cat(aa_surface_points_full, dim=0)
            surface_points = aa_surface_points
            pt_idx = aa_pt_idx


    return surface_points, pt_idx

def bake_surface_features(
    features,
    texel_indices,
    texture_image,
    anti_aliasing=False,
    aa_scale=3,
    pooling_operation=anti_alias_avgpool,
    relative_init=None,
):
    if anti_aliasing:
        texture_image = F.interpolate(
            texture_image.unsqueeze(0).unsqueeze(0),
            scale_factor=aa_scale,
            mode="nearest",
        ).squeeze(0).squeeze(0)
    flat_texture = texture_image.flatten()
    flat_texture[texel_indices] = features
    if relative_init is not None:
        flat_relative_init = relative_init.flatten()
        flat_texture[texel_indices] += flat_relative_init[texel_indices]
    texture = flat_texture.reshape(texture_image.shape[0], texture_image.shape[1])
    if anti_aliasing:
        texture = texture.unsqueeze(0).unsqueeze(0)
        texture = pooling_operation(texture)
        texture = texture.squeeze(0).squeeze(0)
    return texture

def texture_transfer(outdir, savedir, correspondence, group_idx, target_vertices, target_faces, target_triangles_uvs, uv_vertices, texel_indices, mesh_idx, device):
    texture_image = Image.open(f"{outdir}/out_texture_group{correspondence[group_idx]}.png").convert("RGB")
    original_texture = torch.tensor(
        np.array(texture_image) / 255.0, dtype=torch.float32, device=device
    )
    retrived_uvs = compute_uvs_from_3d(target_vertices[target_faces], target_triangles_uvs, uv_vertices, 1000)
    retrived_uvs = retrived_uvs[:, [1, 0]]
    retrived_uvs = 1 - retrived_uvs
    sampled_colors = torch.nn.functional.grid_sample(
        original_texture.permute(2, 1, 0).unsqueeze(0),  # (1, C, H, W)
        (retrived_uvs * 2.0 - 1.0).view(1, -1, 1, 2),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)  # Back to (H, W, C)
    
    new_texture = torch.zeros_like(original_texture)
    all_channels = []
    for rgb_idx in range(3):
        single_channel = bake_surface_features(
            sampled_colors[:, 0, rgb_idx],
            texel_indices,
            new_texture.permute(2, 0, 1).unsqueeze(0)[0, rgb_idx, :, :].clone().detach(),
            anti_aliasing=False,
        )
        all_channels.append(single_channel)
    new_texture = torch.stack(all_channels, dim=0).unsqueeze(0)
    new_texture = new_texture.squeeze(0).permute(2, 1, 0)  # Shape: (H, W, C)
    new_texture = torch.clamp(new_texture, 0.0, 1.0)
    new_texture_image = (new_texture * 255).byte().cpu().numpy()
    
    # dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_texture = np.zeros_like(new_texture_image)
    for c in range(new_texture_image.shape[2]):
        dilated_texture[:, :, c] = cv2.dilate(new_texture_image[:, :, c], kernel)
    
    output_texture_file = f"{savedir}/mesh{mesh_idx}_group{group_idx}.png"
    Image.fromarray(dilated_texture).save(output_texture_file)