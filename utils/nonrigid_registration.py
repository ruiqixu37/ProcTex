import gpytoolbox
import gpytoolbox.copyleft
import trimesh
import numpy as np
from pytorch3d.loss import point_mesh_face_distance

def barycentric_weights(tri, p):
    """
    tri : (3,3)  triangle vertex positions [v0, v1, v2]
    p   : (3,)   a point inside that triangle
    returns (u,v,w) so that p = u*v0 + v*v1 + w*v2
    """
    v0, v1, v2 = tri
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p  = p  - v0

    d00 = v0v1.dot(v0v1)
    d01 = v0v1.dot(v0v2)
    d11 = v0v2.dot(v0v2)
    d20 = v0p .dot(v0v1)
    d21 = v0p .dot(v0v2)

    denom = d00 * d11 - d01 * d01
    v =  ( d11 * d20 - d01 * d21) / denom
    w =  ( d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def sample_points_with_normals(mesh, count, smooth=False):
    pts, face_idx = mesh.sample(count, return_index=True)
    if not smooth:
        return pts, mesh.face_normals[face_idx]
    # smooth normals:
    v_norms = mesh.vertex_normals
    faces   = mesh.faces[face_idx]            # (N,3)
    tri_pos = mesh.vertices[faces]            # (N,3,3)
    tri_vn  = v_norms[   faces]                # (N,3,3)
    us, vs, ws = [], [], []
    for tri, p in zip(tri_pos, pts):
        u, v, w = barycentric_weights(tri, p)
        us.append(u); vs.append(v); ws.append(w)
    u = np.array(us); v = np.array(vs); w = np.array(ws)
    normals = (tri_vn[:,0,:]*u[:,None] +
               tri_vn[:,1,:]*v[:,None] +
               tri_vn[:,2,:]*w[:,None])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return pts, normals

# usage:
# A_surf, A_normals_flat   = sample_points_with_normals(source_mesh, 1000, smooth=False)
# A_surf, A_normals_smooth = sample_points_with_normals(source_mesh, 1000, smooth=True)

def compute_cage(vertices: np.ndarray, faces: np.ndarray, num_verts=50):
    """
    vertices: (V,3)
    faces:    (F,3)
    """
    U,G = gpytoolbox.copyleft.lazy_cage(vertices,faces, num_faces=num_verts)

    return torch.from_numpy(U[np.newaxis]), torch.from_numpy(G[np.newaxis].astype(np.int64))

# hijacked from deepcage source
import torch

class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, idx, dim, out_size, fill=0.0):
        out = torch.full(out_size, fill, device=src.device, dtype=src.dtype)
        ctx.save_for_backward(idx)
        out.scatter_add_(dim, idx, src)
        ctx.mark_non_differentiable(idx)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, ograd):
        idx, = ctx.saved_tensors
        grad = torch.gather(ograd, ctx.dim, idx)
        return grad, None, None, None, None

_scatter_add = ScatterAdd.apply

def scatter_add(src, idx, dim, out_size=None, fill=0.0):
    if out_size is None:
        out_size = list(src.size())
        dim_size = idx.max().item()+1
        out_size[dim] = dim_size
    return _scatter_add(src, idx, dim, out_size, fill)

PI = 3.14159265358979323846

def mean_value_coordinates_3D(query, vertices, faces, verbose=False):
    """
    Tao Ju et.al. MVC for 3D triangle meshes
    params:
        query    (B,P,3)
        vertices (B,N,3)
        faces    (B,F,3)
    return:
        wj       (B,P,N)
    """
    B, F, _ = faces.shape
    _, P, _ = query.shape
    _, N, _ = vertices.shape
    # u_i = p_i - x (B,P,N,3)
    uj = vertices.unsqueeze(1) - query.unsqueeze(2)
    # \|u_i\| (B,P,N,1)
    dj = torch.norm(uj, dim=-1, p=2, keepdim=True)
    uj = torch.nn.functional.normalize(uj, p=2, dim=-1, eps=1e-12, out=None)

    # gather triangle B,P,F,3,3
    ui = torch.gather(uj.unsqueeze(2).expand(-1,-1,F,-1,-1),
                                   3,
                                   faces.unsqueeze(1).unsqueeze(-1).expand(-1,P,-1,-1,3))
    # li = \|u_{i+1}-u_{i-1}\| (B,P,F,3)
    li = torch.norm(ui[:,:,:,[1, 2, 0],:] - ui[:, :, :,[2, 0, 1],:], dim=-1, p=2)
    eps = 2e-5
    li = torch.where(li>=2, li-(li.detach()-(2-eps)), li)
    li = torch.where(li<=-2, li-(li.detach()+(2-eps)), li)
    # asin(x) is inf at +/-1
    # θi =  2arcsin[li/2] (B,P,F,3)
    theta_i = 2*torch.asin(li/2)
    # assert(check_values(theta_i))
    # B,P,F,1
    h = torch.sum(theta_i, dim=-1, keepdim=True)/2
    # wi← sin[θi]d{i−1}d{i+1}
    # (B,P,F,3) ci ← (2sin[h]sin[h−θi])/(sin[θ_{i+1}]sin[θ_{i−1}])−1
    ci = 2*torch.sin(h)*torch.sin(h-theta_i)/(torch.sin(theta_i[:,:,:,[1, 2, 0]])*torch.sin(theta_i[:,:,:,[2, 0, 1]]))-1

    # NOTE: because of floating point ci can be slightly larger than 1, causing problem with sqrt(1-ci^2)
    # NOTE: sqrt(x)' is nan for x=0, hence use eps
    eps = 1e-5
    ci = torch.where(ci>=1, ci-(ci.detach()-(1-eps)), ci)
    ci = torch.where(ci<=-1, ci-(ci.detach()+(1-eps)), ci)
    # si← sign[det[u1,u2,u3]]sqrt(1-ci^2)
    # (B,P,F)*(B,P,F,3)

    si = torch.sign(torch.det(ui)).unsqueeze(-1)*torch.sqrt(1-ci**2)  # sqrt gradient nan for 0
    # assert(check_values(si))
    # (B,P,F,3)
    di = torch.gather(dj.unsqueeze(2).squeeze(-1).expand(-1,-1,F,-1), 3,
                      faces.unsqueeze(1).expand(-1,P,-1,-1))
    # assert(check_values(di))
    # if si.requires_grad:
    #     vertices.register_hook(save_grad("mvc/dv"))
    #     li.register_hook(save_grad("mvc/dli"))
    #     theta_i.register_hook(save_grad("mvc/dtheta"))
    #     ci.register_hook(save_grad("mvc/dci"))
    #     si.register_hook(save_grad("mvc/dsi"))
    #     di.register_hook(save_grad("mvc/ddi"))

    # wi← (θi −c[i+1]θ[i−1] −c[i−1]θ[i+1])/(disin[θi+1]s[i−1])
    # B,P,F,3
    # CHECK is there a 2* in the denominator
    wi = (theta_i-ci[:,:,:,[1,2,0]]*theta_i[:,:,:,[2,0,1]]-ci[:,:,:,[2,0,1]]*theta_i[:,:,:,[1,2,0]])/(di*torch.sin(theta_i[:,:,:,[1,2,0]])*si[:,:,:,[2,0,1]])
    # if ∃i,|si| ≤ ε, set wi to 0. coplaner with T but outside
    # ignore coplaner outside triangle
    # alternative check
    # (B,F,3,3)
    # triangle_points = torch.gather(vertices.unsqueeze(1).expand(-1,F,-1,-1), 2, faces.unsqueeze(-1).expand(-1,-1,-1,3))
    # # (B,P,F,3), (B,1,F,3) -> (B,P,F,1)
    # determinant = dot_product(triangle_points[:,:,:,0].unsqueeze(1)-query.unsqueeze(2),
    #                           torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0],
    #                                       triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1).unsqueeze(1), dim=-1, keepdim=True).detach()
    # # (B,P,F,1)
    # sqrdist = determinant*determinant / (4 * sqrNorm(torch.cross(triangle_points[:,:,:,1]-triangle_points[:,:,:,0], triangle_points[:,:,:,2]-triangle_points[:,:,:,0], dim=-1), keepdim=True))

    wi = torch.where(torch.any(torch.abs(si) <= 1e-5, keepdim=True, dim=-1), torch.zeros_like(wi), wi)
    # wi = torch.where(sqrdist <= 1e-5, torch.zeros_like(wi), wi)

    # if π −h < ε, x lies on t, use 2D barycentric coordinates
    # inside triangle
    inside_triangle = (PI-h).squeeze(-1)<1e-4
    # set all F for this P to zero
    wi = torch.where(torch.any(inside_triangle, dim=-1, keepdim=True).unsqueeze(-1), torch.zeros_like(wi), wi)
    # CHECK is it di https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf or li http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.516.1856&rep=rep1&type=pdf
    wi = torch.where(inside_triangle.unsqueeze(-1).expand(-1,-1,-1,wi.shape[-1]), torch.sin(theta_i)*di[:,:,:,[2,0,1]]*di[:,:,:,[1,2,0]], wi)

    # sum over all faces face -> vertex (B,P,F*3) -> (B,P,N)
    wj = scatter_add(wi.reshape(B,P,-1).contiguous(), faces.unsqueeze(1).expand(-1,P,-1,-1).reshape(B,P,-1), 2, out_size=(B,P,N))

    # close to vertex (B,P,N)
    close_to_point = dj.squeeze(-1) < 1e-8
    # set all F for this P to zero
    wj = torch.where(torch.any(close_to_point, dim=-1, keepdim=True), torch.zeros_like(wj), wj)
    wj = torch.where(close_to_point, torch.ones_like(wj), wj)

    # (B,P,1)
    sumWj = torch.sum(wj, dim=-1, keepdim=True)
    sumWj = torch.where(sumWj==0, torch.ones_like(sumWj), sumWj)

    wj_normalised = wj / sumWj
    # if wj.requires_grad:
    #     saved_variables["mvc/wi"] = wi
    #     wi.register_hook(save_grad("mvc/dwi"))
    #     wj.register_hook(save_grad("mvc/dwj"))
    if verbose:
        return wj_normalised, wi
    else:
        return wj_normalised


import torch
import torch.nn.functional as F
from pytorch3d.ops import estimate_pointcloud_normals

def normal_preserving_loss(
    pc_pred: torch.Tensor,    # (B, N, 3)
    pc_gt: torch.Tensor,      # (B, N, 3)
    n_gt: torch.Tensor = None, # (B, N, 3)
    neighborhood_size: int = 20,
) -> torch.Tensor:
    """
    PCA-normal-preserving loss using PyTorch3D's estimate_pointcloud_normals.
    Args:
      pc_pred, pc_gt: predicted and GT point clouds, shape (B, N, 3)
      neighborhood_size: #neighbors for PCA
    Returns:
      Scalar loss.
    """
    # Estimate normals (uses PCA + sign disambiguation)
    if n_gt is None:
        n_gt = estimate_pointcloud_normals(pc_gt, neighborhood_size)
    n_pred = estimate_pointcloud_normals(pc_pred, neighborhood_size)

    # normalize
    n_pred = F.normalize(n_pred, dim=-1)
    n_gt = F.normalize(n_gt, dim=-1)

    cosf = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
    cos = cosf(n_pred, n_gt)
    return torch.mean(1-cos)

from functools import partial

import torch
import numpy as np

import pytorch3d
import pytorch3d.loss
from pytorch3d.loss import mesh_laplacian_smoothing, point_mesh_face_distance
from sklearn.decomposition import PCA

import tqdm
import pytorch3d.structures
from pytorch3d.structures import Meshes, Pointclouds

class MVCRegularizer(torch.nn.Module):
    """
    penalize MVC with large absolute value and negative values
    alpha * large_weight^2 + beta * (negative_weight)^2
    """
    def __init__(self, alpha=1.0, beta=1.0, threshold=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, weights):
        # ignore all weights <= 5
        # B, N, F, _ = loss.shape
        loss = 0
        if self.alpha > 0:
            large_loss = torch.log(torch.nn.functional.relu(weights.abs()-self.threshold)+1)
            # large_loss = large_loss ** 2
            loss += (torch.mean(large_loss)) * self.alpha
        if self.beta > 0:
            neg_loss = torch.nn.functional.relu(-weights)
            neg_loss = neg_loss ** 2
            loss += (torch.mean(neg_loss)) * self.beta

        return loss

def deform_with_MVC(cage: torch.Tensor, deformed_cage: torch.Tensor, cage_face: torch.LongTensor, query: torch.Tensor, verbose: bool=False):
    """
    cage (B,C,3)
    deformed_cage (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
#     weights = weights.detach()
    deformed = torch.sum(weights.unsqueeze(-1) * deformed_cage.unsqueeze(1), dim=2)
    if verbose:
        return deformed, weights, weights_unnormed
    return deformed

def optimize_cage_mvc(
    cage_init: torch.Tensor,
    cage_faces: torch.LongTensor,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    lr: float,
    plateau_threshold: float = 1e-5,
    plateau_period: int = 100,
    convergence_threshold: float = 1e-6,
    num_epochs: int = 1000,
    lap_weight: float = 0.1,
    mvc_weight: float = 1.0,
    shape_preservation_weight: float = 1.0,
    normal_preservation_weight: float = 1.0,
    align_loss_weight: float = 1.0,
    p2f_loss_weight: float = 1.0,
    x_symmetry_loss_weight: float = 1.0,
    y_symmetry_loss_weight: float = 0.0,
    z_symmetry_loss_weight: float = 1.0,
    neighborhood_size: int = 1,
    surface_penalty_weight: float = 1.0,
    surface_mesh: pytorch3d.structures.Meshes = None,
    mask: torch.Tensor = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    cage_init:     (B, 3, V) initial cage vertices
    cage_faces:    (B, F, 3)      cage face indices
    source_points: (B, N, 3)      points on source mesh
    target_points: (B, N, 3)      corresponding points on target mesh
    returns:       cage_v optimized vertices
    """

    cage_init = cage_init.clone()#.to(device)
    cage_faces = cage_faces.clone()#.to(device)

    cage_v = cage_init.clone()#.to(device)
    cage_v.requires_grad_(True)

    optimizer = torch.optim.Adam([cage_v], lr=lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(num_epochs*0.4), gamma=0.3
    )

    lap_loss_fn = partial(mesh_laplacian_smoothing, method="cot")
    align_loss_fn = partial(pytorch3d.loss.chamfer_distance)

    mvc_reg_fn = MVCRegularizer(threshold=50, beta=1.0, alpha=0.0)
    p2f_loss_fn = LocalFeatureLoss(K=neighborhood_size, reduction='mean')

    normal_loss_fn = partial(normal_preserving_loss, neighborhood_size=neighborhood_size)
    # symmetry_loss_fn = partial(normal_preserving_loss, neighborhood_size=5, angular=True)
    # normal_loss_fn = GTNormalLoss(K=20)
    # reference MVC weights from source
    weights_ref = mean_value_coordinates_3D(source_points, cage_init, cage_faces, verbose=False)
    # pbar = tqdm.tqdm(range(num_epochs))
    # pbar = range(num_epochs)

    prev_loss = 1e10
    plateau_iterations = 0
    # optimization loop
    # for i in pbar:
    for i in range(num_epochs):
        optimizer.zero_grad()
        weights = mean_value_coordinates_3D(source_points, cage_v, cage_faces, verbose=False)
        # print(source_points.shape, cage_v.shape, cage_faces.shape)
        loss = torch.tensor(0.0, device=device)

        # shape preservation loss - don't deviate too much from the reference weights
        if shape_preservation_weight > 0:
            loss += torch.mean((weights - weights_ref) ** 2)

        # optional regularizers
        reg = torch.tensor(0.0, device=device)

        if lap_weight > 0:
            # print(cage_v.shape, cage_init.shape)
            batch = Meshes([cage_v[0], cage_init[0]], [cage_faces[0], cage_faces[0]]).to(device)
            reg += lap_loss_fn(batch).mean() * lap_weight

        if mvc_weight > 0:
            reg += mvc_reg_fn(weights) * mvc_weight

        # deformed_points = deform_with_GC(cage_init, cage_v, cage_faces, source_points)
        deformed_points = deform_with_MVC(cage_init, cage_v, cage_faces, source_points)

        if align_loss_weight > 0:
            align_loss, _ = align_loss_fn(deformed_points, target_points)
            loss += align_loss * align_loss_weight

        if x_symmetry_loss_weight > 0:
            # reflect about x-axis
            reflected_target_points = deformed_points * torch.tensor([-1, 1, 1], device=device).view(1, 1, 3)
            shape_symm_loss, _ = align_loss_fn(deformed_points, reflected_target_points)
            loss += shape_symm_loss * x_symmetry_loss_weight

            # add cage reflection
            reflected_cage_v = cage_v * torch.tensor([-1, 1, 1], device=device).view(1, 1, 3)
            cage_symm_loss, _ = align_loss_fn(cage_v, reflected_cage_v)
            loss += cage_symm_loss * x_symmetry_loss_weight

        if y_symmetry_loss_weight > 0:
            reflected_target_points = target_points * torch.tensor([1, -1, 1], device=device).view(1, 1, 3)
            shape_symm_loss, _ = align_loss_fn(deformed_points, reflected_target_points)
            loss += shape_symm_loss * y_symmetry_loss_weight

            reflected_cage_v = cage_v * torch.tensor([1, -1, 1], device=device).view(1, 1, 3)
            cage_symm_loss, _ = align_loss_fn(cage_v, reflected_cage_v)
            loss += cage_symm_loss * y_symmetry_loss_weight

        if z_symmetry_loss_weight > 0:
            # reflect about z-axis
            reflected_target_points = target_points * torch.tensor([1, 1, -1], device=device).view(1, 1, 3)
            shape_symm_loss, _ = align_loss_fn(deformed_points, reflected_target_points)
            loss += shape_symm_loss * z_symmetry_loss_weight

            reflected_cage_v = cage_v * torch.tensor([1, 1, -1], device=device).view(1, 1, 3)
            cage_symm_loss, _ = align_loss_fn(cage_v, reflected_cage_v)
            loss += cage_symm_loss * z_symmetry_loss_weight

        # p2f pca surface consistency loss
        if p2f_loss_weight > 0:
            loss += p2f_loss_fn(deformed_points, target_points) * p2f_loss_weight

        # pca normal consistency loss
        if normal_preservation_weight > 0:
            normal_loss = normal_loss_fn(deformed_points, source_points)
            loss += normal_loss * normal_preservation_weight

        if surface_penalty_weight > 0:
            defpc = Pointclouds(points=[deformed_points[0]]).to(device)
            point_mesh_face_distance_loss = point_mesh_face_distance(surface_mesh, defpc)
            loss += surface_penalty_weight * point_mesh_face_distance_loss

        loss += reg

        if loss.item() < convergence_threshold:
            break

        loss.backward()


        # 3) freeze old vertices by zeroing their gradients
        if mask is not None:
            cage_v.grad[~mask] = 0
        # cage_v.grad *= mask              # grads at old‐vert indices become zero

        optimizer.step()
        scheduler.step()
        # pbar.set_description(f"Processing {loss.item():.10f}")

        if abs(prev_loss - loss.item()) < plateau_threshold:
            plateau_iterations += 1
        else:
            plateau_iterations = 0

        if plateau_iterations > plateau_period:
            break

        prev_loss = loss.item()
        # if i % 200 == 0 and i != 0:
        #     # print(cage_init.shape, cage_faces.shape)
        #     scene = visualize_cage(cage_v[0].detach().cpu(), cage_faces[0].detach().cpu())
        #     # scene.show(viewer="notebook")
        #     # scene.show()
        #     # data = scene.save_image(resolution=(512,512), line_settings= {'point_size': 20})
        #     # image = np.array(Image.open(io.BytesIO(data)))

        #     # # show image
        #     return scene
        #     # plt.imshow(image)
        #     # plt.show()
        #     break
    return cage_v.detach(), cage_faces.detach(), loss.item()

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

class LocalFeatureLoss(nn.Module):
    """
    Penalize point-to-surface deviation via local PCA normals.
    Given points xyz1, xyz2 of shape (B, N, 3):
      1. For each point, find its K nearest neighbors in the same cloud.
      2. Fit PCA (via SVD) to each local patch to estimate a normal.
      3. Project (point - patch_center) onto the normal: ptof.
      4. Apply loss on |ptof_xyz1| vs |ptof_xyz2| and penalize extra bending.
    """
    def __init__(self, K=10, reduction='mean'):
        super().__init__()
        self.K = K
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor) -> torch.Tensor:
        B, N, C = xyz1.shape

        # 1. KNN on xyz1: get distances, indices, and gathered neighbors
        knn1 = knn_points(xyz1, xyz1, K=self.K, return_nn=True)
        neigh1 = knn1.knn                 # (B, N, K, C)
        idx1   = knn1.idx                 # (B, N, K)
        # 2. Compute local patch centers and center the neighbors
        center1 = neigh1.mean(dim=2, keepdim=True)   # (B, N, 1, C)
        pts1_centered = neigh1 - center1             # (B, N, K, C)

        # 3. Flatten patches and run SVD to get normals
        patches1 = pts1_centered.view(B * N, self.K, C)   # (B*N, K, C)
        U1, S1, Vh1 = torch.linalg.svd(patches1)          # full SVD :contentReference[oaicite:3]{index=3}
        normals1 = Vh1[:, -1, :].view(B, N, C).detach()   # smallest singular vector

        # 4. Point-to-plane signed distances for xyz1
        disp1 = xyz1 - center1.squeeze(2)                 # (B, N, C)
        ptof1 = (disp1 * normals1).sum(dim=-1)            # (B, N)

        # Repeat steps for xyz2 using the **same** neighborhoods (idx1)
        neigh2 = knn_gather(xyz2, idx1)                   # (B, N, K, C)
        center2 = neigh2.mean(dim=2, keepdim=True)
        pts2_centered = neigh2 - center2
        patches2 = pts2_centered.view(B * N, self.K, C)
        U2, S2, Vh2 = torch.linalg.svd(patches2)          # :contentReference[oaicite:4]{index=4}
        normals2 = Vh2[:, -1, :].view(B, N, C).detach()

        disp2 = xyz2 - center2.squeeze(2)
        ptof2 = (disp2 * normals2).sum(dim=-1)

        # 5. Loss: match absolute point-to-plane distances
        loss_plane = self.loss_fn(ptof1.abs(), ptof2.abs())  # MSE on |ptof| :contentReference[oaicite:5]{index=5}

        # 6. Extra bending: penalize where xyz2 bends inward relative to xyz1
        bent = ptof2 - ptof1
        bent = F.relu(bent)              # zero out negative (no outward bend)
        loss_bend = self.loss_fn(bent, torch.zeros_like(bent))

        return loss_plane + 5.0 * loss_bend

def pca_scale_alignment(source_vertices: np.ndarray, target_vertices: np.ndarray):
    """
    PCA-based isotropic scaling (with rotation) alignment between source and target meshes.
    Returns the aligned source vertices and the 4x4 homogeneous transformation matrix.

    Args:
        source_vertices: (N, 3) array of source points.
        target_vertices: (M, 3) array of target points.

    Returns:
        aligned_vertices: (N, 3) array of transformed source points.
        pca_homogeneous: (4, 4) homogeneous transform matrix such that
            aligned = (pca_homogeneous @ [source; 1]).T[:,:3]
    """
    # Compute centroids
    source_center = np.mean(source_vertices, axis=0)
    target_center = np.mean(target_vertices, axis=0)

    # Center the point clouds
    src_centered = source_vertices - source_center
    tgt_centered = target_vertices - target_center

    # PCA on both
    src_pca = PCA(n_components=3).fit(src_centered)
    tgt_pca = PCA(n_components=3).fit(tgt_centered)

    # Compute isotropic scale factors along each principal axis
    scale_factors = np.sqrt(tgt_pca.explained_variance_ / src_pca.explained_variance_)

    # Build the rotation+scale in source PCA basis
    axes = src_pca.components_
    S_R = axes.T @ np.diag(scale_factors) @ axes

    # Build homogeneous transform: scale+rotate then translate
    pca_homogeneous = np.eye(4)
    pca_homogeneous[:3, :3] = S_R
    # Translation to align centroids
    pca_homogeneous[:3, 3] = target_center - S_R @ source_center

    # Apply to source vertices
    N = source_vertices.shape[0]
    homogeneous_src = np.hstack([source_vertices, np.ones((N, 1))])
    aligned = (pca_homogeneous @ homogeneous_src.T).T[:, :3]

    return aligned, pca_homogeneous

def kabsch_alignment(P, Q):
    """
    Find the optimal rotation R and translation t
    that aligns P to Q in the least‐squares sense.
    P, Q are (N,3) arrays.
    Returns:
      R: (3,3) rotation matrix
      t: (3,)  translation vector
    """
    # 1. Compute centroids
    cent_P = P.mean(axis=0)
    cent_Q = Q.mean(axis=0)
    Pc = P - cent_P
    Qc = Q - cent_Q

    # 2. Cross‐covariance
    H = Pc.T @ Qc

    # 3. SVD
    U, S, Vt = np.linalg.svd(H)

    # 4. Compute rotation
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # Reflection fix
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 5. Compute translation
    #    we want R @ cent_P + t = cent_Q  →  t = cent_Q - R @ cent_P
    t = cent_Q - R @ cent_P

    return R, t

from sklearn.neighbors import KDTree

def icp_kabsch(source, target, max_iterations=50, tol=1e-6,
               init_R=None, init_t=None):
    """
    Align `source` to `target` using ICP with Kabsch at each iteration.
    Returns aligned_source, R_total, t_total, rmse_history.
    """
    src = source.copy()

    # Optional initial guess
    if init_R is not None and init_t is not None:
        src = (init_R @ src.T).T + init_t

    tree = KDTree(target)
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = np.inf
    rmse_history = []

    for i in range(max_iterations):
        # find nearest‐neighbor correspondences
        dists, idxs = tree.query(src, k=1)
        corr = target[idxs.ravel()]

        # solve for best R, t
        R, t = kabsch_alignment(src, corr)

        # apply transform to src
        src = (R @ src.T).T + t

        # accumulate global R_total, t_total
        R_total = R @ R_total
        t_total = R @ t_total + t

        # check convergence
        rmse = np.sqrt((dists**2).mean())
        rmse_history.append(rmse)
        if abs(prev_error - rmse) < tol:
            break
        prev_error = rmse

    return src, R_total, t_total, rmse_history


import numpy as np

def umeyama_similarity(P, Q, with_scaling=True):
    """
    Estimate similarity transform (s, R, t) that maps P to Q:
        Q ≈ s * R @ P + t
    
    P, Q: (N,3) numpy arrays of corresponding points.
    with_scaling: if False, forces s=1 (pure‐rigid).
    Returns:
        s: scalar scale
        R: (3×3) rotation
        t: (3,)    translation
    """
    # 1. centroids
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    # 2. covariance
    H = P_centered.T @ Q_centered / P.shape[0]

    # 3. SVD
    U, S_values, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection check
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 4. scale
    if with_scaling:
        var_P = (P_centered**2).sum() / P.shape[0]
        # sum of singular values
        scale = S_values.sum() / var_P
    else:
        scale = 1.0

    # 5. translation: mu_Q = scale*R*mu_P + t  →  t = mu_Q - scale*R*mu_P
    t = mu_Q - scale * (R @ mu_P)

    return scale, R, t

from sklearn.neighbors import KDTree

def icp_umeyama(source, target, max_iterations=50, tol=1e-6,
               init_R=None, init_t=None):
    """
    Align `source` to `target` using ICP with Kabsch at each iteration.
    Returns aligned_source, R_total, t_total, rmse_history.
    """
    src = source.copy()

    # Optional initial guess
    if init_R is not None and init_t is not None:
        src = (init_R @ src.T).T + init_t

    tree = KDTree(target)
    s_total = 1.0
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = np.inf
    rmse_history = []

    for i in range(max_iterations):
        dists, idxs = tree.query(src, k=1)
        corr = target[idxs.ravel()]

        # estimate similarity
        s, R, t = umeyama_similarity(src, corr, with_scaling=True)

        # apply transform
        src = (s * (R @ src.T)).T + t

        # accumulate
        s_total *= s
        R_total = R @ R_total
        t_total = s * (R @ t_total) + t

        rmse = np.sqrt((dists**2).mean())
        rmse_history.append(rmse)
        if abs(prev_error - rmse) < tol:
            break
        prev_error = rmse

    return src, s_total, R_total, t_total, rmse_history

import torch

def calculate_aabb(vertices):
    """
    Calculates the axis-aligned bounding box (AABB) of a set of vertices.

    Args:
        vertices: A tensor of shape (N, 3) representing the vertices.

    Returns:
        A tuple containing two tensors:
        - min_bounds: A tensor of shape (3,) representing the minimum bounds of the AABB.
        - max_bounds: A tensor of shape (3,) representing the maximum bounds of the AABB.
    """
    min_bounds = torch.min(vertices, dim=0).values
    max_bounds = torch.max(vertices, dim=0).values
    return min_bounds, max_bounds

# # Example usage (assuming 'vertices' is defined as in your original code)
# min_bounds, max_bounds = calculate_aabb(sphere_verts)  # Or cube_verts, proj_pts, etc.

# print("Minimum bounds:", min_bounds)
# print("Maximum bounds:", max_bounds)

def ensure_outward_winding(verts, faces):
    """
    verts: list of (x,y,z)
    faces: list of (i0,i1,i2)
    Returns a new list of faces with all normals pointing outward.
    """
    V = np.array(verts)
    center = V.mean(axis=0)
    new_faces = []
    for (i0, i1, i2) in faces:
        A, B, C = V[i0], V[i1], V[i2]
        normal = np.cross(B - A, C - A)
        face_center = (A + B + C) * (1.0/3.0)
        # if normal points towards center, flip it
        if np.dot(normal, face_center - center) < 0:
            new_faces.append((i0, i2, i1))
        else:
            new_faces.append((i0, i1, i2))
    return new_faces

def subdivided_cube(splits_x, splits_y, splits_z):
    # 1) build full coordinate lists
    xs = [0.0] + sorted(splits_x) + [1.0]
    ys = [0.0] + sorted(splits_y) + [1.0]
    zs = [0.0] + sorted(splits_z) + [1.0]

    verts = []
    vert_idx = {}  # maps (x,y,z) → index
    faces = []

    def add_vert(p):
        if p not in vert_idx:
            vert_idx[p] = len(verts)
            verts.append(p)
        return vert_idx[p]

    def build_face(fixed_axis, fixed_val, var_axes):
        """
        fixed_axis: 0 for X-face, 1 for Y-face, 2 for Z-face
        fixed_val:  0.0 or 1.0
        var_axes:   tuple of the other two axes, in the order (u,v)
        """
        U, V = var_axes
        coords_u = [ (xs if U==0 else ys) if U<2 else zs ][0]
        coords_v = [ (xs if V==0 else ys) if V<2 else zs ][0]

        # sample grid of points on that face
        idx_grid = []
        for i, u in enumerate(coords_u):
            row = []
            for j, v in enumerate(coords_v):
                p = [None, None, None]
                p[fixed_axis] = fixed_val
                p[U], p[V] = u, v
                row.append(add_vert(tuple(p)))
            idx_grid.append(row)

        # build quads → two tris each
        nu, nv = len(coords_u)-1, len(coords_v)-1
        for i in range(nu):
            for j in range(nv):
                a = idx_grid[i  ][j]
                b = idx_grid[i+1][j]
                c = idx_grid[i+1][j+1]
                d = idx_grid[i  ][j+1]
                # choose winding so normal points outward
                if fixed_val == 1.0:
                    faces.append((a,b,c))
                    faces.append((a,c,d))
                else:
                    faces.append((a,c,b))
                    faces.append((a,d,c))

    # build all 6 faces
    # +X face:
    build_face(0, 1.0, (2,1))
    # -X face:
    build_face(0, 0.0, (1,2))
    # +Y face:
    build_face(1, 1.0, (0,2))
    # -Y face:
    build_face(1, 0.0, (2,0))
    # +Z face:
    build_face(2, 1.0, (0,1))
    # -Z face:
    build_face(2, 0.0, (1,0))
    faces = ensure_outward_winding(verts, faces) # lol...
    # remap verts from [0,1] to [-1,1]
    V = np.array(verts, dtype=np.float32)
    V = V * 2.0 - 1.0

    return torch.from_numpy(V), torch.from_numpy(np.array(faces))

def compute_cage_aabb(V, splits_per_axis=[.5], margin=.01):
    min_bounds, max_bounds = calculate_aabb(V)  # Or cube_verts, proj_pts, etc.
    VC, FC = subdivided_cube(splits_per_axis, splits_per_axis, splits_per_axis)
    # print(VC.shape, FC.shape)
    # print(min_bounds, max_bounds)

    # 2
    # VC ∈ [-1,1]

    # 3) expand bounds by margin (so cage sits slightly outside)
    extents   = max_bounds - min_bounds                        # length-3
    half_size = extents * 0.5 * (1.0 + margin)       # half-size of expanded box
    center    = (max_bounds + min_bounds) * 0.5                # center of original box

    # 4) transform each cage-vertex from [-1,1] to world space:
    #    world_v = center + VC * half_size
    #    (since VC==−1 maps to center−half_size, VC==+1 maps to center+half_size)
    VC_world = center + VC * half_size  # if PyTorch: broadcast to shape [N,3]

    return VC_world.float().unsqueeze(0), FC.unsqueeze(0)

def nonrigid_registration(registration_points: torch.Tensor,
                          source_mesh: trimesh.Trimesh,
                          target_mesh: trimesh.Trimesh,
                          target_normals: torch.Tensor = None,
                          init_subdiv_level=1,
                          lr: float = 1e-3,
                          plateau_threshold: float = 1e-5,
                          plateau_period: int = 100,
                          convergence_threshold: float = 1e-5,
                          num_epochs: int = 1000,
                          num_surface_samples: int = 1000,
                          lap_weight: float = 0.,
                          mvc_weight: float = .05,
                          shape_preservation_weight: float = 0.,
                          normal_preservation_weight: float = 5,
                          align_loss_weight: float = 10.,
                          p2f_loss_weight: float = 1.,
                          x_symmetry_loss_weight = 0.,
                          y_symmetry_loss_weight = 0.,
                          z_symmetry_loss_weight = 0.,
                          neighborhood_size: int = 1,
                          surface_penalty_weight: float = 1.0,
                          use_umeyama: bool = True,
                          mask: torch.Tensor | None = None,
                          device=None):

    # Sample surface points from both meshes
    A_surface = source_mesh.sample(num_surface_samples)
    B_surface = target_mesh.sample(num_surface_samples)
    
    # Sample normals from target mesh
    B_normals_smooth = sample_points_with_normals(target_mesh, num_surface_samples, smooth=True)[1]

    # Apply alignment based on use_umeyama flag
    if use_umeyama:
        A_surface, s_final, R_final, t_final, history = icp_umeyama(
            source=A_surface,
            target=B_surface,
            max_iterations=100,
            tol=1e-8
        )
        s_final = torch.from_numpy(np.array([s_final])).float().to(device)
        R_final = torch.from_numpy(R_final).float().to(device)
        t_final = torch.from_numpy(t_final).float().to(device)
    else:
        # Skip alignment - use identity transformation
        s_final = torch.tensor(1.0).float().to(device)
        R_final = torch.eye(3).float().to(device)
        t_final = torch.zeros(3).float().to(device)

    source_points = torch.from_numpy(A_surface.astype(np.float32)).unsqueeze(0).to(device)
    target_points = torch.from_numpy(B_surface.astype(np.float32)).unsqueeze(0).to(device)
    target_normals = torch.from_numpy(B_normals_smooth.astype(np.float32)).unsqueeze(0).to(device)

    A = source_mesh.vertices        # (V,3)
    AF = source_mesh.faces           # (F,3)

    B = target_mesh.vertices        # (V,3)
    BF = target_mesh.faces          # (F,3)

    A = np.array(A.astype(np.float32))
    AF = np.array(AF.astype(np.int64))

    B = np.array(B.astype(np.float32))
    BF = np.array(BF.astype(np.int64))

    registration_points = (s_final * (registration_points @ R_final)) + t_final

    B = torch.from_numpy(B).float().unsqueeze(0).to(device)   # -> [1, V, 3]
    BF = torch.from_numpy(BF).long().unsqueeze(0).to(device)   # -> [1, F, 3]

    surface_mesh = Meshes(B, BF)

    # 1) build the geometric cage at this level
    if init_subdiv_level == 0:
        # 0‐subdivision (one box)
        S_new, F_new = compute_cage_aabb(torch.from_numpy(source_mesh.vertices),
                                    splits_per_axis=[], margin=0.1)
    else:
        # ℓ‐subdivision (more splits)
        splits = np.linspace(0, 1, num=init_subdiv_level+2)[1:-1].tolist()
        S_new, F_new = compute_cage_aabb(torch.from_numpy(source_mesh.vertices),
                                    splits_per_axis=splits, margin=0.1)
    S_new = S_new.to(device).float()
    F_new = F_new.to(device)

    # print(f"init_subdiv_level: {init_subdiv_level} cage_source: {cage_source.shape}, cage_source_faces: {cage_source_faces.shape}")
    # cage_source, cage_source_faces = compute_cage_aabb(torch.from_numpy(A), [.25,0.5,.75], 0.1)
    # cage_source, cage_source_faces = compute_cage_aabb(torch.from_numpy(A), [.3,0.7], 0.1)

        # cage_source, cage_source_faces = compute_cage_aabb(torch.from_numpy(A), [.1,.25,0.5,.75,.9], 0.1)
    # B = torch.from_numpy(B).float().unsqueeze(0).to(device)   # -> [1, V, 3]
    # BF = torch.from_numpy(BF).long().unsqueeze(0).to(device)   # -> [1, F, 3]

    surface_mesh = Meshes(B, BF)

    init_V = S_new.clone()

    V_opt, F_opt, loss = optimize_cage_mvc(
        cage_init = init_V,
        cage_faces = F_new,
        source_points = source_points.to(device),
        target_points = target_points.to(device),
        target_normals = target_normals.to(device),
        lr = lr,
        plateau_threshold = plateau_threshold,
        plateau_period = plateau_period,
        convergence_threshold = convergence_threshold,
        num_epochs = num_epochs,
        lap_weight = lap_weight,
        mvc_weight = mvc_weight,
        shape_preservation_weight = shape_preservation_weight,
        normal_preservation_weight = normal_preservation_weight,
        align_loss_weight = align_loss_weight,
        p2f_loss_weight = p2f_loss_weight,
        x_symmetry_loss_weight = x_symmetry_loss_weight,
        y_symmetry_loss_weight = y_symmetry_loss_weight,
        z_symmetry_loss_weight = z_symmetry_loss_weight,
        neighborhood_size = neighborhood_size,
        surface_penalty_weight = surface_penalty_weight,
        surface_mesh = surface_mesh,
        mask = mask,
        device = device
    )

    registration_points = registration_points
    print(f"registration_points: {registration_points.shape}")
    remapped_points = deform_with_MVC(
        S_new.to(device),
        V_opt.to(device),
        F_new.long().to(device),
        registration_points.to(device),
        verbose=False
    )

    return remapped_points

if __name__ == "__main__":

  # Load the two meshes
  source_mesh = trimesh.load("/home/rxu37/GitHub/CompInpaint/procedural_data_eg/procedural_pumpkin/simplified/0.obj")
  # source_mesh = trimesh.load("mesh0_component0.obj")
  target_mesh = trimesh.load("/home/rxu37/GitHub/CompInpaint/procedural_data_eg/procedural_pumpkin/simplified/44.obj")
  # target_mesh = trimesh.load("mesh5_component0.obj")
  cage_points = 250
  num_samples = 1000
  learning_rate = 5e-3 # works for the chair, 1e-4 is more appropriate for vases i've tested
  
  A_surface = source_mesh.sample(num_samples)
  B_surface = target_mesh.sample(num_samples)
  # A_surface, A_ind = source_mesh.sample(num_samples, return_index=True)
  # B_surface, B_ind = target_mesh.sample(num_samples, return_index=True)
  
  # A_surface, _ = pca_scale_alignment(A_surface, B_surface)
  
  # Extract vertices from the meshes
  A = source_mesh.vertices        # (V,3)
  B = target_mesh.vertices        # (V,3)
  AF = source_mesh.faces           # (F,3)
  BF = target_mesh.faces          # (F,3)
  
  A = np.array(A)
  B = np.array(B)
  AF = np.array(AF)
  BF = np.array(BF)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  source_points = torch.from_numpy(A_surface.astype(np.float32)).unsqueeze(0).to(device)
  target_points = torch.from_numpy(B_surface.astype(np.float32)).unsqueeze(0).to(device)
#   target_normals = torch.from_numpy(B_normals_smooth.astype(np.float32)).unsqueeze(0).to(device)
  
  cage_source, cage_source_faces = compute_cage(A, AF, cage_points)
  cage_target, cage_target_faces = compute_cage(B, BF, cage_points)
  
  cage_source = cage_source.to(device)
  cage_source_faces = cage_source_faces.to(device)
  
  #   deformed_points = nonrigid_registration(
  #     source_points,#double check
  #     source_mesh,
  #     target_mesh,
  #     lr = learning_rate,
  #     num_epochs = 1500,
  #     lap_weight = 0.,
  #     mvc_weight = .05,
  #     shape_preservation_weight = 0.,
  #     normal_preservation_weight = 1.0,
  #     align_loss_weight = 10.,
  #     p2f_loss_weight = 1.,
  #     device = device
  #   )
  
  #   pct = trimesh.PointCloud(A_surface, colors=np.zeros((num_samples, 3)) + np.array([255, 0, 0]))
  pct = trimesh.PointCloud(B_surface, colors=np.zeros((num_samples, 3)) + np.array([0, 0, 255]))
  #   pct2 = trimesh.PointCloud(deformed_points[0].cpu().numpy(), colors=np.zeros((num_samples, 3)) + np.array([0, 0, 255]))
  
  scene = trimesh.Scene()
  scene.add_geometry(pct, node_name="source_points", geom_name="source_points")
  #   scene.add_geometry(pct2, node_name="remapped_points", geom_name="remapped_points")
  scene.show(flags={'wireframe': True, 'point_size': 50})
