from typing import List

import numpy as np
import torch
from pytorch3d.renderer import NDCMultinomialRaysampler, ray_bundle_to_ray_points
from pytorch3d.utils import cameras_from_opencv_projection
from torch.nn import functional as F


def get_position_map_from_depth(depth, mask, intrinsics, extrinsics, image_wh=None):
    """Compute the position map from the depth map and the camera parameters for a batch of views.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        intrinsics (torch.Tensor): The camera intrinsics matrices with the shape (B, 3, 3).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    u_coord, v_coord = torch.meshgrid(
        torch.arange(image_wh[0]), torch.arange(image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map by back-projecting depth pixels to 3D space
    x = (
        (u_coord - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (v_coord - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_position_map_from_depth_ortho(
    depth, mask, extrinsics, ortho_scale, image_wh=None
):
    """Compute the position map from the depth map and the camera parameters for a batch of views
    using orthographic projection with a given ortho_scale.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        ortho_scale (torch.Tensor): The scaling factor for the orthographic projection with the shape (B, 1, 1, 1).
        image_wh (Tuple[int, int]): Optional. The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    # Generating grid of coordinates in the image space
    u_coord, v_coord = torch.meshgrid(
        torch.arange(0, image_wh[0]), torch.arange(0, image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map using orthographic projection with ortho_scale
    x = (u_coord - image_wh[0] / 2) / ortho_scale / image_wh[0]
    y = (v_coord - image_wh[1] / 2) / ortho_scale / image_wh[1]
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_opencv_from_blender(matrix_world, fov=None, image_size=None):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]

    if fov is None:  # orthographic camera
        return R, T

    R, T = R.unsqueeze(0), T.unsqueeze(0)
    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = (
        torch.from_numpy(intrinsics).unsqueeze(0).float().to(matrix_world.device)
    )
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2]).to(
        matrix_world.device
    )
    opencv_cam_matrix[:, [0, 1], [0, 1]] *= image_size / 2

    return R, T, opencv_cam_matrix


def compute_plucker_embed(camera, image_width, image_height):
    """Computes Plucker coordinates for a Pytorch3D camera."""

    # get camera center
    cam_pos = camera.get_camera_center()

    # get ray bundle
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=image_width,
        image_height=image_height,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(camera)
    ray_dirs = F.normalize(src_ray_bundle.directions, dim=-1)
    cross = torch.cross(cam_pos[:, None, None, :], ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)
    plucker = plucker.permute(0, 3, 1, 2)

    return plucker  # (B, 6, H, W)


def get_plucker_embeds_from_cameras(
    c2w: List[torch.Tensor], fov: List[float], image_size: int
):
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.

    Parameters:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image

    Returns:
        plucker_embeds: plucker embeddings (B, 6, H, W)
    """
    plucker_embeds = []
    # compute pairwise mask and plucker embeddings
    for cam_matrix, cam_fov in zip(c2w, fov):
        # get pytorch3d frames (blender to opencv, then opencv to pytorch3d)
        R, T, intrinsics = get_opencv_from_blender(cam_matrix, cam_fov, image_size)
        camera_pytorch3d = cameras_from_opencv_projection(
            R,
            T,
            intrinsics,
            torch.tensor([image_size, image_size])
            .float()
            .unsqueeze(0)
            .to(cam_matrix.device),
        )

        plucker = compute_plucker_embed(
            camera_pytorch3d, image_size, image_size
        ).squeeze()
        plucker_embeds.append(plucker)

    plucker_embeds = torch.stack(plucker_embeds)

    return plucker_embeds


def get_plucker_embeds_from_cameras_ortho(
    c2w: List[torch.Tensor], ortho_scale: List[float], image_size: int
):
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.

    Parameters:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image

    Returns:
        plucker_embeds: plucker embeddings (B, 6, H, W)
    """
    plucker_embeds = []
    # compute pairwise mask and plucker embeddings
    for cam_matrix, scale in zip(c2w, ortho_scale):
        # get pytorch3d frames (blender to opencv, then opencv to pytorch3d)
        R, T = get_opencv_from_blender(cam_matrix)
        cam_pos = -R.T @ T
        view_dir = R.T @ torch.tensor([0, 0, 1]).float().to(cam_matrix.device)
        # normalize camera position
        cam_pos = F.normalize(cam_pos, dim=0)
        plucker = torch.concat([view_dir, cam_pos])
        plucker = plucker.unsqueeze(-1).unsqueeze(-1).repeat(1, image_size, image_size)
        plucker_embeds.append(plucker)

    plucker_embeds = torch.stack(plucker_embeds)

    return plucker_embeds
