from matplotlib import pyplot as plt
import numpy as np
import torch

import numpy as np
from typing import List
import sys
sys.path.append('./submodules/gaussian-splatting/')
from scene.cameras import Camera
from PIL import Image
import imageio
from scipy.interpolate import splprep, splev

def render_gaussians_rgb(generator3DGS, viewpoint_cam, visualize=False):
    """
    Simply render gaussians from the generator3DGS from the viewpoint_cam.
    Args:
        generator3DGS : instance of the Generator3DGS class from the networks.py file
        viewpoint_cam : camera instance
        visualize : boolean flag. If True, will call pyplot function and render image inplace
    Returns:
        uint8 numpy array with shape (H, W, 3) representing the image
    """
    with torch.no_grad():
        render_pkg = generator3DGS(viewpoint_cam)
        image = render_pkg["render"]
        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()

        return image_np

def render_gaussians_D_scores(generator3DGS, viewpoint_cam, mask=None, mask_channel=0, visualize=False):
    """
        Simply render D_scores of gaussians from the generator3DGS from the viewpoint_cam.
        Args:
            generator3DGS : instance of the Generator3DGS class from the networks.py file
            viewpoint_cam : camera instance
            visualize : boolean flag. If True, will call pyplot function and render image inplace
            mask : optional mask to highlight specific gaussians. Must be of shape (N) where N is the numnber
                of gaussians in generator3DGS.gaussians. Must be a torch tensor of floats, please scale according
                to how much color you want to have. Recommended mask value is 10.
            mask_channel: to which color channel should we add mask
        Returns:
            uint8 numpy array with shape (H, W, 3) representing the generator3DGS.gaussians.D_scores rendered as colors
        """
    with torch.no_grad():
        # Visualize D_scores
        generator3DGS.gaussians._features_dc = generator3DGS.gaussians._features_dc * 1e-4 + \
                                               torch.stack([generator3DGS.gaussians.D_scores] * 3, axis=-1)
        generator3DGS.gaussians._features_rest = generator3DGS.gaussians._features_rest * 1e-4
        if mask is not None:
            generator3DGS.gaussians._features_dc[..., mask_channel] += mask.unsqueeze(-1)
        render_pkg = generator3DGS(viewpoint_cam)
        image = render_pkg["render"]
        image_np = image.clone().detach().cpu().numpy().transpose(1, 2, 0)

        # Clip values to be in the range [0, 1]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()

        if mask is not None:
            generator3DGS.gaussians._features_dc[..., mask_channel] -= mask.unsqueeze(-1)

        generator3DGS.gaussians._features_dc = (generator3DGS.gaussians._features_dc - \
                                                     torch.stack([generator3DGS.gaussians.D_scores] * 3, axis=-1)) * 1e4
        generator3DGS.gaussians._features_rest = generator3DGS.gaussians._features_rest * 1e4

        return image_np
    


def normalize(v):
    """
    Normalize a vector to unit length.

    Parameters:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Unit vector in the same direction as `v`.
    """
    return v / np.linalg.norm(v)

def look_at_rotation(camera_position: np.ndarray, target: np.ndarray, world_up=np.array([0, 1, 0])):
    """
    Compute a rotation matrix for a camera looking at a target point.

    Parameters:
        camera_position (np.ndarray): The 3D position of the camera.
        target (np.ndarray): The point the camera should look at.
        world_up (np.ndarray): A vector that defines the global 'up' direction.

    Returns:
        np.ndarray: A 3x3 rotation matrix (camera-to-world) with columns [right, up, forward].
    """
    z_axis = normalize(target - camera_position)         # Forward direction
    x_axis = normalize(np.cross(world_up, z_axis))       # Right direction
    y_axis = np.cross(z_axis, x_axis)                    # Recomputed up
    return np.stack([x_axis, y_axis, z_axis], axis=1)

    
def generate_circular_camera_path(existing_cameras: List[Camera], N: int = 12, radius_scale: float = 1.0, d: float = 2.0) -> List[Camera]:
    """
    Generate a circular path of cameras around an existing camera group, 
    with each new camera oriented to look at the average viewing direction.

    Parameters:
        existing_cameras (List[Camera]): List of existing camera objects to estimate average orientation and layout.
        N (int): Number of new cameras to generate along the circular path.
        radius_scale (float): Scale factor to adjust the radius of the circle.
        d (float): Distance ahead of each camera used to estimate its look-at point.

    Returns:
        List[Camera]: A list of newly generated Camera objects forming a circular path and oriented toward a shared view center.
    """
    # Step 1: Compute average camera position
    center = np.mean([cam.T for cam in existing_cameras], axis=0)

    # Estimate where each camera is looking
    # d denotes how far ahead each camera sees — you can scale this
    look_targets = [cam.T + cam.R[:, 2] * d for cam in existing_cameras]
    center_of_view = np.mean(look_targets, axis=0)

    # Step 2: Define circular plane basis using fixed up vector
    avg_forward = normalize(np.mean([cam.R[:, 2] for cam in existing_cameras], axis=0))
    up_guess = np.array([0, 1, 0])
    right = normalize(np.cross(avg_forward, up_guess))
    up = normalize(np.cross(right, avg_forward))

    # Step 3: Estimate radius
    avg_radius = np.mean([np.linalg.norm(cam.T - center) for cam in existing_cameras]) * radius_scale

    # Step 4: Create cameras on a circular path
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reference_cam = existing_cameras[0]
    new_cameras = []

    
    for i, a in enumerate(angles):
        position = center + avg_radius * (np.cos(a) * right + np.sin(a) * up)
        
        R = look_at_rotation(position, center_of_view)
        new_cameras.append(Camera(
            R=R, #reference_cam.R.copy() * 0.9 + R * 0.1,                     # Use same orientation
            T=position,                                   # New position
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"circular_a={a:.3f}",
            uid=i
        ))

    return new_cameras


def save_numpy_frames_as_gif(frames, output_path="animation.gif", duration=100):
    """
    Save a list of RGB NumPy frames as a looping GIF animation.

    Parameters:
        frames (List[np.ndarray]): List of RGB images as uint8 NumPy arrays (shape HxWx3).
        output_path (str): Path to save the output GIF.
        duration (int): Duration per frame in milliseconds.

    Returns:
        None
    """
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,  # duration per frame in ms
        loop=0
    )
    print(f"GIF saved to: {output_path}")

def center_crop_frame(frame: np.ndarray, crop_fraction: float) -> np.ndarray:
    """
    Crop the central region of the frame by the given fraction.

    Parameters:
        frame (np.ndarray): Input RGB image (H, W, 3).
        crop_fraction (float): Fraction of the original size to retain (e.g., 0.8 keeps 80%).

    Returns:
        np.ndarray: Cropped RGB image.
    """
    if crop_fraction >= 1.0:
        return frame

    h, w, _ = frame.shape
    new_h, new_w = int(h * crop_fraction), int(w * crop_fraction)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    return frame[start_y:start_y + new_h, start_x:start_x + new_w, :]



def generate_smooth_closed_camera_path(existing_cameras: List[Camera], N: int = 120, d: float = 2.0, s=.25) -> List[Camera]:
    """
    Generate a smooth, closed path interpolating the positions of existing cameras.

    Parameters:
        existing_cameras (List[Camera]): List of existing cameras.
        N (int): Number of points (cameras) to sample along the smooth path.
        d (float): Distance ahead for estimating the center of view.

    Returns:
        List[Camera]: A list of smoothly moving Camera objects along a closed loop.
    """
    # Step 1: Extract camera positions
    positions = np.array([cam.T for cam in existing_cameras])
    
    # Step 2: Estimate center of view
    look_targets = [cam.T + cam.R[:, 2] * d for cam in existing_cameras]
    center_of_view = np.mean(look_targets, axis=0)

    # Step 3: Fit a smooth closed spline through the positions
    positions = np.vstack([positions, positions[0]])  # close the loop
    tck, u = splprep(positions.T, s=s, per=True)  # periodic=True for closed loop

    # Step 4: Sample points along the spline
    u_fine = np.linspace(0, 1, N)
    smooth_path = np.stack(splev(u_fine, tck), axis=-1)

    # Step 5: Generate cameras along the smooth path
    reference_cam = existing_cameras[0]
    new_cameras = []

    for i, pos in enumerate(smooth_path):
        R = look_at_rotation(pos, center_of_view)
        new_cameras.append(Camera(
            R=R,
            T=pos,
            FoVx=reference_cam.FoVx,
            FoVy=reference_cam.FoVy,
            resolution=(reference_cam.image_width, reference_cam.image_height),
            colmap_id=-1,
            depth_params=None,
            image=Image.fromarray(np.zeros((reference_cam.image_height, reference_cam.image_width, 3), dtype=np.uint8)),
            invdepthmap=None,
            image_name=f"smooth_path_i={i}",
            uid=i
        ))

    return new_cameras


def save_numpy_frames_as_mp4(frames, output_path="animation.mp4", fps=10, center_crop: float = 1.0):
    """
    Save a list of RGB NumPy frames as an MP4 video with optional center cropping.

    Parameters:
        frames (List[np.ndarray]): List of RGB images as uint8 NumPy arrays (shape HxWx3).
        output_path (str): Path to save the output MP4.
        fps (int): Frames per second for playback speed.
        center_crop (float): Fraction (0 < center_crop <= 1.0) of central region to retain. 
                             Use 1.0 for no cropping; 0.8 to crop to 80% center region.

    Returns:
        None
    """
    with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames:
            cropped = center_crop_frame(frame, center_crop)
            writer.append_data(cropped)
    print(f"MP4 saved to: {output_path}")