# patch_projection.py
import numpy as np
import cv2
import os
import random
import sys

# ---------------------------
# 1. CARLA → camera projection
# ---------------------------
import numpy as np
sys.path.append('/home/28s_mur@lab.graphicon.ru/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
import carla


def carla_rotation_to_matrix(rot):
    """Convert CARLA rotation to a 3×3 rotation matrix (yaw→pitch→roll)."""
    pitch = np.radians(rot.pitch)
    yaw   = np.radians(rot.yaw)
    roll  = np.radians(rot.roll)

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    # rotation order in CARLA = yaw → pitch → roll
    R = R_roll @ R_pitch @ R_yaw
    return R


def world_to_camera(world_point: carla.Location, camera: carla.Sensor):
    """
    Convert world coordinates → camera coordinates in CARLA 0.9.9
    """

    cam_transform = camera.get_transform()
    cam_loc = cam_transform.location
    cam_rot = cam_transform.rotation
    print(cam_loc)
    # Rotation matrix
    R = carla_rotation_to_matrix(cam_rot)

    # In CARLA camera coordinate system:
    # - X forward
    # - Y right
    # - Z down (OpenCV: Z forward, Y down)
    # We need OpenCV-like convention:
    R = R.T  # inverse rotation
    loc_vec = np.array([world_point.x - cam_loc.x,
                        world_point.y - cam_loc.y,
                        world_point.z - cam_loc.z])

    cam_coords = R @ loc_vec

    # Convert CARLA (x forward, y right, z up) → camera/OpenCV (z forward, x right, y down)
    x = cam_coords[0]
    y = -cam_coords[1]
    z = cam_coords[2]

    return np.array([x, y, z])



def camera_to_pixel(cam_coords, K):
    """
    Project 3D camera coords to pixel coordinates.
    K — 3×3 camera intrinsic matrix
    """
    x, y, z = cam_coords
    if z <= 0:
        return None  # Behind camera

    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]

    return int(u), int(v)


# ---------------------------
# 2. Project patch quad → 2D
# ---------------------------

def project_patch_to_image(camera, K, patch_plane_3d, patch_img, image):
    """
    Projects a flat adversarial patch (patch_img) onto the camera image using 4 3D points.

    patch_plane_3d:
        list of 4 carla.Location objects → corners of the patch in 3D (clockwise)

    patch_img:
        raw patch image (H_p, W_p, 3 or 4)

    image:
        original camera RGB image (H, W, 3)

    Returns:
        image_out — image with the patch projected onto it.
    """
    print("!!!!!")
    # Project 4 patch corners to 2D pixels
    dst_pts = []
    for wp in patch_plane_3d:
        cam_coord = world_to_camera(wp, camera)
        pix = camera_to_pixel(cam_coord, K)
        if pix is None:
            return image  # Patch behind camera → skip
        dst_pts.append(pix)

    dst_pts = np.array(dst_pts, dtype=np.float32)
    print("dst_pts =", dst_pts)

    # Source patch corners (flat image plane)
    h_p, w_p = patch_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w_p, 0],
        [w_p, h_p],
        [0, h_p]
    ], dtype=np.float32)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts)

    H_img, W_img = image.shape[:2]

    # Warp patch to camera image
    warped_patch = cv2.warpPerspective(patch_img, H, (W_img, H_img))

    # If RGBA: use alpha channel
    if warped_patch.shape[2] == 4:
        alpha = warped_patch[:, :, 3] / 255.0
        alpha = alpha[..., None]
        patch_rgb = warped_patch[:, :, :3]
        image = (image * (1 - alpha) + patch_rgb * alpha).astype(np.uint8)
        return image

    # Otherwise: simple overwrite mask on non-zero pixels
    gray = cv2.cvtColor(warped_patch, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    image[mask] = warped_patch[mask]

    return image


# ---------------------------
# 3. Utility to build intrinsics
# ---------------------------
def build_intrinsic_matrix(camera: carla.Sensor) -> np.ndarray:
    """
    Build camera intrinsic matrix K for a CARLA camera sensor.

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    where fx, fy computed from horizontal FOV and image width.
    """
    cam_bp = camera.attributes  # camera blueprint attributes

    image_w = int(cam_bp["image_size_x"])
    image_h = int(cam_bp["image_size_y"])
    fov = float(cam_bp["fov"])  # in degrees (horizontal FOV)

    # horizontal fov -> focal length in pixels
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    K = np.array([
        [focal, 0.0,   image_w / 2.0],
        [0.0,   focal, image_h / 2.0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float32)

    return K
