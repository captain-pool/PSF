# Copied from ShapeFlow

"""Rendering utility functions.
"""
import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import pyrender  # noqa: E402


def render_cloud(
    batched_cloud,
    eye,
    center,
    world_up,
    res=(640, 640),
    light_intensity=3.0,
    ambient_intensity=0.5,
    **kwargs,
):
    """Render a shapenet mesh using default settings.

    Args:
      trimesh_mesh: trimesh mesh instance, or a list of trimesh meshes
        (or point clouds).
      eye: array with shape [3,] containing the XYZ world
        space position of the camera.
      center: array with shape [3,] containing a position
        along the center of the camera's gaze.
      world_up: np.float32 array with shape [3,] specifying the
        world's up direction; the output camera will have no tilt with respect
        to this direction.
      res: 2-tuple of int, [width, height], resolution (in pixels) of output
        images.
      light_intensity: float, light intensity.
      ambient_intensity: float, ambient light intensity.
      kwargs: additional flags to pass to pyrender renderer.
    Returns:
      color_img: [*res, 3] color image.
      depth_img: [*res, 1] depth image.
      world_to_cam: [4, 4] camera to world matrix.
      projection_matrix: [4, 4] projection matrix, aka cam_to_img matrix.
    """
    eye = list2npy(eye).astype(np.float32)
    center = list2npy(center).astype(np.float32)
    world_up = list2npy(world_up).astype(np.float32)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # setup camera pose matrix
    scene = pyrender.Scene(
        ambient_light=ambient_intensity*np.ones([3], dtype=float)
    )


    world_to_cam = look_at(eye[None], center[None], world_up[None])
    world_to_cam = world_to_cam[0]
    cam_pose = np.linalg.inv(world_to_cam)
    scene.add(camera, pose=cam_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.SpotLight(
        color=np.ones(3, dtype=np.float32),
        intensity=light_intensity,
        innerConeAngle=np.pi / 16.0,
    )
    scene.add(light, pose=cam_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(*res, **kwargs)

    device = batched_cloud.device


    # Images
    images = torch.zeros((len(batched_cloud), *res, 3), dtype=batched_cloud.dtype).to(device)

    for idx in range(len(batched_cloud)):

        tmesh = trimesh.points.PointCloud(batched_cloud[i])

        if isinstance(tmesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(tmesh)
        elif isinstance(tmesh, trimesh.PointCloud):
            if tmesh.colors is not None:
                colors = np.array(tmesh.colors)
            else:
                colors = np.ones_like(tmesh.vertices)
            mesh = pyrender.Mesh.from_points(
                np.array(tmesh.vertices), colors=colors
            )
        node = scene.add(mesh)
        color_img, depth_img = r.render(scene)
        images[idx] = torch.tensor(color_img, dtype=batched_cloud.dtype).to(device)
        scene.remove_node(node)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    return images



def list2npy(array):
    return array if isinstance(array, np.ndarray) else np.array(array)


def look_at(eye, center, world_up):
    """Computes camera viewing matrices (numpy implementation).

    Args:
    eye: np.float32 array with shape [batch_size, 3] containing the XYZ world
      space position of the camera.
    center: np.float32 array with shape [batch_size, 3] containing a position
      along the center of the camera's gaze.
    world_up: np.float32 array with shape [batch_size, 3] specifying the
      world's up direction; the output camera will have no tilt with respect to
      this direction.

    Returns:
    A [batch_size, 4, 4] np.float32 array containing a right-handed camera
    extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]
    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
    assert np.all(forward_norm > vector_degeneracy_cutoff)
    forward /= forward_norm

    to_side = np.cross(forward, world_up)
    to_side_norm = np.linalg.norm(to_side, axis=1, keepdims=True)
    assert np.all(to_side_norm > vector_degeneracy_cutoff)
    to_side /= to_side_norm
    cam_up = np.cross(to_side, forward)

    w_column = np.array(
        batch_size * [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )  # [batch_size, 4]
    w_column = w_column.reshape([batch_size, 4, 1])
    view_rotation = np.stack(
        [to_side, cam_up, -forward, np.zeros_like(to_side, dtype=np.float32)],
        axis=1,
    )  # [batch_size, 4, 3] matrix
    view_rotation = np.concatenate(
        [view_rotation, w_column], axis=2
    )  # [batch_size, 4, 4]

    identity_batch = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
    view_translation = np.concatenate(
        [identity_batch, np.expand_dims(-eye, 2)], 2
    )
    view_translation = np.concatenate(
        [view_translation, w_column.reshape([batch_size, 1, 4])], 1
    )
    camera_matrices = np.matmul(view_rotation, view_translation)
    return camera_matrices
