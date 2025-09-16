#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import open3d as o3d

# ==========================
# User config (核心参数)
# ==========================
ROOT_DIR       = "/home/ani/Downloads/Grasp1B"
SCENE_ID       = "scene_0090"
CAMERA_NAME    = "kinect"        # "kinect" or "realsense"
QUAT_ORDER     = "wxyz"          # "wxyz" or "xyzw"
OFFSETS_MODE   = "relative"      # "absolute" or "relative"
AUTO_DETECT_SCALE = True

# TSDF & rendering knobs
DEPTH_TRUNC_M  = 1.0
VOXEL_LEN_M    = 0.004
SDF_TRUNC_M    = 0.02
DRAW_CAM_EVERY = 20
WORLD_AX_SIZE  = 0.1
CAM_AX_SIZE    = 0.05

# Grasp sampling & rendering knobs
A_BINS             = 12
B_BINS             = 4
SCORE_THRESH       = -0.5
STEP_I_POINTS_TGT  = 60
STEP_K_CANDS_TGT   = 10
MAX_GRASPS_PER_OBJ = 1
JAW_WIDTH_MIN      = 0.02
JAW_WIDTH_MAX      = 0.08
FINGER_LEN         = 0.04
FINGER_THICK       = 0.004
GRIPPER_COLOR      = (0.9, 0.3, 0.1)

# ==========================
# Camera utilities
# ==========================
def load_intrinsics(camK_path, img_width, img_height):
    K = np.load(camK_path)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(img_width, img_height, fx, fy, cx, cy)
    return intr

def read_rgbd(rgb_path, depth_path, depth_scale=1000.0, depth_trunc=DEPTH_TRUNC_M):
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

def create_camera_frame(pose, size=CAM_AX_SIZE):
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cam_frame.transform(pose)
    return cam_frame

# ==========================
# Object pose parsing
# ==========================
def quat_to_rotmat(w, x, y, z):
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    R = np.array([
        [1.0 - (yy + zz), xy - wz,        xz + wy       ],
        [xy + wz,         1.0 - (xx + zz), yz - wx       ],
        [xz - wy,         yz + wx,        1.0 - (xx + yy)]
    ], dtype=np.float32)
    return R

def parse_object_poses_from_xml(xml_path, quat_order=QUAT_ORDER):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    poses = {}
    for e in root.findall("./obj"):
        oid_txt = e.findtext("obj_id")
        pos_txt = e.findtext("pos_in_world")
        ori_txt = e.findtext("ori_in_world")
        if oid_txt is None or pos_txt is None or ori_txt is None:
            continue
        obj_id = int(oid_txt.strip())
        t = np.fromstring(pos_txt.strip(), sep=' ').astype(np.float32)
        q = np.fromstring(ori_txt.strip(), sep=' ').astype(np.float32)
        if t.size != 3 or q.size != 4: 
            continue
        qo = quat_order.lower()
        if qo == 'wxyz':
            w, x, y, z = q
        elif qo == 'xyzw':
            x, y, z, w = q
        else:
            raise ValueError("quat_order must be 'wxyz' or 'xyzw'.")
        R = quat_to_rotmat(float(w), float(x), float(y), float(z))
        poses[obj_id] = (R, t)
    return poses

# ==========================
# Basis / approach utils (无调试)
# ==========================
def basis_from_z_and_theta(z_vec, theta):
    z_norm = np.linalg.norm(z_vec)
    z = np.array([0, 0, 1], dtype=np.float32) if z_norm < 1e-8 else z_vec / z_norm
    aux = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(np.dot([1,0,0], z)) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x0 = aux - np.dot(aux, z) * z
    x0 = x0 if np.linalg.norm(x0) < 1e-8 else x0 / np.linalg.norm(x0)
    if np.linalg.norm(x0) < 1e-8: x0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y0 = np.cross(z, x0)
    y0 = y0 if np.linalg.norm(y0) < 1e-8 else y0 / np.linalg.norm(y0)
    if np.linalg.norm(y0) < 1e-8: y0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    c, s = np.cos(theta), np.sin(theta)
    x = c * x0 + s * y0
    y = np.cross(z, x)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = y / (np.linalg.norm(y) + 1e-8)
    return x, y, z

def fix_approach_direction(approach_vec, surface_point, grasp_center):
    v = grasp_center - surface_point
    if np.linalg.norm(v) < 1e-8:
        return approach_vec
    a = approach_vec if np.linalg.norm(approach_vec) >= 1e-8 else v
    if np.dot(a, v) < 0:
        a = -a
    n = np.linalg.norm(a)
    if n < 0.01:
        a = a / (n + 1e-8) * 0.03
    return a

# ==========================
# 简洁版夹爪可视化（实体网格）
# ==========================
def create_mesh_gripper(center, x, y, z, jaw_width, finger_len, finger_thick, color):
    meshes = []
    w = jaw_width * 0.5
    t = finger_thick * 0.5

    def make_box(wd, ht, dp, trans, Rcols):
        m = o3d.geometry.TriangleMesh.create_box(width=wd, height=ht, depth=dp)
        m.translate([-wd/2, 0, -dp/2])
        T = np.eye(4)
        T[:3, :3] = np.column_stack(Rcols)
        T[:3, 3] = trans
        m.transform(T)
        return m

    # 左/右手指
    left_pos  = center + ( w - t/2) * x
    right_pos = center + (-w + t/2) * x
    left  = make_box(finger_thick, finger_len, finger_thick, left_pos,  [x, y, z])
    right = make_box(finger_thick, finger_len, finger_thick, right_pos, [x, y, z])
    left.paint_uniform_color(color); right.paint_uniform_color(color)
    meshes += [left, right]

    # 底座（连接杆）
    base_width = jaw_width + finger_thick
    base = o3d.geometry.TriangleMesh.create_box(width=base_width, height=finger_thick*0.5, depth=finger_thick)
    base.translate([-base_width/2, 0, -t/2])
    Tb = np.eye(4); Tb[:3, :3] = np.column_stack([x, y, z]); Tb[:3, 3] = center
    base.transform(Tb)
    base.paint_uniform_color(tuple(0.7*np.array(color)))
    meshes.append(base)
    return meshes

def create_parallel_gripper(center, z_axis, theta_rad, jaw_width,
                            finger_len=FINGER_LEN, finger_thick=FINGER_THICK,
                            color=GRIPPER_COLOR):
    x, y, z = basis_from_z_and_theta(z_axis, theta_rad)
    return create_mesh_gripper(center, x, y, z, jaw_width, finger_len, finger_thick, color)

# ==========================
# 尺度检测（无打印）
# ==========================
def detect_and_fix_scale_issues(points, offsets, obj_pose_world, scene_aabb):
    points_magnitude = np.max(np.abs(points)) if points.size else 0.0
    offsets_flat = offsets.reshape(-1, 3) if offsets.size else np.zeros((0,3))
    offsets_magnitude = np.max(np.abs(offsets_flat)) if offsets_flat.size else 0.0
    scene_size = scene_aabb.max_bound - scene_aabb.min_bound
    scene_magnitude = float(np.max(scene_size)) if scene_size.size else 1.0
    scale_factor = 1.0
    if points_magnitude > 10.0 and scene_magnitude < 5.0:
        scale_factor = 1000.0
    elif offsets_magnitude > 1.0 and scene_magnitude < 5.0:
        scale_factor = 1000.0
    elif points_magnitude < 0.01 and scene_magnitude > 0.1:
        scale_factor = 1/1000.0
    return scale_factor

def apply_scale_correction(points, offsets, scale_factor):
    if abs(scale_factor - 1.0) > 1e-6:
        return points / scale_factor, offsets / scale_factor
    return points, offsets

# ==========================
# 夹爪标签加载（简化版）
# ==========================
def load_grasp_labels_world(
    grasp_label_dir, obj_id, obj_pose_world,
    max_grasps=MAX_GRASPS_PER_OBJ, score_thresh=SCORE_THRESH,
    a_bins=A_BINS, b_bins=B_BINS,
    width_min=JAW_WIDTH_MIN, width_max=JAW_WIDTH_MAX,
    offsets_mode=OFFSETS_MODE,
    scene_aabb=None
):
    path = os.path.join(grasp_label_dir, f"{obj_id:03d}_labels.npz")
    if not os.path.exists(path):
        return [], []

    label = np.load(path, allow_pickle=True)
    points = label["points"].astype(np.float32)      # (N, 3) in object frame
    offsets = label["offsets"].astype(np.float32)    # (N, K, A, B, 3)
    scores = label["scores"]                         # (N, K, A, B)
    collision = label["collision"]                   # (N, K, A, B)

    R_obj, t_obj = obj_pose_world
    N, K, A, B = scores.shape

    if AUTO_DETECT_SCALE and scene_aabb is not None:
        scale_factor = detect_and_fix_scale_issues(points, offsets, obj_pose_world, scene_aabb)
        points, offsets = apply_scale_correction(points, offsets, scale_factor)

    step_i = max(1, N // max(1, STEP_I_POINTS_TGT))
    step_k = max(1, K // max(1, STEP_K_CANDS_TGT))

    gripper_geoms_flat = []
    surface_points_world = []
    total_count = 0

    for i in range(0, N, step_i):
        for k in range(0, K, step_k):
            s_grid = scores[i, k]
            c_grid = collision[i, k]
            valid = (s_grid >= score_thresh) & (~c_grid)
            if not np.any(valid):
                continue
            s_pick = np.where(valid, s_grid, -np.inf)
            a_idx, b_idx = np.unravel_index(np.argmax(s_pick), s_grid.shape)

            off = offsets[i, k, a_idx, b_idx]
            surface_point_obj = points[i]

            if offsets_mode.lower() == "absolute":
                center_o = off
                approach_o = center_o - surface_point_obj
            elif offsets_mode.lower() == "relative":
                center_o = surface_point_obj + off
                approach_o = off
            else:
                raise ValueError("OFFSETS_MODE must be 'absolute' or 'relative'.")

            surface_point_w = (R_obj @ surface_point_obj) + t_obj
            center_w = (R_obj @ center_o) + t_obj
            approach_w = (R_obj @ approach_o)
            approach_w = fix_approach_direction(approach_w, surface_point_w, center_w)

            theta = 2.0 * np.pi * (a_idx / float(a_bins))
            jaw_width = width_min + (width_max - width_max * 0 + width_max * 0 - width_min) * 0  # placeholder to appease lints
            jaw_width = width_min + (width_max - width_min) * (b_idx / float(max(1, b_bins - 1)))

            gripper_geoms = create_parallel_gripper(center_w, approach_w, theta, jaw_width)
            if isinstance(gripper_geoms, list):
                gripper_geoms_flat.extend(gripper_geoms)
            else:
                gripper_geoms_flat.append(gripper_geoms)

            surface_points_world.append(surface_point_w)

            total_count += 1
            if total_count >= max_grasps:
                break
        if total_count >= max_grasps:
            break

    return gripper_geoms_flat, surface_points_world

# ==========================
# Scene reconstruction
# ==========================
def reconstruct_scene(scene_path, camera_name=CAMERA_NAME, vis_cams=True):
    """Reconstruct a scene using TSDF fusion."""
    cam_path = os.path.join(scene_path, camera_name)
    rgb_dir = os.path.join(cam_path, "rgb")
    depth_dir = os.path.join(cam_path, "depth")
    rgb_files = sorted(os.listdir(rgb_dir))
    depth_files = sorted(os.listdir(depth_dir))
    assert len(rgb_files) and len(depth_files), "No RGB/Depth files found."

    rgb0 = cv2.imread(os.path.join(rgb_dir, rgb_files[0]))
    h, w, _ = rgb0.shape
    intrinsics = load_intrinsics(os.path.join(cam_path, "camK.npy"), w, h)
    cam_poses = np.load(os.path.join(cam_path, "camera_poses.npy"))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LEN_M, sdf_trunc=SDF_TRUNC_M,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    cam_frames = []
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, depth_file)
        rgbd = read_rgbd(rgb_path, depth_path, depth_scale=1000.0, depth_trunc=DEPTH_TRUNC_M)

        T_i_wrt_0 = cam_poses[i]
        volume.integrate(rgbd, intrinsics, np.linalg.inv(T_i_wrt_0))

        if vis_cams and i % DRAW_CAM_EVERY == 0:
            cam_frames.append(create_camera_frame(T_i_wrt_0, size=CAM_AX_SIZE))
        if i % 50 == 0:
            print(f"[TSDF] Processed {i}/{len(rgb_files)} frames...")

    pcd = volume.extract_point_cloud()
    return pcd, cam_frames

# ==========================
# 可视化辅助
# ==========================
def create_surface_point_cloud(surface_points_world, color=[1.0, 0.0, 0.0]):
    if not surface_points_world:
        return None
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(np.array(surface_points_world))
    surface_pcd.paint_uniform_color(color)
    return surface_pcd

def create_object_coordinate_frames(obj_poses, size=0.05):
    frames = []
    for _, (R, t) in obj_poses.items():
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        frame.transform(T)
        frames.append(frame)
    return frames

# ==========================
# 主函数（核心流程）
# ==========================
def main():
    scene_path = os.path.join(ROOT_DIR, "scenes", SCENE_ID)
    grasp_label_dir = os.path.join(ROOT_DIR, "grasp_label")
    cam_path = os.path.join(scene_path, CAMERA_NAME)

    # 1) 重建场景
    pcd, cam_frames = reconstruct_scene(scene_path, camera_name=CAMERA_NAME, vis_cams=True)

    # 2) 解析物体位姿（以第0帧为世界系）
    xml_0000 = os.path.join(cam_path, "annotations", "0000.xml")
    if not os.path.exists(xml_0000):
        raise FileNotFoundError(f"Missing XML: {xml_0000}")
    obj_poses = parse_object_poses_from_xml(xml_0000, quat_order=QUAT_ORDER)

    # 3) 读取场景中的物体列表
    obj_list_path = os.path.join(scene_path, "object_id_list.txt")
    with open(obj_list_path, "r") as f:
        obj_ids = [int(line.strip()) for line in f.readlines()]

    # 4) 物体坐标系（如不需要可注释掉）
    obj_frames = create_object_coordinate_frames(obj_poses, size=0.05)

    # 5) 加载与可视化夹爪（每个物体最多 MAX_GRASPS_PER_OBJ 个）
    aabb = pcd.get_axis_aligned_bounding_box()
    all_gripper_geoms = []
    all_surface_points = []
    for obj_id in obj_ids:
        if obj_id not in obj_poses:
            continue
        gripper_geoms, surface_points = load_grasp_labels_world(
            grasp_label_dir, obj_id, obj_pose_world=obj_poses[obj_id],
            max_grasps=MAX_GRASPS_PER_OBJ, score_thresh=SCORE_THRESH,
            a_bins=A_BINS, b_bins=B_BINS,
            width_min=JAW_WIDTH_MIN, width_max=JAW_WIDTH_MAX,
            offsets_mode=OFFSETS_MODE,
            scene_aabb=aabb
        )
        all_gripper_geoms.extend(gripper_geoms)
        all_surface_points.extend(surface_points)

    # 6) 辅助可视化（表面点）
    auxiliary_geoms = []
    surface_pcd = create_surface_point_cloud(all_surface_points, color=[1.0, 0.0, 0.0])
    if surface_pcd is not None:
        auxiliary_geoms.append(surface_pcd)

    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=WORLD_AX_SIZE)
    # geoms = [pcd, world_frame] + cam_frames + obj_frames + all_gripper_geoms + auxiliary_geoms
    geoms = [pcd, world_frame] + cam_frames + all_gripper_geoms + auxiliary_geoms

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Grasp Visualization - {SCENE_ID}",
        width=1400,
        height=900
    )

if __name__ == "__main__":
    main()
