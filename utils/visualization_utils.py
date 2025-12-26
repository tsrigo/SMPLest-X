import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import pyrender
import trimesh

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1, radius=3, color=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    if color is None:
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        if color is None:
            cv2.circle(kp_mask, p, radius=radius, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(kp_mask, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '\n') 
    obj_file.close()

def perspective_projection(vertices, cam_param):
    # vertices: [N, 3]
    # cam_param: [3]
    fx, fy= cam_param['focal']
    cx, cy = cam_param['princpt']
    vertices[:, 0] = vertices[:, 0] * fx / vertices[:, 2] + cx
    vertices[:, 1] = vertices[:, 1] * fy / vertices[:, 2] + cy
    return vertices

def draw_skeleton(img, joints_2d, alpha=0.7):
    """
    Draw skeleton on image based on SMPLX body and hand joints.
    
    Args:
        img: Input image (H, W, 3)
        joints_2d: 2D joint positions (N, 2) - first 25 are body joints, 25-44 left hand, 45-64 right hand
        alpha: Transparency for blending
    
    Returns:
        Image with skeleton drawn
    """
    # SMPLX body skeleton connections (using first 25 body joints)
    # Joint names: Pelvis(0), L_Hip(1), R_Hip(2), L_Knee(3), R_Knee(4), 
    #              L_Ankle(5), R_Ankle(6), Neck(7), L_Shoulder(8), R_Shoulder(9),
    #              L_Elbow(10), R_Elbow(11), L_Wrist(12), R_Wrist(13),
    #              L_Big_toe(14), L_Small_toe(15), L_Heel(16), R_Big_toe(17), R_Small_toe(18), R_Heel(19),
    #              L_Ear(20), R_Ear(21), L_Eye(22), R_Eye(23), Nose(24)
    # 
    # Left hand joints (25-44):
    #   L_Thumb_1(25), L_Thumb_2(26), L_Thumb_3(27), L_Thumb_4(28),
    #   L_Index_1(29), L_Index_2(30), L_Index_3(31), L_Index_4(32),
    #   L_Middle_1(33), L_Middle_2(34), L_Middle_3(35), L_Middle_4(36),
    #   L_Ring_1(37), L_Ring_2(38), L_Ring_3(39), L_Ring_4(40),
    #   L_Pinky_1(41), L_Pinky_2(42), L_Pinky_3(43), L_Pinky_4(44)
    #
    # Right hand joints (45-64):
    #   R_Thumb_1(45), R_Thumb_2(46), R_Thumb_3(47), R_Thumb_4(48),
    #   R_Index_1(49), R_Index_2(50), R_Index_3(51), R_Index_4(52),
    #   R_Middle_1(53), R_Middle_2(54), R_Middle_3(55), R_Middle_4(56),
    #   R_Ring_1(57), R_Ring_2(58), R_Ring_3(59), R_Ring_4(60),
    #   R_Pinky_1(61), R_Pinky_2(62), R_Pinky_3(63), R_Pinky_4(64)
    
    body_skeleton_lines = [
        # Torso
        (0, 7),   # Pelvis -> Neck
        # Left leg
        (0, 1), (1, 3), (3, 5), (5, 16), (5, 14), (5, 15),  # Pelvis -> L_Hip -> L_Knee -> L_Ankle -> foot
        # Right leg
        (0, 2), (2, 4), (4, 6), (6, 19), (6, 17), (6, 18),  # Pelvis -> R_Hip -> R_Knee -> R_Ankle -> foot
        # Left arm
        (7, 8), (8, 10), (10, 12),  # Neck -> L_Shoulder -> L_Elbow -> L_Wrist
        # Right arm
        (7, 9), (9, 11), (11, 13),  # Neck -> R_Shoulder -> R_Elbow -> R_Wrist
        # Head
        (7, 24), (24, 22), (24, 23), (22, 20), (23, 21),  # Neck -> Nose -> Eyes -> Ears
    ]
    
    # Left hand skeleton (from wrist 12 to finger tips)
    left_hand_skeleton = [
        # Thumb
        (12, 25), (25, 26), (26, 27), (27, 28),  # Wrist -> Thumb_1 -> Thumb_2 -> Thumb_3 -> Thumb_4
        # Index
        (12, 29), (29, 30), (30, 31), (31, 32),  # Wrist -> Index_1 -> Index_2 -> Index_3 -> Index_4
        # Middle
        (12, 33), (33, 34), (34, 35), (35, 36),  # Wrist -> Middle_1 -> Middle_2 -> Middle_3 -> Middle_4
        # Ring
        (12, 37), (37, 38), (38, 39), (39, 40),  # Wrist -> Ring_1 -> Ring_2 -> Ring_3 -> Ring_4
        # Pinky
        (12, 41), (41, 42), (42, 43), (43, 44),  # Wrist -> Pinky_1 -> Pinky_2 -> Pinky_3 -> Pinky_4
    ]
    
    # Right hand skeleton (from wrist 13 to finger tips)
    right_hand_skeleton = [
        # Thumb
        (13, 45), (45, 46), (46, 47), (47, 48),  # Wrist -> Thumb_1 -> Thumb_2 -> Thumb_3 -> Thumb_4
        # Index
        (13, 49), (49, 50), (50, 51), (51, 52),  # Wrist -> Index_1 -> Index_2 -> Index_3 -> Index_4
        # Middle
        (13, 53), (53, 54), (54, 55), (55, 56),  # Wrist -> Middle_1 -> Middle_2 -> Middle_3 -> Middle_4
        # Ring
        (13, 57), (57, 58), (58, 59), (59, 60),  # Wrist -> Ring_1 -> Ring_2 -> Ring_3 -> Ring_4
        # Pinky
        (13, 61), (61, 62), (62, 63), (63, 64),  # Wrist -> Pinky_1 -> Pinky_2 -> Pinky_3 -> Pinky_4
    ]
    
    # Combine all skeleton lines
    skeleton_lines = body_skeleton_lines + left_hand_skeleton + right_hand_skeleton
    
    # Color palette (BGR format for OpenCV)
    cmap = plt.get_cmap('rainbow')
    body_colors = [cmap(i) for i in np.linspace(0, 1, len(body_skeleton_lines) + 2)]
    body_colors = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in body_colors]
    
    # Use distinct colors for hands
    left_hand_color = (255, 100, 100)   # Light blue for left hand
    right_hand_color = (100, 100, 255)  # Light red for right hand
    
    # Create a copy for blending
    skeleton_img = np.copy(img)
    
    # Draw body skeleton lines
    for idx, (i1, i2) in enumerate(body_skeleton_lines):
        if i1 < len(joints_2d) and i2 < len(joints_2d):
            pt1 = (int(joints_2d[i1, 0]), int(joints_2d[i1, 1]))
            pt2 = (int(joints_2d[i2, 0]), int(joints_2d[i2, 1]))
            # Check if points are within image bounds
            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(skeleton_img, pt1, pt2, body_colors[idx], thickness=3, lineType=cv2.LINE_AA)
    
    # Draw left hand skeleton lines
    for i1, i2 in left_hand_skeleton:
        if i1 < len(joints_2d) and i2 < len(joints_2d):
            pt1 = (int(joints_2d[i1, 0]), int(joints_2d[i1, 1]))
            pt2 = (int(joints_2d[i2, 0]), int(joints_2d[i2, 1]))
            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(skeleton_img, pt1, pt2, left_hand_color, thickness=2, lineType=cv2.LINE_AA)
    
    # Draw right hand skeleton lines
    for i1, i2 in right_hand_skeleton:
        if i1 < len(joints_2d) and i2 < len(joints_2d):
            pt1 = (int(joints_2d[i1, 0]), int(joints_2d[i1, 1]))
            pt2 = (int(joints_2d[i2, 0]), int(joints_2d[i2, 1]))
            if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                cv2.line(skeleton_img, pt1, pt2, right_hand_color, thickness=2, lineType=cv2.LINE_AA)
    
    # Draw body joint points
    for i in range(min(25, len(joints_2d))):  # Only body joints
        pt = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(skeleton_img, pt, 5, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(skeleton_img, pt, 5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    # Draw left hand joint points
    for i in range(25, min(45, len(joints_2d))):
        pt = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(skeleton_img, pt, 3, left_hand_color, thickness=-1, lineType=cv2.LINE_AA)
    
    # Draw right hand joint points
    for i in range(45, min(65, len(joints_2d))):
        pt = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(skeleton_img, pt, 3, right_hand_color, thickness=-1, lineType=cv2.LINE_AA)
    
    # Blend with original image
    return cv2.addWeighted(img, 1.0 - alpha, skeleton_img, alpha, 0)


def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False, draw_skeleton_flag=False, joints_2d=None):
    if mesh_as_vertices:
        # to run on cluster where headless pyrender is not supported for A100/V100
        vertices_2d = perspective_projection(vertices, cam_param)
        img = vis_keypoints(img, vertices_2d, alpha=0.8, radius=2, color=(0, 0, 255))
    else:
        focal, princpt = cam_param['focal'], cam_param['princpt']
        camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
        # the inverse is same
        pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1]])
        

        # render material
        base_color = (1.0, 193/255, 193/255, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0,
                alphaMode='OPAQUE',
                baseColorFactor=base_color)
        
        material_new = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.1,
                roughnessFactor=0.4,
                alphaMode='OPAQUE',
                emissiveFactor=(0.2, 0.2, 0.2),
                baseColorFactor=(0.7, 0.7, 0.7, 1))  
        material = material_new
        
        # get body mesh
        body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
        body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

        # prepare camera and light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        cam_pose = pyrender2opencv @ np.eye(4)
        
        # build scene
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                        ambient_light=(0.3, 0.3, 0.3))
        scene.add(camera, pose=cam_pose)
        scene.add(light, pose=cam_pose)
        scene.add(body_mesh, 'mesh')

        # render scene
        r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                        viewport_height=img.shape[0],
                                        point_size=1.0)
        
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        alpha = 0.8 # set transparency in [0.0, 1.0]

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        valid_mask = valid_mask * alpha
        img = img / 255
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

        img = (output_img * 255).astype(np.uint8)
    
    # Draw skeleton if requested and joints are provided
    if draw_skeleton_flag and joints_2d is not None:
        img = draw_skeleton(img, joints_2d, alpha=0.7)
    
    return img