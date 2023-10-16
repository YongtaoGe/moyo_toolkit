import argparse
import glob
import json
import os
import os.path as osp

import cv2
import pdb
import numpy as np
import torch
import trimesh
from ezc3d import c3d as ezc3d
# import c3d

from tqdm import tqdm
import smplx

MOYO_V_TEMPLATE = '/home/geyongtao/code/moyo_toolkit/data/v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply'


def smplx_breakdown(bdata, device):
    num_betas=10
    num_frames = len(bdata['trans'])
    bdata['poses'] = bdata['fullpose']

    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][:, 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][:, 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][:, 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][:, 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][:, 120:]).float().to(device)
    betas = torch.from_numpy(bdata['betas'][:num_betas][None]).float().to(device)
    trans = torch.from_numpy(bdata['trans']).float().to(device)

    v_template = trimesh.load(MOYO_V_TEMPLATE, process=False)

    body_params = {'global_orient': global_orient, 'body_pose': body_pose,
                   'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                   'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   'v_template': torch.Tensor(v_template.vertices).to(device), 
                   'betas': betas,
                   'transl': trans}
    return body_params


def smplx_to_mesh(smplx_params, frame_num, body_model, model_type='smplx', gender='netrual'):
    with torch.no_grad():
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # betas = torch.from_numpy(body_params['betas']).float().to(device).unsqueeze(0)
        # betas = betas[None, :, :num_betas]  # only using 10 betas
        body_model_output = body_model(transl=smplx_params['transl'][[frame_num]],
                                       global_orient=smplx_params['global_orient'][[frame_num]],
                                       body_pose=smplx_params['body_pose'][[frame_num]],
                                       left_hand_pose=smplx_params['left_hand_pose'][[frame_num]],
                                       right_hand_pose=smplx_params['right_hand_pose'][[frame_num]],
                                       betas=smplx_params['betas']
                                       )

        # pelvis = body_model_output.joints[:, 0]
        # import ipdb; ipdb.set_trace()
        # mesh = visualize_mesh(body_model_output, body_model.faces)
        # mesh.export(out_ply)
    return body_model_output


def project2d(j3d, cam_params, downsample_factor=1.0):
    """
    Project 3D points to 2D
    Args:
        j3d: (N, 3) 3D joints
        cam_params: dict
        downsample_factor: resize factor

    Returns:
        j2d : 2D joint locations
    """
    mm2m = 1000
    j3d = torch.tensor(j3d, dtype=torch.float32)

    # cam intrinsics
    f = cam_params['focal'] * downsample_factor
    cx = cam_params['princpt'][0] * downsample_factor
    cy = cam_params['princpt'][1] * downsample_factor

    # cam extrinsics
    R = torch.tensor(cam_params['rotation'])
    t = -torch.mm(R, torch.tensor(cam_params['position'])[:, None]).squeeze()  # t= -RC

    # cam matrix
    K = torch.tensor([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]]).to(j3d.device)

    Rt = torch.cat([R, t[:, None]], dim=1).to(j3d.device)

    # apply extrinsics
    bs = j3d.shape[0]
    # j3d_transpose = torch.cat([j3d, torch.ones_like(j3d[..., [0]])], dim=2).permute(0, 2, 1)
    # j3d_cam = torch.bmm(Rt[None, :, :].expand(bs, -1, -1), j3d_transpose)
    j3d_cam = torch.bmm(Rt[None, :, :].expand(bs, -1, -1), j3d[:, :, None])

    # j2d = torch.bmm(K[None, :].expand(bs, -1, -1), j3d_cam)
    j2d = torch.bmm(K[None, :].expand(bs, -1, -1), j3d_cam)
    j2d = j2d / j2d[:, [-1]]
    return j2d[:, :-1, :].squeeze()


def visualize_on_img(j2d, img_name, out_dir):
    # Visualize the joints
    import pdb
    pdb.set_trace()
    pose_name = img_name.split('/')[-2]
    img_num = img_name.split('/')[-1]
    img = cv2.imread(img_name)
    fname = img_name.split('/')[-1]
    os.makedirs(out_dir, exist_ok=True)
    ext = osp.splitext(fname)[1]
    for n in range(j2d.shape[0]):
        # check if nan
        if np.any(np.isnan(j2d[n, :])):
            continue
        cor_x, cor_y = int(j2d[n, 0]), int(j2d[n, 1])
        cv2.circle(img, (cor_x, cor_y), 1, (0, 255, 0), 5)

    out_img_path = osp.join(out_dir, fname).replace(f'{ext}', '_markers.png')
    cv2.imwrite(out_img_path, img)
    print(f'{out_img_path} is saved')


if __name__=='__main__':
    model_folder = '/home/geyongtao/data/body_models/'
    device = 'cuda'
    downsample_factor = 0.5
    out_dir = '/home/geyongtao/code/moyo_toolkit'
    image_paths = '/home/geyongtao/data/moyo/images/train/220926_yogi_body_hands_03596_Rajakapotasana-a/YOGI_Cam_01'
    mosh_paths = '/home/geyongtao/data/moyo/mosh/train/220926_yogi_body_hands_03596_Rajakapotasana-a_stageii.pkl'
    c3d_paths = '/home/geyongtao/data/moyo/vicon/train/c3d/220926_yogi_body_hands_03596_Rajakapotasana-a.c3d'
    cam_path = '/home/geyongtao/data/moyo/cameras/20220926/cameras_param.json'
    model_type='smplx'
    num_betas=10
    v_template = trimesh.load(MOYO_V_TEMPLATE, process=False)
    v_template = torch.Tensor(v_template.vertices).to(device)

    body_model_params = dict(
                                model_path=model_folder,
                                model_type=model_type,
                                gender='neutral',
                                # v_template=smplx_params['v_template'],
                                # v_template=MOYO_V_TEMPLATE,
                                # joint_mapper=joint_mapper,
                                # batch_size=trans.shape[0],
                                batch_size=1,
                                create_global_orient=True,
                                create_body_pose=True,
                                create_betas=True,
                                num_betas=num_betas,
                                create_left_hand_pose=True,
                                create_right_hand_pose=True,
                                create_expression=True,
                                create_jaw_pose=True,
                                create_leye_pose=True,
                                create_reye_pose=True,
                                create_transl=True,
                                use_pca=False,
                                flat_hand_mean=True,
                                dtype=torch.float32
                                )
    body_model = smplx.create(**body_model_params).to(device)
    body_params = dict(np.load(mosh_paths, allow_pickle=True))
    body_params = smplx_breakdown(body_params, device)

    for img_path in os.listdir(image_paths):
        img_name = osp.basename(img_path)

        name_splits = img_name.split('_')
        frame_num = int(name_splits[-1][:-4])
        body_model_out = smplx_to_mesh(body_params, frame_num, body_model)
        j3d = body_model_out['joints']

        # get soma fit
        # try:
        #     c3d_path = glob.glob(osp.join(c3d_folder, f'{c3d_name}.c3d'))[0]
        # except:
        #     print(f'{c3d_folder}/{c3d_name}_stageii.pkl does not exist. SKIPPING!!!')
        #     continue

        c3d = ezc3d(c3d_paths)
        # (2760, 85, 4) numpy array
        markers3d = c3d['data']['points'].transpose(2, 1, 0)  # Frames x NumPoints x 4 (x,y,z,1) in homogenous coordinates
        # try:
        #     c3d_var = '_'.join(c3d_name.split('_')[5:])
        #     selected_frame = frame_select_dict[c3d_var]
        # except:
        #     print(f'{c3d_var} does not exist in frame_selection_dict. SKIPPING!!!')
        #     continue

        j3d = markers3d[frame_num]

        # Load camera matrix from json file
        with open(cam_path, 'rb') as fp:
            cameras = json.load(fp)

        cam_num = int(name_splits[name_splits.index('Cam') + 1])
        cam_id = f'cam_{cam_num}'

        cam_params = cameras[cam_id]
        j2d = project2d(j3d, cam_params, downsample_factor=downsample_factor)

        # convert to openpose format
        visualize_on_img(j2d.cpu().numpy(), os.path.join(image_paths, img_path), out_dir)