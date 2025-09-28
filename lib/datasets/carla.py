import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import cv2
from PIL import Image
import copy
import matplotlib.pyplot as plt
import logging
from skimage import feature
from skimage.morphology import disk, dilation

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
from lib.helpers.file_io import read_numpy
from lib.helpers.unidrive import run_project_pixels_cuda
import pdb

class CARLA(data.Dataset):
    def __init__(self, root_dir, split, cfg, ext_config= None):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([512, 512]) if 'resolution' not in cfg.keys() else np.array(cfg['resolution']) # W * H
        self.eval_dataset  = cfg['type']        if 'eval_dataset' not in cfg.keys() else cfg['eval_dataset']
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck', 'Bus', 'Train'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        ##w,h,l
        # Ped  [1.7431 0.8494 0.911 ]
        # Car  [1.8032 2.1036 4.8104]
        # Cyc  [1.7336 0.823  1.753 ]
        self.cls_mean_size = np.array([[1.7431, 0.8494, 0.9110],
                                       [1.8032, 2.1036, 4.8104],
                                       [1.7336, 0.8230, 1.7530]
                                       ])
                              
        # data split loading
        assert split in ['train', 'val', 'train_small', 'val_small', 'train_5k', 'train_10k', 'trainval', 'test'] or 'train' in split
        self.split = split

        if ext_config is None:
            self.ext_config = 'pitch0' if 'ext_config'     not in cfg.keys() else cfg['ext_config']
        else:
            self.ext_config = ext_config

        # path configuration
        if self.eval_dataset == 'kitti':
            split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
            self.data_dir = os.path.join(root_dir, 'KITTI', 'testing' if split == 'test' else 'training')
            self.image_dir = os.path.join(self.data_dir, 'image_2')
            self.depth_dir = os.path.join(self.data_dir, 'depth')
            self.calib_dir = os.path.join(self.data_dir, 'calib')
            self.label_dir = os.path.join(self.data_dir, 'label_2')
        else:
            split_dir = os.path.join(root_dir, self.eval_dataset, 'ImageSets', split + '.txt')
            self.data_dir = os.path.join(root_dir, self.eval_dataset, self.ext_config, 'validation' if 'val' in split else 'training')
            self.image_dir = os.path.join(self.data_dir, 'image')
            self.calib_dir = os.path.join(self.data_dir, 'calib')
            self.label_dir = os.path.join(self.data_dir, 'label')
            self.seman_dir = os.path.join(self.data_dir, 'seman')
            self.depth_dir = os.path.join(self.data_dir, 'depth')

        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # data augmentation configuration
        self.data_augmentation = True if 'train' in split else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4
        self.transform_image_method = None if 'transform_image_method' not in cfg else cfg['transform_image_method']
        self.transform_image   = False    if self.transform_image_method is None else True
        self.normal_as_image   = False    if 'normal_as_image'    not in cfg else cfg['normal_as_image']
        self.load_depth_normal = False    if 'load_depth_normal'  not in cfg else cfg['load_depth_normal']
        self.normal_name       = "oracle" if 'normal_name'        not in cfg else cfg['normal_name']
        if self.load_depth_normal:
            logging.info("Using {} normal".format(self.normal_name))
        if self.normal_name in ["oracle", "seman"]:
            self.normal_dir = os.path.join(self.data_dir, 'normal')
        elif self.normal_name == "dsine":
            self.normal_dir = os.path.join(self.data_dir, 'dsine')
        elif self.normal_name == "metnor":
            self.normal_dir = os.path.join(self.data_dir, 'metnor')
        else:
            raise NotImplementedError

        self.load_canny        = False    if 'load_canny'         not in cfg else cfg['load_canny']
        self.canny_dilation    = 0        if 'canny_dilation'     not in cfg else cfg['canny_dilation']

        if self.split in ["val", "test"] and self.transform_image:
            self.ht_change   = 0 if self.ext_config == "pitch0" else -float(self.ext_config.replace("height",""))*0.0254
            self.f = 405.24
            self.depth_plane = 3.14 if 'depth_plane' not in cfg.keys() else cfg['depth_plane']
            logging.info("Transforming {} images by {} with ht_change= {:.2f}m f= {:.2f} depth_plane= {:.2f}m ".format(self.split, self.transform_image_method, self.ht_change, self.f, self.depth_plane))

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.jpg' % idx)
        assert os.path.exists(img_file)
        # print("=> Read ", img_file)
        img        = cv2.imread  (img_file, cv2.IMREAD_COLOR)
        img_cv_rgb = cv2.cvtColor(img     , cv2.COLOR_BGR2RGB)
        img_pil    = Image.fromarray(img_cv_rgb)
        return  img_pil  # (H, W, 3) RGB mode

    def get_seman(self, idx):
        seman_file = os.path.join(self.seman_dir, '%06d.npy' % idx)
        seman      = read_numpy(seman_file, show_message= False)
        seman      = Image.fromarray(seman)
        return seman

    def get_depth(self, idx):
        depth_file = os.path.join(self.depth_dir, '%06d.npy' % idx)
        depth      = read_numpy(depth_file, show_message= False)
        depth      = Image.fromarray(depth)
        return depth

    def get_normal(self, idx):
        normal_file = os.path.join(self.normal_dir, '%06d.png' % idx)
        normal      = cv2.imread  (normal_file, cv2.IMREAD_COLOR)
        normal_pil  = Image.fromarray(normal)
        return normal_pil

    def seman_to_normal(self, seman):
        seman = np.array(seman)
        h, w  = seman.shape
        normal       = np.zeros((h, w, 3))
        normal[:, :] = np.array([0.5, 0.5, 1.0])
        # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
        # Horizontal objects such as roads, sidewalks, terrain, water, roadline, ground
        mask  = np.logical_or.reduce([seman == 1, seman == 2, seman == 10, seman == 23, seman == 24, seman == 25])
        normal[mask] = np.array([0.5, 1.0, 0.5])
        normal = normal * 255.0
        normal = normal.astype(np.uint8)
        normal = Image.fromarray(normal)
        return normal

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)


    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        calib = self.get_calib(index)
        # image loading
        if self.normal_as_image:
            img = self.get_normal(index)
        else:
            img = self.get_image(index)
        seman = self.get_seman(index)
        if self.load_depth_normal:
            #depth_map = self.get_depth(index)
            if self.normal_name == "seman":
                normal    = self.seman_to_normal(seman)
            else:
                normal    = self.get_normal(index)
        img_size = np.array(img.size)

        if self.split in ["val", "test"] and self.transform_image:
            raw2can = np.array([[1, 0, 0], [0, 1, self.f * self.ht_change / self.depth_plane], [0, 0, 1]])
            if self.transform_image_method == "cv2":
                img2 = cv2.warpPerspective(np.array(img), raw2can, img_size, flags=cv2.INTER_LINEAR)
            elif self.transform_image_method == "unidrive":
                ONES = np.array([[0.,0.,0.,1.]])
                intrinsics = calib.intrinsics
                raw2can_extrinsics = copy.deepcopy(calib.gd_to_cam)
                raw2can_extrinsics[1, 3] -= 1.510957639130
                raw2can_extrinsics = np.concatenate((raw2can_extrinsics, ONES), axis= 0)
                raw2can_extrinsics[1, 3] *= -1
                img2 = run_project_pixels_cuda(np.array(img), intrinsics, raw2can_extrinsics, np.eye(4), Dis= self.depth_plane)
            else:
                raise NotImplementedError
            img = Image.fromarray(img2)
            calib.gd_to_cam[1, 3] = 1.510957639130
            if self.load_depth_normal:
                normal = Image.fromarray(cv2.warpPerspective(np.array(normal), raw2can, img_size, flags=cv2.INTER_LINEAR))
        else:
            raw2can = np.eye(3)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                seman = seman.transpose(Image.FLIP_LEFT_RIGHT)
                if self.load_depth_normal:
                    # depth_map = depth_map.transpose(Image.FLIP_LEFT_RIGHT)
                    normal    = normal.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        seman = seman.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if self.load_depth_normal:
            # depth_map = depth_map.transform(tuple(self.resolution.tolist()),
            #                         method=Image.AFFINE,
            #                         data=tuple(trans_inv.reshape(-1).tolist()),
            #                         resample=Image.BILINEAR)
            normal    = normal.transform(tuple(self.resolution.tolist()),
                                    method=Image.AFFINE,
                                    data=tuple(trans_inv.reshape(-1).tolist()),
                                    resample=Image.BILINEAR)
        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)                   
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W
        seman = np.array(seman).astype(np.float32)
        if self.load_depth_normal:
            # depth_map = np.array(depth_map).astype(np.float32) # H * W
            # depth_map = depth_map[np.newaxis, :, :]            # 1 * H * W
            normal    = np.array(normal).astype(np.float32) / 255.0
            normal    = normal.transpose(2, 0, 1)              # C * H * W
            # normal in range (-1, 1)
            normal    = 2 * normal - 1.0

        features_size = self.resolution // self.downsample# W * H
        #  ============================   get labels   ==============================
        if True:#self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            centers_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue
    
                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue
    
                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()

                # filter boxes with zero or negative 2D height or width
                temp_w, temp_h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                if temp_w <= 0 or temp_h <= 0:
                    continue

                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample
    
                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample      
            
                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare', 'Sign']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
                centers_2d[i] = center_2d
    
                # encoding depth
                depth[i] = objects[i].pos[-1]
    
                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)
    
                # encoding 3d offset & size_3d
                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:    
                    mask_2d[i] = 1           
            targets = {'depth': depth,
                   'center_2d': centers_2d,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'cls_ids': cls_ids,
                   'seman': seman,
                   'mask_2d': mask_2d}
            if self.load_depth_normal:
                # targets['depth_map'] = depth_map
                targets['normal']    = normal
            if self.load_canny:
                # calculate canny edges on images
                edge = feature.canny(np.mean(img, axis=0), sigma=2)
                if self.canny_dilation > 0:
                    edge = dilation(edge, disk(self.canny_dilation))
                targets['canny']     = edge[np.newaxis, :, :]
        # else:
        #     targets = {'seman': seman}
        #     if self.load_depth_normal:
        #         targets['depth_map'] = depth_map
        #         targets['normal']    = normal
        # collect return data
        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'raw2can': raw2can,
                'gd_to_cam': calib.gd_to_cam,
                'intrinsics': calib.intrinsics,
                'trans': trans,
                'bbox_downsample_ratio': img_size/features_size}   
        return inputs, calib.P2, coord_range, targets, info   #calib.P2
