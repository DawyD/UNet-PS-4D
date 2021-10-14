"""
DataGenerator for CyclesPS Dataset
This file use substantial portion of code from the original CNN-PS repository https://github.com/satoshi-ikehata/CNN-PS/
"""

import numpy as np
import cv2
import os
import gc

from data.datagenerator import DataGenerator
from data.utils import rotate_images
from misc.projections import standard_proj

from tensorflow.keras.models import Model


class CyclesDataGenerator(DataGenerator):
    def __init__(self, datapath, objlist=None, batch_size=256,
                 spatial_patch_size=5, obs_map_size=32, shuffle=False, random_illums=False,
                 keep_axis=True, validation_split=None, nr_rotations=1, rotation_start=0, rotation_end=2 * np.pi,
                 projection=standard_proj, add_raw=False, images=None, normals=None, masks=None, illum_dirs=None,
                 order=2, divide_maps=False, round_nearest=True, rot_2D=False, verbose=False):

        self.datapath = datapath
        self.objlist = objlist if objlist is not None else sorted(os.listdir(datapath + '/PRPS'))
        self.verbose = verbose

        super(CyclesDataGenerator, self).__init__(
            batch_size=batch_size,
            spatial_patch_size=spatial_patch_size,
            obs_map_size=obs_map_size,
            shuffle=shuffle,
            random_illums=random_illums,
            keep_axis=keep_axis,
            validation_split=validation_split,
            nr_rotations=nr_rotations,
            rotation_start=rotation_start,
            rotation_end=rotation_end,
            projection=projection,
            add_raw=add_raw,
            images=images,
            normals=normals,
            masks=masks,
            illum_dirs=illum_dirs,
            order=order,
            divide_maps=divide_maps,
            round_nearest=round_nearest,
            rot_2D=rot_2D)

    def load_data(self):
        objid = 0
        for obj in self.objlist:
            for dirb, dirn, scale in zip(['PRPS_Diffuse/' + '%s' % obj, 'PRPS/' + '%s' % obj, 'PRPS/' + '%s' % obj],
                                         ['images_diffuse', 'images_specular', 'images_metallic'],
                                         [1, 0.5, 0.5]):
                if self.verbose:
                    print("\rPre-loading image ({:}/{:}) {:} ".format(objid + 1, self.nr_objects, dirb), end="")

                nr_ch = 3 if self.add_raw else 1
                sample_path = os.path.join(self.datapath, dirb, dirn)
                imgs, nmls, msks, light_dirs = self.load_sample(sample_path, scale, -1, nr_ch)

                self.fill_data(imgs, nmls, msks, light_dirs, objid)

                if self.verbose:
                    print("", end="\x1b[1K\r")

                objid += 1

        if self.verbose:
            print()

    def get_max_shape(self, rotations=None):
        """
        Returns a shape of an array (height, width, channels) which all images of various sizes under all rotations fit
        :param rotations: List of rotation angles (in radians)
        :return: max_shape [nr_objects, height, width, channels]
        """

        max_shape = [0, 0, 0, 0]
        for obj in self.objlist:
            for p, scale in zip(['PRPS_Diffuse/' + '%s' % obj,
                                 'PRPS/' + '%s' % obj,
                                 'PRPS/' + '%s' % obj], [1, 0.5, 0.5]):
                max_shape[0] += 1

                normal_path = os.path.join(self.datapath, p, 'gt_normal.tif')
                if not os.path.exists(normal_path):
                    raise ValueError("Path\"{:}\"does not exists.".format(normal_path))

                normals = cv2.imread(normal_path, -1)
                normals = cv2.resize(normals, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

                f = open(os.path.join(self.datapath, p, 'light.txt'))
                data = f.read()
                f.close()
                lines = data.split('\n')
                nr_illums = len(lines) - 1  # the last line is empty (how to fix it?)

                if nr_illums > max_shape[3]:
                    max_shape[3] = nr_illums

                if rotations is not None:
                    # In case of rotations, the width and height might be larger
                    for angle in rotations:
                        img_shape = rotate_images(2 * np.pi - angle, normals[..., 0], axes=(0, 1), order=0).shape
                        for k in range(2):
                            if img_shape[k] > max_shape[k+1]:
                                max_shape[k+1] = img_shape[k]
                else:
                    for k in range(2):
                        if normals.shape[k] > max_shape[k+1]:
                            max_shape[k+1] = normals.shape[k]
        gc.collect()

        return max_shape

    @staticmethod
    def load_sample(dirpath, scale, illum_ids=-1, nr_channels=1):
        assert illum_ids == -1
        normal_path = os.path.join(dirpath, '../gt_normal.tif')
        inboundary_path = os.path.join(dirpath, '../inboundary.png')
        onboundary_path = os.path.join(dirpath, '../onboundary.png')

        if not os.path.exists(normal_path):
            raise ValueError("Path\"{:}\"does not exists.".format(normal_path))

        # read ground truth surface normal
        normals = np.float32(cv2.imread(normal_path, -1)) / 65535.0  # [-1,1]
        normals = normals[:, :, ::-1]
        normals = 2 * normals - 1
        normals = cv2.resize(normals, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        normals = normals / np.sqrt(np.sum(normals**2, axis=-1, keepdims=True))
        height, width = np.shape(normals)[:2]

        # read mask images_metallic
        if os.path.exists(inboundary_path) and os.path.exists(onboundary_path):
            inboundary = cv2.imread(inboundary_path, -1)
            inboundary = cv2.resize(inboundary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            inboundary = inboundary > 0

            onboundary = cv2.imread(onboundary_path, -1)
            onboundary = cv2.resize(onboundary, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            onboundary = onboundary > 0

            masks = inboundary | onboundary
        else:
            masks = normals[..., 2] > 0
        masks = masks[..., None]

        # read light filenames
        f = open(os.path.join(dirpath, '../light.txt'))
        data = f.read()
        f.close()
        lines = data.split('\n')
        nr_illums = len(lines) - 1  # the last line is empty (how to fix it?)

        light_directions = np.zeros((nr_illums, 3), np.float32)
        for i, l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                light_directions[i, 0] = float(s[0])
                light_directions[i, 1] = float(s[1])
                light_directions[i, 2] = float(s[2])

        # read images
        images = np.zeros((height, width, nr_illums, nr_channels), np.float32)

        for i in range(nr_illums):
            if i % np.floor(nr_illums / 10) == 0:
                print('.', end='')

            image_path = os.path.join(dirpath, '%05d.tif' % i)

            cv2_im = cv2.imread(image_path, -1) / 65535.0
            cv2_im = cv2.resize(cv2_im, (height, width), interpolation=cv2.INTER_NEAREST)
            if nr_channels == 1:
                cv2_im = (cv2_im[:, :, 0:1] + cv2_im[:, :, 1:2] + cv2_im[:, :, 2:3]) / 3
            images[:, :, i] = cv2_im

        return images, normals, masks, light_directions

    @staticmethod
    def load_sample_test(dir_path, obj_path, scale, index=-1):
        assert index == -1
        obj, dirn = obj_path.split("/")
        return CyclesDataGenerator.load_sample(dir_path + obj, dirn, scale)
