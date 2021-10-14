"""
DataGenerator for DiLiGenT Dataset
This file use substantial portion of code from the original CNN-PS repository https://github.com/satoshi-ikehata/CNN-PS/
"""

import numpy as np
import os
import cv2
import gc

from tensorflow.keras.models import Model

from data.datagenerator import DataGenerator
from data.utils import rotate_images
from misc.projections import standard_proj


class DiLiGenTDataGenerator(DataGenerator):
    def __init__(self, datapath, objlist=None, batch_size=256,
                 spatial_patch_size=5, obs_map_size=32, shuffle=False, random_illums=False,
                 keep_axis=True, validation_split=None, nr_rotations=1, rotation_start=0, rotation_end=2 * np.pi,
                 projection=standard_proj, add_raw=False, images=None, normals=None, masks=None, illum_dirs=None,
                 order=2, divide_maps=False, round_nearest=True, rot_2D=False, verbose=False):

        super(DiLiGenTDataGenerator, self).__init__(
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

        self.verbose = verbose
        self.datapath = datapath
        self.objlist = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG',
                        'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG'] if objlist is None else objlist

    def load_data(self):
        nr_channels = 3 if self.add_raw else 1
        for objid, obj in enumerate(self.objlist):
            if self.verbose:
                print("\rPre-loading image ({:}/{:}) {:} ".format(objid + 1, self.nr_objects, obj), end="")

            imgs, nmls, msks, light_dirs = self.load_sample(os.path.join(self.datapath, obj), 1.0, -1, nr_channels)

            self.fill_data(imgs, nmls, msks, light_dirs, objid)

            if self.verbose:
                print("", end="\x1b[1K\r")

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
            max_shape[0] += 1
            normals = cv2.imread(os.path.join(self.datapath, obj, '/001.png'), -1)
            # normals = cv2.resize(normals, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            f = open(os.path.join(self.datapath, 'light_directions.txt'))
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

    # prepare observation maps for test data (i.e., DiLiGenT dataset)
    @staticmethod
    def load_sample(dir_path, scale, illum_ids=-1, nr_channels=1):
        #  print('loading ' + '%s ' % d, end="")
        # dirpath = os.path.join(dir_path, d)
        setName = os.path.basename(dir_path.rstrip('/'))  # if dirpath ends in '/' basename returns the empty string
        normal_path = os.path.join(dir_path, 'normal.txt')
        mask_path = os.path.join(dir_path, 'mask.png')

        # get image imgSize
        image_path = os.path.join(dir_path, '001.png')
        cv2_im = cv2.imread(image_path, -1)
        img_shape = np.shape(cv2_im)[:2]

        # read ground truth surface normal
        f = open(normal_path)
        data = f.read()
        f.close()
        lines = np.float32(np.array(data.split('\n')))
        normals = np.reshape(lines, img_shape + (3,))
        normals = cv2.resize(normals, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # when test on Harvest, the surface noraml needs to be fliped upside down
        if setName == 'harvestPNG':
            print("(warning: normals will be flipped upside down)", end="")
            normals = np.flipud(normals)

        resized_shape = np.shape(normals)[:2]

        # read mask
        masks = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks = cv2.resize(masks, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        masks = np.array(masks)[..., None] > 0

        # read light directions
        f = open(os.path.join(dir_path, 'light_directions.txt'))
        data = f.read()
        f.close()
        lines = data.split('\n')
        numLight = len(lines) - 1  # the last line is empty (how to fix it?)

        light_directions = np.zeros((numLight, 3), np.float32)
        for i, l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                light_directions[i, 0] = float(s[0])
                light_directions[i, 1] = float(s[1])
                light_directions[i, 2] = float(s[2])

        # read light intensities
        f = open(os.path.join(dir_path, 'light_intensities.txt'))
        data = f.read()
        f.close()
        lines = data.split('\n')

        light_intensities = np.zeros((numLight, 3), np.float32)
        for i, l in enumerate(lines):
            s = l.split(' ')
            if len(s) == 3:
                light_intensities[i, 0] = float(s[0])
                light_intensities[i, 1] = float(s[1])
                light_intensities[i, 2] = float(s[2])

        if illum_ids == -1:
            if setName == 'bearPNG':
                # the first 20 images of bearPNG have errors, see paper
                illum_ids = range(20, numLight)
                print("(warning: the first 20 illiuminations will be ignored)", end="")
            else:
                illum_ids = range(0, numLight)

        light_directions = light_directions[illum_ids, :]
        light_intensities = light_intensities[illum_ids, :]
        numLight = len(illum_ids)

        # read images
        images = np.zeros(resized_shape + (numLight, nr_channels), np.float32)

        for i, idx in enumerate(illum_ids):
            if i % np.floor(numLight / 10) == 0:
                print('.', end='')
            image_path = os.path.join(dir_path, '%03d.png' % (idx + 1))
            cv2_im = cv2.imread(image_path, -1) / 65535.0
            cv2_im = cv2.resize(cv2_im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if nr_channels == 1:
                cv2_im = (cv2_im[:, :, 0:1] / light_intensities[i, 0] + cv2_im[:, :, 1:2] / light_intensities[i, 1] + cv2_im[:, :, 2:3] / light_intensities[i, 2]) / 3
            else:
                cv2_im /= light_intensities[i, None, None, :]
            images[:, :, i] = cv2_im

        return images, normals, masks, light_directions
