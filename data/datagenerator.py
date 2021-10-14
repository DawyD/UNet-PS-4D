"""
Base (abstract) class for the DataGenerators
This file use substantial portion of code from the original CNN-PS repository https://github.com/satoshi-ikehata/CNN-PS/
"""

import numpy as np
import math
from tensorflow.keras.utils import Sequence
from misc.projections import standard_proj
import gc
from time import time
from data.utils import rotate_images, rotate_vectors90, rotate_vectors
import logging


class DataGenerator(Sequence):
    def __init__(self, batch_size=256, spatial_patch_size=5, obs_map_size=32, shuffle=True, random_illums=True,
                 keep_axis=True, validation_split=None, nr_rotations=1, rotation_start=0, rotation_end=2 * np.pi,
                 projection=standard_proj, add_raw=False, images=None, normals=None, masks=None, illum_dirs=None,
                 order=2, divide_maps=False, round_nearest=True, rot_2D=False):
        """
        DataGenerator outputting observation maps and normals.
        If images, normals, masks and illum_dirs are set, then they are used for this DataGenerator,
         otherwise an abstract method load_data is called.

        :param batch_size: Batch size
        :param spatial_patch_size: Size of the spatial patches
        :param obs_map_size: Size of the observation maps
        :param shuffle: True if the order of the samples should be random
        :param random_illums: True if the random subsets of available illuminations should be used
        :param keep_axis: If False it squeezes the spatial dimensions in the output in case of 1x1 spatial size
        :param validation_split: Reserve portion of the data for validation purposes
        :param nr_rotations: Number of possible rotations of each of the sample to be included
        :param rotation_start: Lowest angle of the allowed rotations interval
        :param rotation_end: Largest angle of the allowed rotations interval
        :param projection: Projection from the 3D unit sphere to a 2D observation map
        :param add_raw: True if non-normalized colour channels should be included in the observation maps
        :param images: List of image stack of shape [nr_objects] * [height, width, nr_illums]
        :param normals: List of normals stack of shape [nr_objects] * [height, width, 3]
        :param masks: List of masks stack of shape [nr_objects] * [height, width, 1]
        :param illum_dirs: List of illumination directions shape [nr_objects] * [nr_illums, 3]
        :param order: Order of the spline interpolation in the case of spatial rotation
        :param divide_maps: True if each observation map should be divided by its maximum value
        :param round_nearest: False if the projected coordinates should be rounded using floor instead of round operation
        :param rot_2D: Force spatial rotation also for the 1x1 patch sizes
        """

        t = time()

        self.batch_size = batch_size
        self.neighbourhood_size = spatial_patch_size
        self.shuffle = shuffle
        self.random_illums = random_illums
        self.obs_map_size = obs_map_size
        self.contract_spatial_axes = self.neighbourhood_size == 1 and not keep_axis
        self.add_raw = add_raw
        self.order = order
        self.divide_maps = divide_maps

        self.rotations_2D = True if spatial_patch_size != 1 else rot_2D

        """ ------ Allocating the data and handling the rotations ------ """
        # Split rotations into 90deg and others:
        assert rotation_start < rotation_end
        if self.rotations_2D and nr_rotations == 12 and self.isclose(rotation_start, 0) and self.isclose(rotation_end, 2*np.pi):
            self.nr_rotations = 3
            self.rotation_start = 0
            self.rotation_end = np.pi / 2
            self.add_rot90 = True
            logging.info("Pre-loading only 0, 30, 60 deg rotations, 90 deg rotations will be made online")
        elif self.rotations_2D and nr_rotations == 8 and self.isclose(rotation_start, 0) and self.isclose(rotation_end, 2*np.pi):
            self.nr_rotations = 2
            self.rotation_start = 0
            self.rotation_end = np.pi / 2
            self.add_rot90 = True
            logging.info("Pre-loading only 0, 45 deg rotations, 90 deg rotations will be made online")
        elif self.rotations_2D and nr_rotations == 4 and self.isclose(rotation_start, 0) and self.isclose(rotation_end, 2*np.pi):
            self.nr_rotations = 1
            self.rotation_start = 0
            self.rotation_end = np.pi / 2
            self.add_rot90 = True
            logging.info("Pre-loading only 0 deg rotation, 90 deg rotations will be made online")
        else:
            self.nr_rotations = nr_rotations
            self.rotation_start = rotation_start
            self.rotation_end = rotation_end
            self.add_rot90 = False

        # Get rotations angles:
        self.rotations = self.get_rotation_angles(self.nr_rotations, self.rotation_start, self.rotation_end)

        # Get maximum shape
        if images is not None and normals is not None and masks is not None and illum_dirs is not None:
            max_shape = self.get_max_shape_img(normals, len(illum_dirs), self.rotations)
        elif images is None and normals is None and masks is None and illum_dirs is None:
            max_shape = self.get_max_shape(self.rotations if self.rotations_2D else None)
        else:
            raise ValueError("Either specify the images, normals, masks, and illum_dirs do not specify any of these.")
        self.nr_objects = max_shape[0]

        self.ns_half = self.neighbourhood_size // 2
        alloc_shape = [max_shape[1] + 2 * self.ns_half, max_shape[2] + 2 * self.ns_half, max_shape[3]]
        nr_channels = 3 if self.add_raw else 1

        # Allocate arrays:
        if self.rotations_2D:
            self.images = np.zeros([self.nr_rotations, self.nr_objects] + alloc_shape + [nr_channels], np.float32)
            self.masks = np.zeros([self.nr_rotations, self.nr_objects] + alloc_shape[:2] + [1], np.float32)
            self.org_indices = -1 * np.ones([self.nr_rotations, self.nr_objects] + alloc_shape[0:2], np.int32)
        else:
            self.images = np.zeros([1, self.nr_objects] + alloc_shape + [nr_channels], np.float32)
            self.masks = np.zeros([1, self.nr_objects] + alloc_shape[:2] + [1], np.float32)
        self.normals = np.zeros([self.nr_rotations, self.nr_objects] + alloc_shape[:2] + [3], np.float32)
        self.illum_dirs = np.zeros([self.nr_rotations, self.nr_objects] + alloc_shape[2:3] + [3], np.float32)
        self.illum_dirs[:, :, :, 2] = -1  # set angle for invalid illuminations
        self.shapes = np.zeros([self.nr_objects, 3], dtype=np.int32)

        logging.info("Images - shape: {:}, size {:.1f} GB".format(self.images.shape, self.images.nbytes / 1024 / 1024 / 1024))
        logging.info("Masks - shape: {:}, size {:.1f} MB".format(self.masks.shape, self.masks.nbytes / 1024 / 1024))
        logging.info("Normals - shape: {:}, size {:.1f} MB".format(self.normals.shape, self.normals.nbytes / 1024 / 1024))
        logging.info("Illum. dirs - shape: {:}, size {:.1f} KB".format(self.illum_dirs.shape, self.illum_dirs.nbytes / 1024))

        tf = time()
        if images is not None and normals is not None and masks is not None and illum_dirs is not None:
            self.fill_data(images, normals, masks, illum_dirs, 0)
        else:
            self.load_data()

        logging.info("Data filled in {:}s".format(time() - tf))

        self.images = self.images.reshape((-1,) + self.images.shape[2:])
        self.masks = self.masks.reshape((-1,) + self.masks.shape[2:])
        self.normals = self.normals.reshape((-1,) + self.normals.shape[2:])
        self.illum_dirs = self.illum_dirs.reshape((-1,) + self.illum_dirs.shape[2:])

        """ </------ Allocating the data and handling the rotations ------> """

        # Get maximal value for each valid pixel (zero for invalid pixels)
        self.imax = np.amax(self.images[..., 0], axis=-1, keepdims=True)
        self.imax *= self.masks

        # Pad the images so that the spatial patch (neighbourhood) centred at a valid pixel fits the array
        self.padded_img_shape = self.images.shape
        r = np.arange(-self.ns_half, self.ns_half + 1)
        self._path_offsets = r[:, None] * self.images.shape[2] + r  # The patch offsets into the flat image array (2D)
        self.images = np.reshape(self.images, (-1,) + self.padded_img_shape[-2:])  # Flatten the image array

        # Pad the normals so that the spatial patch (neighbourhood) centred at a valid pixel fits the array
        # Note: Due to the rotations, the length of normals might be different than length of the images
        self.padded_normals_shape = self.normals.shape
        self.normals = np.reshape(self.normals, (-1, self.normals.shape[-1])) # Flatten the array

        self.idx_obj, self.idx_y, self.idx_x, _ = np.nonzero(self.imax)

        if self.rotations_2D:
            self.len_norot = len(self.idx_obj) // self.nr_rotations  # TODO: works only for rot90
            self._len = len(self.idx_obj)
        else:
            self.len_norot = len(self.idx_obj)
            self._len = len(self.idx_obj) * self.nr_rotations

        if self.add_rot90:
            self.len_norot90 = self._len
            self._len *= 4

        self.imax = np.reshape(self.imax, (-1, 1))

        # Create indices and split them into train / validation sets
        self.indices = np.arange(self._len)

        if (validation_split is not None) and (validation_split != 0):
            self.indices, self.validation_indices = self.split_train_valid_indices(self.indices, validation_split)

        # Pre-compute angle between view=[0,0,1] and light (in degrees) (for all lights)
        self.angle_lv = 180 * np.arccos(self.illum_dirs[:, :, 2]) / np.pi

        # Pre-compute projection coordinates into the observation map
        light_x, light_y = projection(self.illum_dirs[..., 0], self.illum_dirs[..., 1], self.illum_dirs[..., 2])
        if round_nearest:
            light_x = np.int64(np.round((light_x + 1) * (self.obs_map_size - 1) / 2))
            light_y = np.int64(np.round((light_y + 1) * (self.obs_map_size - 1) / 2))
        else: # To be compatible with the original code of CNN-PS
            light_x = np.int64((light_x + 1) * (self.obs_map_size - 1) / 2)
            light_y = np.int64((light_y + 1) * (self.obs_map_size - 1) / 2)
        self.light_idx = light_y * self.obs_map_size + light_x

        self.on_epoch_end()

        logging.info("DataGenerator loaded in {:}s".format(time() - t))

    def __len__(self):
        """Returns the number of batches per epoch"""
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        """Generates one batch of data"""
        # Generate indices of the batch

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        idx_flat_nmls, idx_flat_imgs, idx_obj_nmls, rot90_ids = self.decouple_indices(indices)

        normals, images, imaxs, light_ids, angles = self.extract_data(idx_flat_nmls, idx_flat_imgs, idx_obj_nmls)

        # Select random illumination mask
        nr_illuminations = self.padded_img_shape[-2]
        if self.random_illums:
            thresholds = np.random.randint(20, 90, size=len(indices))
            anglemask_tmp = angles < thresholds[:, None]
            anglemask = np.zeros_like(anglemask_tmp)

            max_illums = np.random.randint(50, np.min([1000, nr_illuminations]), size=len(indices))
            for am, am_tmp, max_ill in zip(anglemask, anglemask_tmp, max_illums):
                ids = np.where(am_tmp)[0]
                np.random.shuffle(ids)
                am[ids[:max_ill]] = 1
        else:
            anglemask = angles < 90

        # Create the observation map (embed) out of the pixels and light angle indices
        embed = self.create_obs_map(images, imaxs, light_ids, anglemask)

        # Rotate the images by k*90 degrees according to the rot90_ids (in case these rotations should be used)
        if self.add_rot90:
            self.rotate_90_inplace(embed, normals, rot90_ids)

        return embed, normals

    def on_epoch_end(self):
        """Shuffle the indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def get_validation_generator(self, batch_size=None):
        return DataValidGenerator(self, batch_size)

    def get_indices(self, index):
        """Returns the coordinates to the source images for a batch with specified index"""
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        if self.add_rot90:
            indices = indices % self.len_norot90

        if (self.nr_rotations != 1 or self.rotation_start != 0) and (not self.rotations_2D):
            indices = indices % self.len_norot

        idx_obj = self.idx_obj[indices]
        idx_y = self.idx_y[indices]
        idx_x = self.idx_x[indices]

        if (self.nr_rotations != 1 or self.rotation_start != 0) and self.rotations_2D:
            rot_ids = idx_obj // self.nr_objects
            idx_obj = idx_obj % self.nr_objects

            org_indices = self.org_indices[rot_ids, idx_obj, idx_y, idx_x]
            widths = self.shapes[idx_obj, 1]
            idx_y = org_indices // widths
            idx_x = org_indices % widths

        return idx_obj, idx_y, idx_x

    def decouple_indices(self, indices):
        """
        Converts the indices in a batch into the indices into the normals flattened array, images flattened array,
        objects array. Also returns the deisred 90 degree rotation associated with the indices

        Note: In the 1x1 spatial case with rotations, only the normals and
              illum_dirs arrays are expanded with the pre-loaded rotations
        """
        if self.add_rot90:
            rot90_ids = indices // self.len_norot90
            indices = indices % self.len_norot90
        else:
            rot90_ids = None

        if (not self.rotations_2D) and self.nr_rotations != 1:
            # In this case, images and imax and idx arrays have length of self.nr_objects,
            # while normals and illum_dirs array have length of self.nr_objects * self.nr_rotations
            rot_ids = indices // self.len_norot
            indices = indices % self.len_norot
            idx_obj_imgs = self.idx_obj[indices]
            idx_obj_nmls = self.idx_obj[indices] + (rot_ids * self.nr_objects)
        else:
            idx_obj_nmls = self.idx_obj[indices]
            idx_obj_imgs = idx_obj_nmls

        idx_y = self.idx_y[indices]
        idx_x = self.idx_x[indices]
        idx_flat_nmls = np.ravel_multi_index((idx_obj_nmls, idx_y, idx_x), dims=self.padded_normals_shape[:3])

        if (not self.rotations_2D) and self.nr_rotations != 1:
            idx_flat_imgs = np.ravel_multi_index((idx_obj_imgs, idx_y, idx_x), dims=self.padded_img_shape[:3])
        else:
            idx_flat_imgs = idx_flat_nmls

        return idx_flat_nmls, idx_flat_imgs, idx_obj_nmls, rot90_ids

    def get_max_shape_img(self, normals, num_light, rotations):
        max_shape = [1, normals.shape[0], normals.shape[1], num_light]

        if self.rotations_2D:
            # In case of rotations, the width and height might be larger
            for angle in rotations:
                img_shape = rotate_images(2 * np.pi - angle, normals[..., 0], axes=(0, 1), order=0).shape
                for k in range(2):
                    if img_shape[k] > max_shape[k + 1]:
                        max_shape[k + 1] = img_shape[k]

        return max_shape

    def load_data(self):
        raise ValueError("I'm just an abstract method. Please implement me.")

    @staticmethod
    def get_rotation_angles(nr_rotations, rotation_start, rotation_end):
        rot_angle = (rotation_end - rotation_start) / nr_rotations
        return np.array([(rotation_start + (i * rot_angle)) % (2 * np.pi) for i in range(nr_rotations)], dtype=np.float32)

    @staticmethod
    def isclose(x, y):
        return math.isclose(x, y, abs_tol=0.01)

    def split_train_valid_indices(self, indices, validation_split):
        if self.nr_rotations != 1:
            train_indices = []
            val_indices = []
            org_len = self._len // self.nr_rotations
            partial_val_split = 1 - (validation_split / self.nr_rotations)
            for r in range(self.nr_rotations):
                train_indices.append(indices[r * org_len:r * org_len + int(org_len * partial_val_split)])
                val_indices.append(indices[r * org_len + int(org_len * partial_val_split):(r+1) * org_len])

            train_indices = np.concatenate(train_indices)
            val_indices = np.concatenate(val_indices)
        else:
            val_indices = self.indices[int(self._len * (1 - validation_split)):]
            train_indices = self.indices[:int(self._len * (1-validation_split))]

        logging.info("Nr training samples: {:}                    ".format(len(train_indices)))
        logging.info("Nr validation samples: {:}                    ".format(len(val_indices)))

        return train_indices, val_indices

    def extract_data(self, idx_flat_nmls, idx_flat_imgs, idx_obj_nmls):
        """
        Given set of indices return the appropriate data samples
        :param idx_flat_nmls: Indices into a flattened normals array
        :param idx_flat_imgs: Indices into a flattened images array
        :param idx_obj_nmls: Indices of the objects in the normals and illum-related arrays

        Note: In the 1x1 spatial case with rotations, only the normals and
              illum_dirs arrays are expanded with the pre-loaded rotations
        """
        light_ids = self.light_idx[idx_obj_nmls]
        # weights = self.imax[idx_flat_imgs] > 0

        normals = self.normals[idx_flat_nmls]

        patch_indices = self._path_offsets + idx_flat_imgs[:, None, None]  # shape: (batch_size, spatial_y, spatial_x)
        # Select pixels
        image_patches = self.images[patch_indices.flat, ...]
        image_patches = image_patches.reshape((len(idx_obj_nmls), self.neighbourhood_size, self.neighbourhood_size,) + self.padded_img_shape[-2:])
        # mask_patches = self.masks.take(patch_indices)[..., None]
        imax_patches = self.imax.take(patch_indices)[..., None]

        light_angles = self.angle_lv[idx_obj_nmls, :]

        return normals, image_patches, imax_patches, light_ids, light_angles

    def create_obs_map(self, images, imaxs, light_ids, anglemask):
        nr_channels = 4 if self.add_raw else 1

        anglemask = anglemask.reshape((anglemask.shape[0], 1, 1, anglemask.shape[-1])).astype(np.bool_)

        # Create the observation map (embed) out of the pixels and light angle indices
        pixel_mask = (imaxs > 0) & anglemask
        temp = images * pixel_mask[..., None]
        if self.add_raw:
            temp = np.concatenate([(temp[..., 0:1] + temp[..., 1:2] + temp[..., 2:3]) / 3, temp], axis=-1)
        if self.divide_maps:
            np.divide(temp[..., 0], imaxs, out=temp[..., 0], where=pixel_mask)

        embeds = np.zeros((images.shape[0], self.neighbourhood_size, self.neighbourhood_size,
                          self.obs_map_size * self.obs_map_size, nr_channels), np.float32)
        np.put_along_axis(embeds[..., 0], light_ids[:, None, None, :], temp[..., 0], axis=3)
        if self.add_raw:
            np.put_along_axis(embeds[..., 1], light_ids[:, None, None, :], temp[..., 1], axis=3)
            np.put_along_axis(embeds[..., 2], light_ids[:, None, None, :], temp[..., 2], axis=3)
            np.put_along_axis(embeds[..., 3], light_ids[:, None, None, :], temp[..., 3], axis=3)

        if self.contract_spatial_axes:
            embeds = embeds.reshape((embeds.shape[0], self.obs_map_size, self.obs_map_size, nr_channels))
        else:
            embeds = embeds.reshape((embeds.shape[0], self.neighbourhood_size, self.neighbourhood_size,
                                     self.obs_map_size, self.obs_map_size, nr_channels))

        return embeds

    def rotate_90_inplace(self, embeds, normals, rot90_ids):
        for i, k in enumerate(rot90_ids):
            if k != 0:
                if self.neighbourhood_size == 1 and self.contract_spatial_axes:
                    embeds[i] = np.rot90(embeds[i], k=-k, axes=(0, 1))  # observation map clockwise (because y is reversed)
                elif self.neighbourhood_size == 1:
                    embeds[i] = np.rot90(embeds[i], k=-k, axes=(2, 3))  # observation map clockwise (because y is reversed)
                else:
                    embeds[i] = np.rot90(embeds[i], k=k, axes=(0, 1))  # spatial counter-clockwise
                    embeds[i] = np.rot90(embeds[i], k=-k, axes=(2, 3))  # observation map clockwise (because y is reversed)
                normals[i] = rotate_vectors90(normals[i], k=k)  # normals counter clockwise

    def fill_data(self, imgs, nmls, msks, light_dirs, objid):
        # Pre-load the non-90-degree rotations:
        ofst = self.ns_half
        if self.rotations_2D:  # In the case of neighbourhood rotate also the images,
            # rotate images
            for a, angle in enumerate(self.rotations):
                rot_images = rotate_images(angle, imgs, axes=(0, 1), order=self.order)
                rot_normals = rotate_images(angle, nmls, axes=(0, 1), order=0)
                rot_masks = rotate_images(angle, msks, axes=(0, 1), order=0)
                org_indices = np.arange(imgs.shape[0] * imgs.shape[1]).reshape(imgs.shape[:2])
                org_indices = rotate_images(angle, org_indices, axes=(0, 1), order=0)
                shape = rot_images.shape
                self.images[a,      objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :shape[2]] = rot_images
                self.normals[a,     objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :] = rot_normals
                self.masks[a,       objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :] = rot_masks
                self.org_indices[a, objid, ofst:ofst+shape[0], ofst:ofst+shape[1]] = org_indices

                del rot_images, rot_normals, rot_masks, org_indices
        else:  # In the case of no neighbourhood rotate no images are rotated
            shape = imgs.shape
            self.images[0, objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :shape[2]] = imgs
            self.masks[0, objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :] = msks
            for a, angle in enumerate(self.rotations):
                self.normals[a, objid, ofst:ofst+shape[0], ofst:ofst+shape[1], :] = nmls

        # Rotate the normal directions and illumination directions
        shape = imgs.shape
        self.shapes[objid] = shape[:3]
        for a, angle in enumerate(self.rotations):
            self.illum_dirs[a, objid, :shape[2], :] = rotate_vectors(light_dirs, angle)
            self.normals[a, objid] = rotate_vectors(self.normals[a, objid], angle)  # rotate normals

        del imgs, nmls, msks, light_dirs
        gc.collect()

    def get_max_shape(self, rotations):
        raise NotImplementedError("This is just an abstract method, please override it.")


class DataValidGenerator(Sequence):
    def __init__(self, parent: DataGenerator, batch_size):
        self.parent = parent
        self.indices = parent.validation_indices
        self.batch_size = batch_size if batch_size is not None else self.parent.batch_size

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        parent = self.parent
        """Generates one batch of data"""
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        idx_flat_nmls, idx_flat_imgs, idx_obj_nmls, rot90_ids = parent.decouple_indices(indices)
        normals, images, imaxs, light_ids, angles = parent.extract_data(idx_flat_nmls, idx_flat_imgs, idx_obj_nmls)
        # Select random illumination mask
        anglemask = (angles < 90).reshape((len(indices), 1, 1, parent.padded_img_shape[-2])).astype(np.float32)
        # Create the observation map (embed) out of the pixels and light angle indices
        embed = parent.create_obs_map(images, imaxs, light_ids, anglemask)
        if parent.add_rot90:
            parent.rotate_90_inplace(embed, normals, rot90_ids)

        return embed, normals
