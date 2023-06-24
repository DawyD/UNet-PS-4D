import tensorflow as tf
import numpy as np
from os import path

from misc.projections import standard_proj
from data.datagenerator import DataGenerator, rotate_vectors

from scipy.ndimage import center_of_mass
from time import time


# Test and evaluate network
def test_network(model, dir_path, objlist, loading_fn, nr_rotations, obs_map_size, spatial_size,
                 keepaxis=True, projection=standard_proj, is_output_gauss=False, batch_size=768, add_raw=False,
                 divide_maps=False, order=2, gauss_k_size=5, rot_2D=False, print_time=False):
    """
    Tests the network and prints the errors
    :param model: Trained keras model
    :param dir_path: Dataset path
    :param objlist: List of objects to be tested (dir_path/obj)
    :param loading_fn: A function that given path, scale, illumination ids, and number of channels outputs
                       images, normals, masks and light directions
    :param nr_rotations: Number of rotations over which the result is averaged (for isotropic BRDFs)
    :param obs_map_size: Size of the observation map (has to be compatible with the model)
    :param spatial_size: Size of the spatial neighbourhood (has to be compatible with the model)
    :param keepaxis: If False it squeezes the spatial dimensions in the case of 1x1 spatial size
    :param projection: Projection from the 3D unit sphere to a 2D plane
    :param is_output_gauss: True if the output of the model is a heat-map instead of the 3-vector
    :param batch_size: Batch size for evaluation
    :param add_raw: True if non-normalized colour channels should be included in the observation maps
    :param divide_maps: True if each observation map should be divided by its maximum value
    :param order: Order of the spline interpolation in the case of spatial rotation
    :param gauss_k_size: Nr of pixels in the neighbourhood of the maximal value of the heat-map to be used for
                         estimating the normal with the sub-pixel accuracy
    :param rot_2D: Force spatial rotation also for the 1x1 patch sizes
    :param print_time: Print elapsed time
    """

    # Define and print the angles of rotations
    angles = [i * 2.0 * np.pi / nr_rotations for i in range(nr_rotations)]
    print('Angles: ', end="")
    for r, angle in enumerate(angles):
        print('{:.0f}, '.format(180 * angle / np.pi), end="")
    print(' => Average')
    results = []

    # Define the time counters
    t_loading = 0
    t_it = 0
    t_generation = 0
    t_prediction = 0
    t_backprojection = 0
    t_writing = 0

    # Iterate over the objects
    nr_channels = 3 if add_raw else 1
    for obj in objlist:
        images, normals, masks, illum_dirs = loading_fn(path.join(dir_path, obj), 1, -1, nr_channels)

        print("{:}: ".format(obj), end="")
        img_shape = np.shape(images)[:2]
        est = np.zeros((nr_rotations,) + img_shape + (3,), np.float32)

        # Iterate over the rotations
        for r, angle in enumerate(angles):
            t = time()
            # Define the data generator outputting the observation maps and normals under the desired rotation
            dg = DataGenerator(images=images, normals=normals, masks=masks, illum_dirs=illum_dirs,
                               spatial_patch_size=spatial_size, obs_map_size=obs_map_size, keep_axis=keepaxis,
                               batch_size=batch_size, shuffle=False, random_illums=False, nr_rotations=1,
                               rotation_start=angle, projection=projection, add_raw=add_raw, divide_maps=divide_maps,
                               order=order, rot_2D=rot_2D)
            t_loading += (time() - t)

            error_num = 0
            error_den = 0
            for k in range(len(dg) + 1):

                # Get the observation maps and normals
                t = time()
                embeds, nmls = dg[k]
                t_generation += (time() - t)
                t_it += len(embeds)

                t = time()
                # Predict the normals
                outputs = model(embeds, training=False)
                t_prediction += (time() - t)

                t = time()
                # If output is a heat-map, estimate the cartesian coordinates of the normals
                if is_output_gauss:
                    outputs = get_vectors_from_maps(outputs, gauss_k_size=gauss_k_size)
                else:
                    outputs = outputs.numpy()
                t_backprojection += (time() - t)

                # Compute the error between the GT normals and predictions in degrees
                error_num += 180 * np.sum(np.arccos(np.minimum(1, np.abs(np.einsum('ij,ij->i', outputs, nmls))))) / np.pi
                error_den += len(nmls)

                # Rotate the vector to the principal orientation of the scene
                # and write it at the corresponding spatial position
                t = time()
                outputs = rotate_vectors(outputs, -angle)
                idx_obj, idx_y, idx_x = dg.get_indices(k)
                est[r, idx_y, idx_x] = outputs
                t_writing += (time() - t)

            # Print the average error for all the pixels of a single object under single rotation
            print('{:.2f}, '.format(error_num / error_den), end="", flush=True)

        # Compute the mean over the rotated versions of predictions
        est = np.mean(est, axis=0)
        norm = np.sqrt(est[..., 0] * est[..., 0] + est[..., 1] * est[..., 1] + est[..., 2] * est[..., 2])[..., None]
        est2 = np.zeros_like(est)
        np.divide(est, norm, out=est2, where=norm != 0)

        # Get the valid pixels
        valid = (np.max(np.mean(images, axis=-1), axis=-1, keepdims=True) > 0) & masks

        # Compute the error between the averaged predictions and the GT normals
        err = np.abs(np.einsum('kij,kij->ki', est2, normals))[..., None]  # dot product
        err = 180 * np.arccos(np.minimum(1, err)) * valid / np.pi
        res = np.sum(err) / np.sum(valid)
        print('=> {:.2f}'.format(res))
        results.append(res)

    # print the average results for all the objects
    for i, res in enumerate(results):
        if i != 0:
            print(",", end="")
        print("{:.2f}".format(res), end="")
    print()

    if print_time:
        print("Average loading time: {:0.2f}us".format(1000*1000*t_loading / t_it))
        print("Average generation time: {:0.2f}us".format(1000*1000*t_generation / t_it))
        print("Average prediction time: {:0.2f}us".format(1000*1000*t_prediction / t_it))
        print("Average backprojection time: {:0.2f}us".format(1000*1000 * t_backprojection / t_it))
        print("Average writing time: {:0.2f}us".format(1000*1000*t_writing / t_it))


def get_vectors_from_maps(maps, gauss_k_size: int = 1):
    """
    Estimate the normal direction from a heat-map with sub-pixel accuracy (if gauss_k_size > 1)
    :param maps: Tensorflow tensor with observation maps of shape [B, H, W]
    :param gauss_k_size: Nr of pixels in the neighbourhood of the maximal value of the heat-map to be used for
                         estimating the maximum with the sub-pixel accuracy
    """
    # Get the coordinates of the maximum value in the heat-map
    w = np.array(maps.shape[1:3])
    max_idx = tf.argmax(tf.reshape(maps, (-1, w[0] * w[1])), axis=1)
    bbs = tf.transpose(tf.unravel_index(max_idx, w))[:, ::-1].numpy()

    if gauss_k_size != 1:
        ks_half = gauss_k_size // 2
        # pad the observation map so that the patch from the heat-map we extract is always valid
        padded_outputs = tf.pad(maps[..., 0], ((0, 0), (ks_half, ks_half), (ks_half, ks_half)), mode="reflect")

        # extract the patch of size gauss_k_size around the max-value and compute the center of mass to get the
        # sub-pixel estimate of the true maximum. Note: fitting a Gaussian would be more accurate, but also slower
        p1 = bbs + gauss_k_size
        xy = np.empty((len(bbs), 2), dtype=np.float32)
        for b in range(len(bbs)):
            xy[b, 1], xy[b, 0] = center_of_mass(padded_outputs[b, bbs[b, 1]:p1[b, 1], bbs[b, 0]:p1[b, 0]].numpy())
        xy = xy + bbs - ks_half

        if not np.all(np.isfinite(xy)):
            print("Some outputs are NaNs, falling back to k_size=1")
            xy = bbs
    else:
        xy = bbs

    # Rescaling the coordinates in the heat-map into [-1, 1] range
    xy = (xy / ((w - 1) / 2)) - 1

    # Compute the z axis for the xy coordinates on the unit sphere.
    z = np.sqrt(1 - np.minimum(xy[:, 0] ** 2 + xy[:, 1] ** 2, 1.))[..., None]

    return np.concatenate((xy, z), axis=1)

