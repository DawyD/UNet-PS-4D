import numpy as np
import math
from scipy.ndimage import rotate


def get_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_images(angle, images, axes=(1, 2), order=3):
    if math.isclose(angle, 0, abs_tol=0.01) or (images.shape[axes[0]] == 1 and images.shape[axes[1]] == 1):
        return images

    if math.isclose(angle, np.pi / 2, abs_tol=0.01):
        images = np.rot90(images, k=1, axes=axes)
    elif math.isclose(angle, np.pi, abs_tol=0.01):
        images = np.rot90(images, k=2, axes=axes)
    elif math.isclose(angle, 3 * np.pi / 2, abs_tol=0.01):
        images = np.rot90(images, k=3, axes=axes)
    else:
        images = rotate(images, angle * 180 / np.pi, axes=axes, reshape=True, order=order)
    return images


def rotate_vectors90(vectors, k):
    if k == 1:
        vectors = np.concatenate((np.flip(vectors[..., :2], axis=-1), vectors[..., 2:]), axis=-1)
        vectors[..., 0] *= -1
    if k == 2:
        vectors[..., :2] *= -1
    if k == 3:
        vectors = np.concatenate((np.flip(vectors[..., :2], axis=-1), vectors[..., 2:]), axis=-1)
        vectors[..., 1] *= -1
    return vectors


def rotate_vectors(x, angle):
    """Inplace rotation of vectors (assuming the last dimension can be interpreted as x,y,z coordinates)"""
    res = x.copy()
    if math.isclose(angle, 0):
        return res
    rotation_matrix = get_rotation_matrix(angle)
    to_rot = np.stack([x[..., 0], x[..., 1]], axis=0)
    shape = to_rot.shape
    to_rot = to_rot.reshape((2, -1))
    to_rot = np.dot(rotation_matrix, to_rot)
    to_rot = to_rot.reshape(shape)
    res[..., 0] = to_rot[0]
    res[..., 1] = to_rot[1]
    return res


def rotate_indices(x, y, angle, shape_x, shape_y):
    if math.isclose(angle, 0, abs_tol=0.01):
        return x, y
    if math.isclose(angle, np.pi / 2, abs_tol=0.01):
        return y, np.abs(x - shape_x + 1)
    elif math.isclose(angle, np.pi, abs_tol=0.01):
        return np.abs(x - shape_x + 1), np.abs(y - shape_y + 1)
    elif math.isclose(angle, 3 * np.pi / 2, abs_tol=0.01):
        return np.abs(y - shape_y + 1), x
    else:
        raise NotImplementedError()
