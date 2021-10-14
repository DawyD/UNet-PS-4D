import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from data.cyclesps import CyclesDataGenerator
from misc.metrics import AvgAngleMetric, GaussAvgAngleMetrics

from models.cnnps import densenet_2D
from models.pxnet import pxnet_2D_v2, pxnet_2D_v1
from models.unet import unet
from models.cnnps_sep4D import densenet_separable4D, densent_separable4D_3x3, densent_separable4D_5x5
from models.unet_sep4D import unet_sep4d

from misc.layres import Random90Rotation, RotateVector

from misc.projections import parse_projection
from misc.losses import GaussMSE, ProjectedSoftmax2D
from misc.layres import Gauss2D

from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


layers_dict = {"RotateVector": RotateVector,
               "Random90Rotation": Random90Rotation,
               "AvgAngleMetric": AvgAngleMetric,
               "GaussMSE": GaussMSE,
               "SpatialGaussMSE": GaussMSE,
               "GaussAvgAngleMetrics": GaussAvgAngleMetrics,
               "Gauss2D": Gauss2D,
               "ProjectedSoftmax2D": ProjectedSoftmax2D,
               "tf": tf}

parser = argparse.ArgumentParser(description='Photometric Stereo network training')
parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs')
parser.add_argument('--size', '-w', type=int, default=48, help='Size of the observation map')
parser.add_argument('--neighbours', '-n', type=int, default=3, help='Size of the spatial patch')
parser.add_argument('--rotations', '-K', type=int, default=12, help='Number of rotations')
parser.add_argument('--model', '-m', type=str, default="cnnps",
                    help='Architecture type [cnnps, cnnps4D, unet, unet4D, pxnet]')
parser.add_argument('--batch_size', '-b', type=int, default=768, help='Batch size')
parser.add_argument('--save_every', '-v', type=int, default=1, help='Save model every [V] epochs')
parser.add_argument('--nr_features', '-f', type=int, default=16, help='Nuber of features for U-Net based models')
parser.add_argument('--nr_blocks', '-q', type=int, default=3, help='Nuber of blocks for U-Net based models')
parser.add_argument('--hm_std', type=float, default=2, help='Heat-map standard deviation for U-Net based models')
parser.add_argument('--order', type=int, default=3, help='Spline interpolation order used if neighbours > 1')
parser.add_argument('--use_BN', dest="use_BN", action="store_true", help='Use BatchNormalization layer in U-Net')
parser.add_argument('--dividemaps', dest="dividemaps", action="store_true",
                    help='Divide observation maps by the max value e.g. for cnnps model')
parser.add_argument('--suffix', type=str, default="", help='Suffix to the name under which the models will be saved')
parser.add_argument('--memory_limit', '-g', type=int, default=-1,
                    help='If > 0, GPU memory to allocate in MB on the first GPU. If <= 0 all GPUs are fully allocated')
parser.add_argument('--add_raw', dest="add_raw", action="store_true",
                    help="Adds RAW (non-normalized) colour channels to the observation maps")
parser.add_argument('--optimizer', type=str, default="rmsprop", help="Optimizer")
parser.add_argument('--dataset', type=str, default="cycles", help="Training dataset [cycles]")
parser.add_argument('--dataset_path', type=str, default="datasets/CyclesPS/", help="Path to the dataset")

args = parser.parse_args()

# Set random seeds
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# Set the memory limit for the GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus and args.memory_limit > 0:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

# Define strategy for Multi-GPU training
mirrored_strategy = tf.distribute.MirroredStrategy()

# Define projection from the 3D unit sphere to a 2D plane
projection = parse_projection("standard")

# Define & compile the selected model
nr_channels = 4 if args.add_raw else 1

with mirrored_strategy.scope():
    if args.model == "cnnps":
        args.neighbours = 1
        model = densenet_2D(args.size, args.size, nr_channels)
        model.compile(optimizer=args.optimizer, loss='mean_squared_error', metrics=[AvgAngleMetric()])
        is4D = False

    elif args.model == "cnnps4D":
        if args.neighbours == 3:
            model = densent_separable4D_3x3((args.size, args.size), nr_channels, False)
        elif args.neighbours == 5:
            model = densent_separable4D_5x5((args.size, args.size), nr_channels, False)
        else:
            model = densenet_separable4D((args.size, args.size), (args.neighbours, args.neighbours), nr_channels, False)
        model.compile(optimizer=args.optimizer, loss='mean_squared_error', metrics=[AvgAngleMetric()])
        is4D = True

    elif args.model == "unet":
        args.neighbours = 1
        model = unet((args.size, args.size, nr_channels), 1, args.nr_features, args.nr_blocks, 2, args.useBN)

        lr_schedule = ExponentialDecay(0.0005, decay_steps=(1000000/args.batch_size), decay_rate=0.985, staircase=True)
        if args.optimizer == "rmsprop":
            optimizer = RMSprop(learning_rate=lr_schedule)
        elif args.optimizer == "adam":
            optimizer = Adam(learning_rate=lr_schedule)
        else:
            raise ValueError("Only rmsprop and adam optimizers are supported by this model")

        model.compile(optimizer=optimizer,
                      loss=GaussMSE(args.hm_std, args.size),  # loss=ProjectedSoftmax2D(args.size),
                      metrics=[GaussAvgAngleMetrics(args.size, 1, spherical=False)])
        is4D = False

    elif args.model == "unet4D":
        if args.neighbours == 3 or args.neighbours == 5 or args.neighbours == 7 or args.neighbours == 9:
            model = unet_sep4d((args.neighbours, args.neighbours, args.size, args.size, nr_channels), 1, args.nr_features, args.nr_blocks, 2, args.useBN)
        else:
            raise NotImplementedError("Only spatial patches of 3x3, 5x5, 7x7 and 9x9 are supported by this model")

        lr_schedule = ExponentialDecay(0.001, decay_steps=(1000000 / args.batch_size), decay_rate=0.985, staircase=True)
        if args.optimizer == "rmsprop":
            optimizer = RMSprop(learning_rate=lr_schedule)
        elif args.optimizer == "adam":
            optimizer = Adam(learning_rate=lr_schedule)
        else:
            raise ValueError("Only rmsprop and adam optimizers are supported by this model")

        model.compile(optimizer=optimizer,
                      loss=GaussMSE(args.hm_std, args.size),
                      metrics=[GaussAvgAngleMetrics(args.size, 1, spherical=False)])
        is4D = True

    elif args.model == "pxnet":
        # model = pxnet_2D_v1(args.size, args.size, nr_channels)
        model = pxnet_2D_v2(args.size, args.size, nr_channels)
        model.compile(optimizer=args.optimizer, loss='mean_squared_error', metrics=[AvgAngleMetric()])
        is4D = False

    else:
        raise NotImplementedError("Unknown model, use: cnnps, cnnps4D, unet, unet4D, or pxnet.")

model.summary(line_length=120)
print(model.optimizer)
print(model.loss)


# Define the training & validation data generators
dataset_args = {
    'batch_size': args.batch_size, 'shuffle': True, 'random_illums': True,
    'spatial_patch_size': args.neighbours, 'obs_map_size': args.size, 'keep_axis': is4D, 'projection': projection,
    'add_raw': args.add_raw, 'divide_maps': args.dividemaps,
    'nr_rotations': args.rotations, 'order': args.order, 'rot_2D': False,
}

if args.dataset == "cycles":
    dg_train = CyclesDataGenerator(args.dataset_path, objlist=None, validation_split=0.1, **dataset_args)
    dg_valid = dg_train.get_validation_generator()
else:
    raise ValueError("Unknown dataset, only 'cycles' dataset is currently supported")

# Define the callbacks for the training
model_chackpoint = ModelCheckpoint("checkpoints/M" + args.model + "_K" + str(args.rotations) +
                                   "_N" + str(args.neighbours) + "_B" + str(args.batch_size) +
                                   "_s" + str(args.seed) + args.suffix + "-{epoch:02d}-{val_loss:.4f}.hdf5",
                                   monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                                   mode='auto', period=args.save_every)

# Launch the training
hist = model.fit(dg_train, epochs=args.epochs, verbose=1, callbacks=[model_chackpoint],
                 validation_data=dg_valid, initial_epoch=0)
