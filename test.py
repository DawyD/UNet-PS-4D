import tensorflow as tf
from tensorflow.keras.models import load_model
from misc.layres import RotateVector, Random90Rotation, Gauss2D
from misc.losses import GaussMSE, ProjectedSoftmax2D
from misc.metrics import AvgAngleMetric, GaussAvgAngleMetrics
from misc.projections import parse_projection
from misc.evaluate import test_network

import argparse

from data.cyclesps import CyclesDataGenerator
from data.diligent import DiLiGenTDataGenerator

parser = argparse.ArgumentParser(description='Dynamic SGM Net')
parser.add_argument('model', type=str, default="weight_and_model.hdf5", help='Path to the stored keras model')
parser.add_argument('--dataset', '-d', type=str, default="diligent",
                    help='Name of the dataset [diligent, cycles_uniform17, cycles_uniform305]')
parser.add_argument('--dataset_path', type=str, default="DiLiGenT/pmsData/", help="Path to the dataset")
parser.add_argument('--mode', '-e', type=str, default="test", help='Select mode [test, predict]')
parser.add_argument('--rotations', '-K', type=int, default=12, help='Save path')
parser.add_argument('--projection', '-p', type=str, default="standard", help='Projection type [standard]')
parser.add_argument('--memory_limit', '-g', type=int, default=-1, help='Ratio of GPU memory to allocate')
parser.add_argument('--batch_size', '-b', type=int, default=256, help="Batch size")
parser.add_argument('--dividemaps', dest="dividemaps", action="store_true", help='Divide observation maps by the max value e.g. for 2D model')
parser.add_argument('--order', type=int, default=3, help='Spline interpolation order used if neighbours > 1')
parser.add_argument('--k_size', type=int, default=5, help='Size of the patch in the heatmap to estimate the centre of the mass')
args = parser.parse_args()
print(args)

# Define projection from the 3D unit sphere to a 2D plane (if applicable)
projection = parse_projection(args.projection)

# Define the the test dataset
if args.dataset == "diligent":
    # Set the directory list
    objlist = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG', 'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']
    loading_fn = DiLiGenTDataGenerator.load_sample
    scale = 1
elif args.dataset == "cycles_uniform17":
    # Set the directory list
    objlist = ["{:}/{:}".format(j, i) for i in ['images_specular', 'images_metallic'] for j in ['paperbowl.obj', 'sphere.obj', 'turtle.obj']]
    loading_fn = CyclesDataGenerator.load_sample
    scale = 1
elif args.dataset == "cycles_uniform305":
    # Set the directory list
    objlist = ["{:}/{:}".format(j, i) for i in ['images_specular', 'images_metallic'] for j in ['paperbowl.obj', 'sphere.obj', 'turtle.obj']]
    loading_fn = CyclesDataGenerator.load_sample
    scale = 1
else:
    raise ValueError("Unknown dataset")

# Restrict TensorFlow to only allocate portion of memory on the first GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus and args.memory_limit > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.memory_limit)])

# Load pretrained model
layers_dict = {"RotateVector": RotateVector,
               "Random90Rotation": Random90Rotation,
               "AvgAngleMetric": AvgAngleMetric,
               "GaussMSE": GaussMSE,
               "SpatialGaussMSE": GaussMSE,
               "GaussAvgAngleMetrics": GaussAvgAngleMetrics,
               "Gauss2D": Gauss2D,
               "ProjectedSoftmax2D": ProjectedSoftmax2D,
               "tf": tf}

print(args.model)
model = load_model(args.model, layers_dict)

# Determine the input and output types based on the input and output shape
is4D = len(model.inputs[0].shape) > 4
add_raw = (model.inputs[0].shape[-1] == 4)
if is4D:
    obs_map_size = model.inputs[0].shape[3]
    neighbourhood_size = model.inputs[0].shape[1]
    is_output_gauss = len(model.outputs[0].shape) > 3 and model.outputs[0].shape[-1] == 1
else:
    obs_map_size = model.inputs[0].shape[1]
    neighbourhood_size = 1
    is_output_gauss = len(model.outputs[0].shape) == 4 and model.outputs[0].shape[-1] == 1

print("Model input shape:", model.inputs[0].shape,
      "=> 4D:", is4D,
      "=> obs. map size:", obs_map_size,
      "=> neigbourhood size:", neighbourhood_size,
      "=> add_raw:", add_raw)
print("Model output shape:", model.outputs[0].shape,
      "=> heat-map:", is_output_gauss)

#model.summary(line_length=120)
#print(model.optimizer)
#print(model.loss)

print(args.dataset_path)
print(objlist)

# Test the network
test_network(model, args.dataset_path, objlist, loading_fn, args.rotations, obs_map_size, neighbourhood_size,
             keepaxis=is4D, projection=projection, is_output_gauss=is_output_gauss,
             add_raw=add_raw, batch_size=args.batch_size, divide_maps=args.dividemaps,
             order=args.order, gauss_k_size=args.k_size, rot_2D=False, print_time=True)
