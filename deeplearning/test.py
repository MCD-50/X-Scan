"""Test file."""
import torch
import progressbar
from model import Resnet9
from dataloader import get_train_val_iterators
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torchnet.meter import AUCMeter, ConfusionMeter
import configargparse
import numpy as np
from PIL import Image
from matplotlib import cm
from skimage import io, segmentation, morphology
import os


parser = configargparse.get_argument_parser()
parser.add('-c', '--config', required=True, is_config_file=True,
           help='config file path')
parser.add('--n_classes', required=True, type=int, help="Number of classes.")
parser.add('--mpath', required=True, help='Model path to be tested.')

options, unknown_options = parser.parse_known_args()

model_data = torch.load(options.mpath)
sd = model_data['network']
model = Resnet9()
model.load_state_dict(sd)

model.eval()
model.cuda()
# print(model)
model = nn.DataParallel(model)


def zero_one_normalize(array):
    """Normalize array in 0 and 1.

    .. math :: I_{norm} = (I - I_{min}) / (I_{max} - I_{min})

    Arguments
    ---------
    array: ndarray
        Input array to be normalized.

    Returns
    -------
    ndarray
        Returns a normalized array. If min value and max
        value of array are equal, then returns the array.

    """
    if array.min() == array.max():
        if array.max() == 0:
            return array
        return array / array.max()
    array = array - array.min()
    array = array / array.max()
    return array


def save_heatmaps(images, heatmaps):
    for i in range(images.shape[0]):
        image = images[i][0]
        image = zero_one_normalize(image)
        prob_image = heatmap[i][1]
        lower_clip = np.percentile(prob_image, 10)
        upper_clip = np.percentile(prob_image, 90)
        prob_image = np.clip(prob_image, lower_clip, upper_clip)
        # prob_image = zero_one_normalize(prob_image)
        # print(prob_image.max(), prob_image.mean(), prob_image.min())
        threshold_prob = 0.4
        prob_image[prob_image < threshold_prob] = 0
        prob_image[prob_image >= threshold_prob] = 1
        prob_image = prob_image.astype(int)
        prob_image = morphology.remove_small_objects(prob_image, min_size=100)
        overlaid_image = segmentation.mark_boundaries(
            image, prob_image, color=(1, 0, 0), mode='thick')
        # print(merged_image)
        output_path = os.path.join(
            'examples', '{}_{}_{}_{}'.format(
                prob[i][1], target[i], options.buckets,
                os.path.basename(data['filepath'][i])))
        io.imsave(output_path, overlaid_image)


auc_meter = AUCMeter()
conf_meter = ConfusionMeter(options.n_classes)
iterator = get_train_val_iterators(options)
bar = progressbar.ProgressBar()
for batch_idx, data in bar(enumerate(iterator['val']())):
    output = model(Variable(data['input'].cuda(), volatile=True))
    target = data['target'].cpu().numpy()
    prob_tensor = F.softmax(output['classification']).data
    prob = prob_tensor.cpu().numpy()
    heatmap = F.softmax(output['segmentation']).data.cpu().numpy()
    auc_meter.add(prob[:, 1], target)
    conf_meter.add(prob_tensor, data['target'])

    input_images = data['input'].cpu().numpy()
    # save_heatmaps(input_images, heatmap)

print(auc_meter.value()[0])
print(conf_meter.value())
