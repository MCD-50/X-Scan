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
from skimage import io
import os


parser = configargparse.get_argument_parser()
parser.add('-c', '--config', required=True, is_config_file=True,
           help='config file path')
parser.add('--n_classes', required=True, type=int, help="Number of classes.")

options, unknown_options = parser.parse_known_args()

model_path = 'best_model_peffusion.pt'
tag = 'peffusion'

model_data = torch.load(model_path)
sd = model_data['network']
model = Resnet9()
model.load_state_dict(sd)

model.eval()
model.cuda()
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


def merge_image_heatmap(image, heatmap, color=(255, 255, 255)):
    """Merge an image and heatmap to create a single image.

    Arguments
    ---------
    image: ndarray
        A H*W*3 array. Must be an RGB image.
    heatmap: ndarray
        A H*W array of probability of each pixel, probability is directly
        proportional to the importance of pixel in outcome.
    color: tuple(int)
        A size 3 tuple of int.

    Returns
    -------
    ndarray
        Numpy aray with heatmap embedded onto image and opacity of 0.25.

    """
    assert image.shape[2] == 3, \
        "Input image has {} channels, must have 3.".format(image.shape[2])
    background = np.concatenate([
        image, np.ones(shape=(heatmap.shape[0], heatmap.shape[1], 1))], axis=2)
    background = np.uint8(background * 255)
    background_image = Image.fromarray(background)
    foreground = np.uint8(cm.jet(heatmap) * 255)
    heatmap_opacity = foreground[:, :, 3]
    heatmap_opacity[:] = 64
    threshold_prob = min(0.3, heatmap.max() - 0.05)
    heatmap_opacity[heatmap < threshold_prob] = 0
    foreground_image = Image.fromarray(foreground)
    image = Image.alpha_composite(background_image, foreground_image)
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])
    image_array = np.array(background, dtype=np.uint8)
    return image_array


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
    for i in range(input_images.shape[0]):
        image = np.repeat(input_images[i], 3, axis=0)
        image = zero_one_normalize(image)
        image = np.transpose(image, (1, 2, 0))
        prob_image = zero_one_normalize(heatmap[i][prob[i].argmax(0)])
        lower_clip = np.percentile(prob_image, 10)
        upper_clip = np.percentile(prob_image, 90)
        pixel_weights = np.clip(prob_image, lower_clip, upper_clip)
        pixel_weights = zero_one_normalize(prob_image)
        print(prob_image.max(), prob_image.mean(), prob_image.min())
        merged_image = merge_image_heatmap(image, prob_image)
        # print(merged_image)
        output_path = os.path.join(
            'examples', '{}_{}_{}_{}'.format(
                prob[i][1], target[i], options.buckets,
                os.path.basename(data['filepath'][i])))
        io.imsave(output_path, merged_image)
    break

print(auc_meter.value()[0])
print(conf_meter.value())
