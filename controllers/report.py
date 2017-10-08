from controllers.modules import *
from deeplearing.model import Resnet9
import torch
from skimage import io
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import io, segmentation, morphology


__UPLOADS__ = "static/uploads/"

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


class ReportHandler(RequestHandler):
    """Class to upload the file and generate report."""

    def __init__(self):
        super(ReportHandler, self).__init__()
        self.tags = ["peffusion", "nodule", "atelectasis"]
        self.name = ["Pleural Effusion", "Nodule", "Atelectasis"]
        self.mpath = ["deeplearning/best_model_peffusion.pt",
                      "deeplearning/best_model_nodule.pt",
                      "deeplearning/best_model_atelectasis.pt"]
        self.models = []
        for mpath in self.mpath:
            model_data = torch.load(mpath)
            sd = model_data['network']
            model = Resnet9()
            model.load_state_dict(sd)

            model.eval()
            model.cuda()
            model = nn.DataParallel(model)
            self.models.append(model)
        self.description = [
            "A pleural effusion is a buildup of fluid in the pleural space, an area between the layers of tissue that line the lungs and the chest cavity. It may also be referred to as effusion or pulmonary effusion.", "A lung nodule is defined as a “spot” on the lung that is three centimeters (about 1.5 inches) in diameter or less. These nodules are often referred to as `coin lesions` when described on an imaging test. If an abnormality is seen on an x-ray of the lungs is larger than three centimeters, it is considered a “lung mass” instead of a nodule and is more likely to be cancerous.", "Atelectasis is the collapse or closure of a lung resulting in reduced or absent gas exchange."]

    def get_marked_img(self, img, heatmap):
        image = images[0]
        image = zero_one_normalize(image)
        prob_image = heatmap[1]
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
        return overlaid_image


    def predict(self, filelocation):
        """Predict from image."""
        img = io.imread(filelocation)
        img_low = np.percentile(img, 1)
        img_high = np.percentile(img, 2)
        img = np.clip(img, img_low, img_high)
        img = img - img.min()
        img = img / img.max()
        data_mean = 0.5182097657604547
        data_std = 0.08537056890509876
        img = (img - data_mean) / data_std
        img = img[np.newaxis, np.newaxis, :]
        input_img = Variable(torch.from_numpy(img).cuda())
        tags_prob = []
        marked_img = []
        for i, model in enumerate(self.models):
            output = model(input_img)
            prob_tensor = F.softmax(output['classification']).data
            prob = prob_tensor.cpu().numpy()
            heatmap = F.softmax(output['segmentation']).data.cpu().numpy()
            input_images = input_img.cpu().numpy()
            overlaid_image = self.get_marked_img(input_images[i], heatmap[i])
            tags_prob.append(prob[i][1])
            marked_img.append(overlaid_image)


    def get(self):
        # upload audio file in server
        fle = self.get_argument("file")
        data = {
                "fname" : __UPLOADS__ + fle
        }
        self.render("report.html", data=data)

