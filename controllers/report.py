from controllers.modules import *
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from skimage import io, segmentation, morphology
import os
import base64


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

    def get_marked_img(self, img, heatmap):
        image = img[0]
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
        img = img.astype(np.float32)
        input_img = Variable(torch.from_numpy(img).cuda())
        tags_prob = []
        marked_img = []
        for i, model in enumerate(modelslist):
            output = model(input_img)
            prob_tensor = F.softmax(output['classification']).data
            prob = prob_tensor.cpu().numpy()
            heatmap = F.softmax(output['segmentation']).data.cpu().numpy()
            overlaid_image = self.get_marked_img(img[0], heatmap[0])
            io.imsave("tmp.png", overlaid_image)
            with open("tmp.png", "rb") as imageFile:
                b64img = base64.b64encode(imageFile.read())
                marked_img.append(b64img)
            tags_prob.append(prob[0][1])

        data_to_send = {}
        for i, tag in enumerate(tagslist):
            datum = {
                'name': tagnames[i],
                'prob': tags_prob[i],
                'description': tagdescription[i],
                'img': marked_img[i]
            }
            data_to_send[tag] = datum
        return data_to_send


    def get(self):
        # upload audio file in server
        fle = self.get_argument("file")
        pre = self.predict(os.path.realpath(os.path.join(__UPLOADS__, fle)))
        data = {
                "fname": "http://localhost:8000/" + __UPLOADS__ + fle,
                "pred_data": pre
        }
        self.render("report.html", data=data)
