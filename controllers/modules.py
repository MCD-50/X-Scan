"""
Includes all the required modules to include
"""
# tornado modules
from tornado.ioloop import IOLoop
from tornado.escape import json_encode
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.gen import coroutine
from tornado.options import define, options
import torch
from deeplearning.model import Resnet9
import torch.nn as nn

# external modules
import os
from os.path import join, dirname
import uuid
from collections import defaultdict

# prediction load
tagslist = ["peffusion", "nodule", "atelectasis"]
tagnames = ["Pleural Effusion", "Nodule", "Atelectasis"]
mpath_list = ["deeplearning/best_model_peffusion.pt",
              "deeplearning/best_model_nodule.pt",
              "deeplearning/best_model_atelectasis.pt"]
modelslist = []
for mpath in mpath_list:
    model_data = torch.load(mpath)
    sd = model_data['network']
    model = Resnet9()
    model.load_state_dict(sd)
    model.eval()
    model.cuda()
    model = nn.DataParallel(model)
    modelslist.append(model)
tagdescription = [
    "A pleural effusion is a buildup of fluid in the pleural space, an area between the layers of tissue that line the lungs and the chest cavity. It may also be referred to as effusion or pulmonary effusion.", "A lung nodule is defined as a “spot” on the lung that is three centimeters (about 1.5 inches) in diameter or less. These nodules are often referred to as `coin lesions` when described on an imaging test. If an abnormality is seen on an x-ray of the lungs is larger than three centimeters, it is considered a “lung mass” instead of a nodule and is more likely to be cancerous.", "Atelectasis is the collapse or closure of a lung resulting in reduced or absent gas exchange."]
