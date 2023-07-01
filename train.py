#### read the README.txt file ####
import torch
from IPython.display import Image
from roboflow import Roboflow


# download the dataset from roboflow, use VPN
api_key = "xxxxxxxxxxxxxxxxx"
rf = Roboflow(api_key="")
project = rf.workspace("sorane").project("shorre")
dataset = project.version(3).download("yolov5")

