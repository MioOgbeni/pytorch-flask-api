import io
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nff

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import pytorch_lightning as pl

from PIL import Image

class ChellesNet(pl.LightningModule):
    def __init__(self, n_classes, learning_rate, decay_factor):
        super().__init__()
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)

        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Linear(num_ftrs, n_classes)
        self.acc = pl.metrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

CHECKPOINT_PATH = 'app/checkpoints/epoch=25-val_loss=0.88.ckpt'
model = ChellesNet.load_from_checkpoint(CHECKPOINT_PATH)
#model.to('cuda')
model.eval()

# image -> tensor
def transform_image(image_bytes):
    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliencyMap = saliency.computeSaliency(image)[1]
    saliencyMap = (saliencyMap * 255).astype('uint8')

    blur = cv2.GaussianBlur(saliencyMap,(5,5),0)
    threshMap = cv2.threshold(blur.astype('uint8'),0,255,cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(threshMap,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    cont = np.vstack(contours)
    hull = cv2.convexHull(cont)
    x,y,w,h = cv2.boundingRect(cont)

    image = Image.open(io.BytesIO(image_bytes))
    image.crop((x,y,x+w,y+h))

    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    image = image_tensor

    outputs = model(image)

    prob = nff.softmax(outputs, dim=1)
    top_p, _ = prob.topk(1, dim=1)

    _, predicted = torch.max(outputs, 1)
    return (predicted, top_p)