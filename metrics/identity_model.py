# -*- coding: utf-8 -*-
from metrics.arcface_resnet import resnet_face18
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_model(model, model_path):
    # model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path,
                                 map_location=lambda storage, loc: storage)
    pretrained_dict = {
        k.replace('module.', ''): v
        for k, v in pretrained_dict.items()
    }
    # model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)


def process_image(image):
    image = toGray(image)
    flipped_image = torch.flip(image, dims=(-2, -1))
    image = torch.cat((image, flipped_image), dim=0)
    return image

def toGray(tensor):
    # trained with opencv.imread(img, 0)
    # formula: Y = 0.299 R + 0.587 G + 0.114 B
    R, G, B = torch.chunk(tensor, 3, dim=1)
    tensor = 0.299*R + 0.587*G + 0.114*B
    return tensor

class ArcFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet_face18(use_se=False)
        self.pretrained_model = 'models/pretrained_models/arcface_resnet18_110.pth'
        load_model(self.model, self.pretrained_model)
        print(f'=> Loading ArcFace from {self.pretrained_model}')
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False   

    def forward(self, x):
        # images [-1, 1]
        x_size = x.size(0)
        x = F.interpolate(x, (128, 128), mode='bilinear')
        x = process_image(x)
        output = self.model(x)
        fe_1 = output[:x_size]  # normal feat
        fe_2 = output[x_size:]  # flipped feat
        feature = torch.cat((fe_1, fe_2), dim=1)
        return feature

    def compute_cosine_tensors(self, tensor1, tensor2):
        cosineSIM = nn.CosineSimilarity(dim=1, eps=1e-08)
        feat1 = self(tensor1)
        feat2 = self(tensor2)
        return cosineSIM(feat1, feat2)

    def compute_loss_tensors(self, tensor1, tensor2):
        loss = 1 - self.compute_cosine_tensors(tensor1, tensor2)
        loss = loss.mean()
        return loss