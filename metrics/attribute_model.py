# -*- coding: utf-8 -*-
from metrics.mobilenetv2 import mobilenetv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from misc.utils import denorm
def load_model(model, model_path):
    pretrained_dict = torch.load(model_path,
                                 map_location=lambda storage, loc: storage)
    pretrained_dict = {
        k.replace('module.', ''): v
        for k, v in pretrained_dict['state_dict'].items()
    }    
    model.load_state_dict(pretrained_dict)


class AttNet(nn.Module):
    def __init__(self, mask=False, verbose=True):
        super().__init__()
        self.model = mobilenetv2(mask=mask, num_attributes=7)
        if mask:
            self.pretrained_model = 'models/pretrained_models/attribute_mask_male_female_eyeglasses_hair_bangs_earrings_hat.pth'
        else:
            self.pretrained_model = 'models/pretrained_models/attribute_rgb_male_female_eyeglasses_hair_bangs_earrings_hat.pth'
        if verbose:
            print(f'=> Loading AttNet from {self.pretrained_model}')
        load_model(self.model, self.pretrained_model)        
        self.model.eval()
        self.mask = mask
        # if self.mask:
        #     self.normalize_data = None
        # else:
        #     # Model trained by finetuning imagenet weights 
        #     self.normalize_data = {
        #         'mean':[0.485, 0.456, 0.406],
        #         'std':[0.229, 0.224, 0.225]
        #                             }                            

        for param in self.parameters():
            param.requires_grad = False        

    def forward(self, x, one_hot=False):
        # images [-1, 1], masks [0, 1]
        x_size = x.size(0)
        if self.mask:
            x = F.interpolate(x, (224, 224), mode='nearest')
            x = denorm(x)
        else:
            # x = ((x + 1) / 2.).clamp_(0, 1)
            x = F.interpolate(x, (224, 224), mode='bilinear')
            x = denorm(x)
            # x = self.transformations(x)
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device) 
            x = (x - self.mean) / self.std
        feat = []
        for layer in self.model.features:
            x = layer(x)
            feat.append(x.clone())
        attributes = self.model.predict(x)
        # output = torch.cat(attributes, dim=1)
        output = []
        # import ipdb; ipdb.set_trace()
        for attr in attributes:
            _attr = F.softmax(attr, dim=1)
            output.append(_attr[:,1].unsqueeze(1))
        output = torch.cat(output, dim=1).to(x.device)
        if one_hot:
            output = self.one_hot(output)
        return feat, output

    def one_hot(self, x):
        attr_real0 = (x[:,:2]>0.5).long() # NO GENDER BINARIZATION
        attr_real1 = (x[:,2:]>0.5).long()
        attr_real1 = torch.cat([torch.cat((1-i, i), dim=1) for i in attr_real1.chunk(attr_real1.size(1), dim=1)], dim=1)
        label = torch.cat([attr_real0, attr_real1], dim=1)
        return label

    def compute_loss(self, tensor1, tensor2):
        criterion_l1 = torch.nn.L1Loss()
        feat_x1, cls_x1 = self(tensor1)
        feat_x2, cls_x2 = self(tensor2)
        loss_feat = 0
        for f1, f2 in zip(feat_x1, feat_x2):
            loss_feat += criterion_l1(f1, f2)
        loss_feat /= len(feat_x1)
        loss_cls = criterion_l1(cls_x1, cls_x2)
        return loss_feat, loss_cls     

    def calculate_attr_given_images(self, group_of_images):
        attr_values = []
        num_rand_outputs = len(group_of_images)

        # calculate the average of attributes
        # among all random outputs
        for i in range(num_rand_outputs):
            attr_values.append(self(group_of_images[i])[-1])
        return torch.mean(torch.stack(attr_values, dim=0), dim=0)  