# -*- coding: utf-8 -*-
from metrics.mobilenetv2 import mobilenetv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from misc.utils import denorm
def load_model(model, model_path):
    pretrained_dict = torch.load(model_path,
                                 map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_dict)


class Hopenet(nn.Module):
    # https://github.com/natanielruiz/deep-head-pose
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    # torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66
    def __init__(self, block=torchvision.models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_bins=66):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        # Pre-processing
        # self.transformations = transforms.Compose([ 
        #     #transforms.Resize(224),
        #     # transforms.CenterCrop(224), transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # self.normalize_data = to_cuda({
        #     'mean':[0.485, 0.456, 0.406],
        #     'std':[0.229, 0.224, 0.225]
        #                         })      
  
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor)

        self.pretrained_model = 'models/pretrained_models/hopenet_robust_alpha1.pkl'
        print(f'=> Loading HopeNet from {self.pretrained_model}')
        load_model(self, self.pretrained_model)        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False          

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, (224,224), mode='bilinear') # (-1, 1)
        x = denorm(x) # -> (0, 1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device) 
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        yaw_predicted = F.softmax(pre_yaw, dim=1)
        pitch_predicted = F.softmax(pre_pitch, dim=1)
        roll_predicted = F.softmax(pre_roll, dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor.to(x.device), dim=1, keepdim=True) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor.to(x.device), dim=1, keepdim=True) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor.to(x.device), dim=1, keepdim=True) * 3 - 99
        return torch.cat([yaw_predicted, pitch_predicted, roll_predicted], dim=1)

    def compute_loss(self, tensor1, tensor2, output=False):
        criterion_l2 = torch.nn.MSELoss(reduction='none')
        pose1 = self(tensor1) if not output else tensor1
        pose2 = self(tensor2) if not output else tensor2
        loss_pose = criterion_l2(pose1, pose2).mean(dim=0)
        return loss_pose     

    def calculate_pose_given_images(self, group_of_images):
        pose_values = []
        num_rand_outputs = len(group_of_images)

        # calculate the average of poses (yaw, pitch and roll) 
        # among all random outputs
        for i in range(num_rand_outputs):
            pose_values.append(self(group_of_images[i]))
        return torch.mean(torch.stack(pose_values, dim=0), dim=0)