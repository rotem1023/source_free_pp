import os
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm

from backpack import extend
import copy


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResBase(nn.Module):
    def __init__(self):
        super(ResBase, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
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
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            
            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x
    

class FullModel(torch.nn.Module):
    def __init__(self, netF, netB, netC):
        super(FullModel, self).__init__()
        self.netF = netF
        self.netB = netB
        self.netC = netC

    def forward(self, x):
        features = self.netF(x)
        embeddings = self.netB(features)
        outputs = self.netC(embeddings)
        return outputs
    
class SfppFeatureNetwork(torch.nn.Module):
    def __init__(self, netF, netB):
        super(SfppFeatureNetwork, self).__init__()
        self.netF = netF
        self.netB = netB

    def forward(self, x):
        features = self.netF(x)
        embeddings = self.netB(features)
        return embeddings

    
class SfppModel:
    def __init__(self, model_name, dataset, src_domain, tgt_domain, year):
        netF, netB, netC = load_model_parts(model_name=model_name, dataset=dataset, src_domain=src_domain,
                                        tgt_domain=tgt_domain, year=year)
        self.feature_extractor = SfppFeatureNetwork(netF=netF, netB=netB)
        self.classifier = netC

        # We extend the classifier layer of the model to use the backpack library for the gradient computation
        self.classifier = extend(self.classifier)


def _get_models_dir():
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    return f"{current_file_path}/../data/models"



def load_model_parts(model_name, dataset, src_domain, tgt_domain, year):
    model_path = f"{_get_models_dir()}/{dataset}/{src_domain}{tgt_domain}/{year}/{model_name}.pth"
    dict_to_load = torch.load(model_path)
    
    # default values
    feature_dim=2048
    type_bottleneck='bn'
    E_dims=256
    num_C = 12
    
    netF = ResBase()
    netB = feat_bottleneck(feature_dim=feature_dim, bottleneck_dim=E_dims, type=type_bottleneck)
    netC = feat_classifier(num_C, E_dims, type="wn")
    
    for component in dict_to_load:
        if component == 'M':
            netF.load_state_dict(dict_to_load[component], strict=False)
        elif component == 'E':
            netB.load_state_dict(dict_to_load[component], strict=False)
        elif component=='G':
            netC.load_state_dict(dict_to_load[component], strict=False)
    
    netC.eval()
    # netB.eval()
    # netF.eval()
    return netF, netB, netC


def load_model(model_name, dataset, src_domain, tgt_domain, year):
    netF, netB, netC = load_model_parts(model_name=model_name, dataset=dataset, src_domain=src_domain,
                                        tgt_domain=tgt_domain, year=year)
    # for k, v in netC.named_parameters():
    #     v.requires_grad = False
    full_model = FullModel(netF, netB, netC)
    return full_model

