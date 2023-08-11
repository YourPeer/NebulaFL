
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class resnet20(nn.Module):
    def __init__(self,num_classes=10):
        super(resnet20, self).__init__()


        self.resnet18 = models.resnet18()
        self.resnet18.fc = nn.Linear(512, num_classes)

        # Change BN to GN
        self.resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        self.resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        self.resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        self.resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        self.resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        self.resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        self.resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        self.resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

        assert len(dict(self.resnet18.named_parameters()).keys()) == len(
            self.resnet18.state_dict().keys()), 'More BN layers are there...'

        self.model = self.resnet18


    def forward(self, x,get_features=False):
        if get_features:
            conv_features = nn.Sequential(*list(self.resnet18.children())[:-2])
            z=conv_features(x)
            x = self.resnet18(x)
            return x,z
        else:
            x = self.resnet18(x)
            return x

import torch
if __name__ == "__main__":
    resnet=resnet20(200)
    # for p in resnet.parameters():
    #     print(p.data.size())
    x=torch.rand([10,3,56,56])
    y=resnet(x,True)
    # print(y.size())