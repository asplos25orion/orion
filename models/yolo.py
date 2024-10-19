import torch
import torch.nn as nn
import orion.nn as on

from resnet import *


class YOLOv1(on.Module):
    def __init__(self, features, num_bboxes=2, num_classes=20, model_path=''):
        super().__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.model_path = model_path

        self.features = features
        self.conv_layers = self._make_conv_layers()
        self.fc_layers = self._make_fc_layers()
        
        self._init_weights()

    def _init_weights(self):
        if self.model_path:
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
            self.load_state_dict(state_dict, strict=False)

    def _make_conv_layers(self):
        net = nn.Sequential(
            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127),
            on.Conv2d(512, 512, 3, stride=2, padding=1),
            on.SiLU(degree=127),

            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127),
            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127)
        )

        return net

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            on.Flatten(),
            on.Linear(7 * 7 * 512, 4096),
            on.SiLU(degree=127),
            nn.Identity(),
            on.Linear(4096, S * S * (5 * B + C)),
            #nn.Sigmoid()
        )

        return net

    def forward(self, x):
        #S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.features(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        
        #x = x.view(-1, S, S, 5 * B + C)
        return x


def YOLOv1_ResNet34(model_path):
    assert model_path != ''

    backbone = ResNet34()
    net = YOLOv1(backbone, num_bboxes=2, num_classes=20, model_path=model_path)
    return net


if __name__ == "__main__":
    backbone = ResNet34()
    net = YOLOv1(backbone, model_path='model_best.pth')
    net.eval()

    x = torch.randn(1, 3, 448, 448)

    out1 = net(x)
    print(out1.shape)
    print(out1.min(), out1.max(), out1.mean())
