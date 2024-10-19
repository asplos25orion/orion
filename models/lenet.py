import torch.nn as nn
import orion.nn as on


class ConvBlock(on.Module):
    def __init__(self, Ci, Co, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            on.Conv2d(Ci, Co, kernel_size, stride, padding, bias=False),
            on.BatchNorm2d(Co),
            on.SiLU(degree=127))
    
    def forward(self, x):
        return self.conv(x)
    

class LinearBlock(on.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.linear = nn.Sequential(
            on.Linear(ni, no),
            on.BatchNorm1d(no),
            on.SiLU(degree=127))
        
    def forward(self, x):
        return self.linear(x)


class AlexNet(on.Module):
    cfg = [64, 'M', 192, 'M', 384, 256, 256, 'A']

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers()
        self.flatten = on.Flatten()
        self.classifier = nn.Sequential(
            LinearBlock(1024, 4096),
            LinearBlock(4096, 4096),
            on.Linear(4096, num_classes))

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in self.cfg:
            if x == 'M':
                layers += [on.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [on.AdaptiveAvgPool2d((2, 2))]
            else:
                layers += [ConvBlock(in_channels, x, kernel_size=3, 
                                     stride=1, padding=1)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = AlexNet()
    net.eval()

    x = torch.randn(1,3,32,32)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (3,32,32), depth=10)
    print("Total flops: ", total_flops)




NEW

20:43
import orion.nn as on

class LeNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.bn1 = on.BatchNorm2d(32)
        self.act1 = on.Quad()
        
        self.conv2 = on.Conv2d(32, 64, kernel_size=5, padding=2, stride=2) 
        self.bn2 = on.BatchNorm2d(64)
        self.act2 = on.Quad()    
        
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(7*7*64, 512)
        self.bn3 = on.BatchNorm1d(512)
        self.act3 = on.Quad() 
        
        self.fc2 = on.Linear(512, num_classes)

    def forward(self, x): 
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.bn3(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    from fvcore.nn import FlopCountAnalysis

    net = LeNet()
    net.eval()

    x = torch.randn(1,1,28,28)
    total_flops = FlopCountAnalysis(net, x).total()

    summary(net, (1,28,28), depth=10)
    print("Total flops: ", total_flops)
