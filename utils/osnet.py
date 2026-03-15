import torch
import torch.nn as nn
import torch.nn.functional as F

##########
# Basic building blocks
##########

class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution with 1x1 (linear) + dw 3x3 (nonlinear).
    """
    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

##########
# OSNet building blocks
##########

class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = ConvLayer(in_channels, mid_channels, 1)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = nn.Sequential(
            ConvLayer(mid_channels, mid_channels, 1),
            ConvLayer(mid_channels, mid_channels, 1),
        )
        self.conv3 = ConvLayer(mid_channels, out_channels, 1)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a + x2b + x2c + x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    
    Implementation adapted for simplified usage.
    """
    def __init__(self, num_classes=1000, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                 channels=[64, 256, 384, 512], loss='softmax', **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss

        # Standard Conv1
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Body
        self.layer2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True)
        self.layer3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.layer4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        
        # Head (ReID specific)
        self.conv5 = ConvLayer(channels[3], channels[3], 1)
        
        # We generally use Global Average Pooling later, so we just return features here.
        self.feature_dim = channels[3]

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size):
        layers = []
        
        # Transition/Downsampling (if needed)
        pass 
        # Actually in OSNet, standard ResNet structure:
        # First block does stride if needed.
        # But here `channels` jumps.
        # Let's follow standard implementation logic
        
        layers.append(block(in_channels, out_channels))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels))
            
        # Transition
        if reduce_spatial_size:
            layers.append(nn.Sequential(
                ConvLayer(out_channels, out_channels, 1),
                nn.AvgPool2d(2, stride=2)
            ))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        return x

def osnet_x1_0(pretrained=False, **kwargs):
    # Construct OSNet 1.0
    model = OSNet(blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[64, 256, 384, 512], **kwargs)
    if pretrained:
        # Load custom weights if available, or skip.
        # Since I cannot download from web, I will initialize randomly or user can provide path.
        print("Warning: Pretrained OSNet weights not available, using random init.")
    return model
