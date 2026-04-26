import mlx.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm(channels)

    def __call__(self, x):
        residual = x
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = nn.relu(x + residual)
        return x

class ResNetBackbone(nn.Module):
    def __init__(self, num_blocks: int, num_filters: int):
        super().__init__()
        self.input_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm(num_filters)
        self.blocks = [ResBlock(num_filters) for _ in range(num_blocks)]
        
    def __call__(self, x):
        # x is [N, H, W, C]
        x = nn.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = block(x)
        # Global average pooling over spatial dimensions H(1) and W(2)
        x = x.mean(axis=(1, 2))
        return x
