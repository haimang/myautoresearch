import mlx.nn as nn
from .backbone import ResNetBackbone
from .heads import MultiHeadOutput

class FloorplanNet(nn.Module):
    def __init__(self, num_blocks: int, num_filters: int):
        super().__init__()
        self.backbone = ResNetBackbone(num_blocks, num_filters)
        self.heads = MultiHeadOutput(num_filters)

    def __call__(self, x):
        features = self.backbone(x)
        return self.heads(features)
