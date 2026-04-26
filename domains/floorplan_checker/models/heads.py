import mlx.nn as nn

class MultiHeadOutput(nn.Module):
    def __init__(self, in_features: int, num_bed_classes: int = 5, num_bath_classes: int = 4, num_park_classes: int = 5):
        super().__init__()
        self.head_bed = nn.Linear(in_features, num_bed_classes)
        self.head_bath = nn.Linear(in_features, num_bath_classes)
        self.head_park = nn.Linear(in_features, num_park_classes)

    def __call__(self, features):
        logits_bed = self.head_bed(features)
        logits_bath = self.head_bath(features)
        logits_park = self.head_park(features)
        return logits_bed, logits_bath, logits_park
