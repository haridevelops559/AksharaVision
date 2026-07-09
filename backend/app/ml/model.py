import torch
import torch.nn as nn
import timm


class KannadaClassifier(nn.Module):
    def __init__(
        self,
        backbone_name="swin_tiny_patch4_window7_224",
        embed_dim=512,
        num_classes=113
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        feature_dim = self.backbone.num_features

        self.fc_embed = nn.Linear(feature_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.fc_embed(features)
        embedding = self.bn(embedding)
        embedding = torch.relu(embedding)
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)

        return logits, embedding