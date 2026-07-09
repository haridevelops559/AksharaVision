import os, csv, json, torch, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = "project_files/kannada_classifier_finetuned_full.pth"
CLASSES_PATH = "project_files/classes_list_113_kannada.json"
CENTROID_PATH = "diagnostics/centroid_top3_neighbors.csv"

class KannadaClassifier(nn.Module):
    def __init__(self, backbone_name="swin_tiny_patch4_window7_224",
                 embed_dim=512, num_classes=113):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True,
            num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.fc_embed = nn.Linear(feat_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.fc_embed(feat)
        emb = self.bn(emb)
        emb = torch.relu(emb)
        emb = self.dropout(emb)
        logits = self.classifier(emb)
        return logits, emb


ckpt = torch.load(CKPT_PATH, map_location=device)
model = KannadaClassifier(num_classes=ckpt.get("num_classes", 113)).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

with open(CLASSES_PATH) as f:
    all_classes = json.load(f)

centroids = torch.tensor(
    pd.read_csv(CENTROID_PATH).iloc[:,1:].values,
    dtype=torch.float32
).to(device)

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(image_path, topk=3):
    image = Image.open(image_path).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, emb = model(tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = probs.topk(topk, dim=1)

    preds = [
        (all_classes[idx.item()],
         all_classes[idx.item()].split("_")[1],
         top_probs[0][i].item())
        for i, idx in enumerate(top_idxs[0])
    ]

    emb = F.normalize(emb, dim=1)
    cent = F.normalize(centroids, dim=1)
    sims = torch.mm(emb, cent.T).squeeze(0)
    sim_vals, sim_idxs = sims.topk(topk)

    refined = [preds[0]]
    for i, idx in enumerate(sim_idxs):
        lbl = all_classes[idx.item()]
        if lbl != refined[0][0]:
            refined.append((lbl, lbl.split("_")[1], sim_vals[i].item()))
        if len(refined) == topk:
            break

    return refined
