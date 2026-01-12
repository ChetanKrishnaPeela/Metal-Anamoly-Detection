import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.models as models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class_names = [
    "Burn_Through","Cluster_Porosity","Concavity","Crack",
    "Elongated_Pores","Excess_Penetration","External_Undercut",
    "Lack_of_Fusion","Pin_Hole","Pore",
    "Root_Undercut","Tungsten_Inclusion"
]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 13)
model.load_state_dict(torch.load("xray_defect_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
])

img = Image.open("/Users/p.chetankrishna/xray_defect_classification/image copy.png")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    pred = output.argmax(1).item()

print("Predicted Defect:", class_names[pred])
