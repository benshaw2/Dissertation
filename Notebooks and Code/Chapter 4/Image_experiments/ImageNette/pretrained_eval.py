import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader
import urllib.request
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Clean transform
'''clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Blurred transform
blurred_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(kernel_size=15, sigma=3.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])'''

# Clean transform
clean_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Blurred transform
blurred_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=15, sigma=3.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Download ImageNette
root_dir = './data/imagenette'
clean_dataset = Imagenette(root=root_dir, split='val', size='320px', download=True, transform=clean_transform)
blurred_dataset = Imagenette(root=root_dir, split='val', size='320px', download=True, transform=blurred_transform) # was 160

clean_loader = DataLoader(clean_dataset, batch_size=64, shuffle=False, num_workers=4)
blurred_loader = DataLoader(blurred_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load pretrained ViT
model = vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
model.eval()
model = model.to(device)

# ImageNette WNIDs
imagenette_wnids = [
    'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
    'n03197337', 'n03417042', 'n03425413', 'n03888257', 'n03970156'
]

# Load ImageNet class index mapping from GitHub (valid JSON)
mapping_url = "https://raw.githubusercontent.com/AntonotnaWang/imagenet_and_pytorch_pretrained_model_id_mapping/master/data/modeloutput_id_wnid_class.json"
mapping_path = "./imagenet_mapping.json"

# Download the mapping file if not already present
if not os.path.exists(mapping_path):
    urllib.request.urlretrieve(mapping_url, mapping_path)

# Load the JSON mapping
with open(mapping_path, 'r') as f:
    imagenet_mapping = json.load(f)

# Map WNIDs to ImageNet-1K indices
#wnid_to_index = {entry['wnid']: int(entry['id']) for entry in imagenet_mapping}
wnid_to_index = {v[0]: int(k) for k, v in imagenet_mapping.items()}
#imagenette_indices = [wnid_to_index[wnid] for wnid in imagenette_wnids]


# Diagnostic: Check class label alignment
print("\n--- Class Label Alignment Check ---")
wnid_label_map = {}

# Extract WNIDs from file paths
for path, label in clean_dataset._samples:
    wnid = os.path.basename(os.path.dirname(path))
    if wnid not in wnid_label_map:
        wnid_label_map[wnid] = label

# Sort by label index to match dataset ordering
sorted_wnids = sorted(wnid_label_map.items(), key=lambda x: x[1])
for wnid, label in sorted_wnids:
    imagenet_index = wnid_to_index.get(wnid, None)
    #print(f"Label Index: {label}, WNID: {wnid}, ImageNet-1K Index: {imagenet_index}")

#print("\nFiltered ImageNet indices used for evaluation:")
#print(imagenette_indices)
#print("-----------------------------------\n")

# Reconstruct imagenette_indices in label index order
imagenette_indices = [wnid_to_index[wnid] for wnid, _ in sorted(wnid_label_map.items(), key=lambda x: x[1])]

# Evaluation function with filtered logits
def evaluate_filtered(loader, description):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)  # shape: [batch_size, 1000]
            outputs_filtered = outputs[:, imagenette_indices]  # shape: [batch_size, 10]
            preds = torch.argmax(outputs_filtered, dim=1)
            labels = labels.to(device)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy on {description}: {100 * correct / total:.2f}%")

# Run evaluations
evaluate_filtered(clean_loader, "Clean ImageNette")
evaluate_filtered(blurred_loader, "Blurred ImageNette")
