import os
#os.environ["PYTORCH_SDP_ATTENTION"] = "disable"
import torch
#print(torch.__version__)
#torch.backends.cuda.enable_flash_sdp = False
#torch.backends.cuda.enable_mem_efficient_sdp = False
#torch.backends.cuda.enable_math_sdp = True
import torch.nn as nn
import torch.nn.functional as F
#print(F.scaled_dot_product_attention)

#import functools

## Save original function
#original_sdp_attention = F.scaled_dot_product_attention

#@functools.wraps(original_sdp_attention)
#def safe_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
#    return original_sdp_attention(
#        query, key, value,
#        attn_mask=attn_mask,
#        dropout_p=dropout_p,
#        is_causal=is_causal,
#        backend='math'
#    )

## Monkey-patch
#F.scaled_dot_product_attention = safe_attention

import torchvision.transforms as transforms
from torchvision.datasets import Imagenette
#from torchvision.models import vit_b_16
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import time
import json
import csv
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    #transforms.GaussianBlur(kernel_size=15, sigma=0.3),  #just try it and see...
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

clean_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

blurred_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=15, sigma=3.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Load datasets (assumes already downloaded)
root_dir = './data/imagenette'
train_dataset = Imagenette(root=root_dir, split='train', size='320px', download=False, transform=train_transform)
clean_val_dataset = Imagenette(root=root_dir, split='val', size='320px', download=False, transform=clean_val_transform)
blurred_val_dataset = Imagenette(root=root_dir, split='val', size='320px', download=False, transform=blurred_val_transform)

batch = 24

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)
clean_val_loader = DataLoader(clean_val_dataset, batch_size=batch, shuffle=False, num_workers=4)
blurred_val_loader = DataLoader(blurred_val_dataset, batch_size=batch, shuffle=False, num_workers=4)

'''# Load ImageNet WNID to index mapping
with open('./imagenet_mapping.json', 'r') as f:
    imagenet_mapping = json.load(f)
wnid_to_index = {v[0]: int(k) for k, v in imagenet_mapping.items()}

# Extract WNIDs from dataset
wnid_label_map = {}
for path, label in train_dataset._samples:
    wnid = os.path.basename(os.path.dirname(path))
    if wnid not in wnid_label_map:
        wnid_label_map[wnid] = label

# Reorder indices to match label order
imagenette_indices = [wnid_to_index[wnid] for wnid, _ in sorted(wnid_label_map.items(), key=lambda x: x[1])]'''

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


#  class label alignment
wnid_label_map = {}

# Extract WNIDs from file paths
for path, label in clean_val_dataset._samples:
    wnid = os.path.basename(os.path.dirname(path))
    if wnid not in wnid_label_map:
        wnid_label_map[wnid] = label

# Sort by label index to match dataset ordering
sorted_wnids = sorted(wnid_label_map.items(), key=lambda x: x[1])
for wnid, label in sorted_wnids:
    imagenet_index = wnid_to_index.get(wnid, None)


# Reconstruct imagenette_indices in label index order
imagenette_indices = [wnid_to_index[wnid] for wnid, _ in sorted(wnid_label_map.items(), key=lambda x: x[1])]

# Load pretrained ViT and replace head
#model = vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
#model.heads = nn.Linear(model.heads.in_features, 10)
#model = model.to(device)


# Symmetry term using precomputed output
def SymTerm(y_pred, X, VF, Connection=None):
    comp_funcs = VF(X)
    grads = torch.stack([
        torch.autograd.grad(
            outputs=y_pred[:, i],
            inputs=X,
            grad_outputs=torch.ones_like(y_pred[:, i]),
            create_graph=True
        )[0] * comp_funcs
        for i in range(y_pred.shape[1])
    ])
    grads = torch.sum(grads, dim=2)
    return torch.transpose(grads, 0, 1)


# Placeholder for your custom kernel
def gaussblurIG(img,kernel,num_channels):
    #from torch.nn.functional import conv2d #, grid_sample, interpolate, pad as torch_pad

    kernel = kernel.repeat(num_channels,1,1)
    kernel = kernel.reshape((1,num_channels,kernel.shape[1],kernel.shape[2])).to(device)
        
    img = F.conv2d(img, kernel, padding='same') #, groups=img.shape[-3])
    return img

kerneldf = pd.read_csv("kernel15.csv", header=None) # kernel of size 15.
kernel = torch.tensor(kerneldf.to_numpy()).float() # we might have to convert this to float, at which point we'll lose some precision.
blurVF = lambda x: gaussblurIG(x,kernel,3)

# Loss function
def vector_field_loss(y_pred, y_target, X_f=None, symmetry_loss_weight=0.5,
                      criterion_model=nn.CrossEntropyLoss(), criterion_sym=nn.MSELoss(), sym_mult=1.0):
    if symmetry_loss_weight == 0:
        symmetry_loss = torch.tensor([0.0], device=y_pred.device)
        total_loss = criterion_model(y_pred, y_target)
    else:
        symmetry_loss = sym_mult * criterion_sym(X_f, torch.zeros_like(X_f))
        total_loss = (1.0 - symmetry_loss_weight) * criterion_model(y_pred, y_target) + symmetry_loss_weight * symmetry_loss
    return total_loss, symmetry_loss


# Training loop
def train_with_symmetry(model, trainloader, VF, symmetry_loss_weight=0.5, optimizer_class=torch.optim.Adam,
                        nosymtime=0.5, criterion_model=nn.CrossEntropyLoss(), criterion_sym=nn.MSELoss(),
                        lr=1e-4, momentum=0.0, n_epochs=5, sym_mult=1.0):
    optimizer = optimizer_class(model.parameters(), lr=lr, momentum=momentum)
    sym_losses, times, accs = [], [], []

    for epoch in range(n_epochs):
        tstart = time.time()
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad_()

            temp_symmetry_loss_weight = 0.0 if epoch < n_epochs * nosymtime else symmetry_loss_weight
            y_pred0 = model(inputs) # shape: [batch_size, 1000]
            y_pred = y_pred0[:, imagenette_indices]  # shape: [batch_size, 10]

            if temp_symmetry_loss_weight == 0.0:
                loss, sym_loss = vector_field_loss(y_pred, labels, X_f=None,
                                                   symmetry_loss_weight=temp_symmetry_loss_weight,
                                                   criterion_model=criterion_model,
                                                   criterion_sym=criterion_sym,
                                                   sym_mult=sym_mult)
            else:
                X_f = SymTerm(y_pred, inputs, VF)
                loss, sym_loss = vector_field_loss(y_pred, labels, X_f=X_f,
                                                   symmetry_loss_weight=temp_symmetry_loss_weight,
                                                   criterion_model=criterion_model,
                                                   criterion_sym=criterion_sym,
                                                   sym_mult=sym_mult)

            sym_losses.append(sym_loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], minibatch [{i}], Loss: {loss.item():.6f}")

        times.append(time.time() - tstart)

        # Evaluate on clean and blurred validation sets
        model.eval()
        def evaluate(loader):
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)  # shape: [batch_size, 1000]
                    outputs_filtered = outputs[:, imagenette_indices]  # shape: [batch_size, 10]
                    predicted = torch.argmax(outputs_filtered, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            return correct / total

        acc_clean = evaluate(clean_val_loader)
        acc_blurred = evaluate(blurred_val_loader)
        accs.append((acc_clean, acc_blurred))
        print(f"Epoch [{epoch + 1}/{n_epochs}], Clean Accuracy: {100 * acc_clean:.2f}%, Blurred Accuracy: {100 * acc_blurred:.2f}%")

    return model, sym_losses, times, accs

# Run training
for i in range(5):
    type = "sym" # this gets changed when running without symmetry (symmetry weight of 0)
    #model = vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
    model = resnet50() #resnet50(weights="ResNet50_Weights.DEFAULT")
    #for param in model.parameters():
    #    param.requires_grad = False
    #for param in model.fc.parameters():
    #    param.requires_grad = True
    #model.heads = nn.Linear(model.heads.in_features, 10)
    model = model.to(device)

    result = train_with_symmetry(
        model, train_loader, blurVF,
        symmetry_loss_weight=0.5,
        criterion_model=nn.CrossEntropyLoss(),
        criterion_sym=nn.MSELoss(),
        n_epochs=2,
        nosymtime=0.5,
        optimizer_class=torch.optim.SGD,
        momentum=0.9,
        lr=1e-2,
        sym_mult=(3 * 224 * 224) ** 1
    )

    print("Validation Accuracies (Clean, Blurred):", result[3])

    with open(type + '/' + type + f'_losses_{i}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in result[1]:
            writer.writerow([row])

    with open(type + '/' + type + f'_times_{i}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in result[2]:
            writer.writerow([row])

    pd.DataFrame(result[3], columns=["Clean Accuracy", "Blurred Accuracy"]).to_csv(type + '/' + type + f"_accs_{i}.csv")
    torch.save(model.state_dict(), type + '/' + type + f"_model_{i}.pt")
