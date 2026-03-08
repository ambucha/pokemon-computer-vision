import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import build_model
from src.dataset import val_transforms


# --- Grad-CAM ---

def gradcam(model, img_tensor, device):
    model.eval()

    activations = {}
    gradients   = {}

    def forward_hook(_, __, output): activations['value'] = output
    def backward_hook(_, __, output): gradients['value']  = output[0]

    target_layer = model.layer4[-1].conv2
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    output     = model(img_tensor)
    model.zero_grad()
    output.backward()

    handle_f.remove()
    handle_b.remove()

    if 'value' not in gradients:
        raise RuntimeError("Backward hook didn't fire — check target layer")

    acts  = activations['value'].squeeze()        # [512, 7, 7]
    grads = gradients['value'].squeeze()          # [512, 7, 7]

    weights = grads.mean(dim=(1, 2))              # [512]
    cam     = (weights[:, None, None] * acts).sum(dim=0)  # [7, 7]
    cam     = F.relu(cam)
    cam     = cam - cam.min()
    cam     = cam / (cam.max() + 1e-8)

    return cam.detach().cpu().numpy()


def show_gradcam(img_path, model, device, ax, title):
    img    = Image.open(img_path).convert('RGB')
    tensor = val_transforms(img)

    with torch.no_grad():
        prob = model(tensor.unsqueeze(0).to(device)).sigmoid().item()

    cam = gradcam(model, tensor, device)

    cam_resized = (
        np.array(
            Image.fromarray((cam * 255).astype(np.uint8))
                 .resize((224, 224), Image.BILINEAR)
        ) / 255.0
    )
    img_resized = img.resize((224, 224))

    ax.imshow(img_resized)
    ax.imshow(cam_resized, cmap='jet', alpha=0.45)
    ax.set_title(f'{title}\np={prob:.2f}', fontsize=9)
    ax.axis('off')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    for param in model.parameters():
        param.requires_grad = True

    train_df = pd.read_csv('train_labels.csv')

    n = 4
    samples = (train_df.groupby('label')
                       .apply(lambda x: x.sample(n, random_state=42))
                       .reset_index(drop=True))

    fig, axes = plt.subplots(2, n, figsize=(14, 7))
    for row, label in enumerate([0, 1]):
        group = samples[samples['label'] == label]
        for col, (_, r) in enumerate(group.iterrows()):
            img_path = f'train/{int(r["id"])}.JPG'
            show_gradcam(img_path, model, device, axes[row, col], f'id={int(r["id"])} label={label}')
        axes[row, 0].set_ylabel(f'Class {label}', fontsize=12, fontweight='bold')

    plt.suptitle('Grad-CAM — what the model looks at', fontsize=13)
    plt.tight_layout()
    plt.savefig('gradcam.png', dpi=150)
    plt.show()
    print('Saved gradcam.png')