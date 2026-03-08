import matplotlib.pyplot as plt
from PIL import Image
from src.crop import crop_card

ids = [1, 19, 46, 48, 130, 184, 221, 266]

fig, axes = plt.subplots(2, len(ids), figsize=(18, 6))
for col, img_id in enumerate(ids):
    img = Image.open(f'train/{img_id}.JPG').convert('RGB')
    cropped = crop_card(img)
    axes[0, col].imshow(img)
    axes[0, col].set_title(f'id={img_id}', fontsize=7)
    axes[0, col].axis('off')
    axes[1, col].imshow(cropped)
    axes[1, col].set_title(f'{cropped.size}', fontsize=7)
    axes[1, col].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Cropped', fontsize=10)
plt.tight_layout()
plt.savefig('crop_test.png', dpi=150)
plt.show()
print('Saved crop_test.png')
