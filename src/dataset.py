from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.crop import crop_card


class CropCard:
    def __call__(self, img: Image.Image) -> Image.Image:
        return crop_card(img)


class PokemonDataset(Dataset):
    def __init__(self, df, split, transform=None):
        self.df = df
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f'{self.split}/{int(row["id"])}.JPG').convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['label']


train_transforms = transforms.Compose([
    CropCard(),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),  # random crop forces model to ignore fixed-position cues
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
])

val_transforms = transforms.Compose([
    CropCard(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
