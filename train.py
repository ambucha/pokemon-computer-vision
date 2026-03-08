import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import PokemonDataset, train_transforms, val_transforms
from src.model import build_model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        correct    += ((preds.sigmoid() > 0.5) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            preds = model(imgs)
            loss  = criterion(preds, labels)

            total_loss += loss.item() * len(imgs)
            correct    += ((preds.sigmoid() > 0.5) == labels).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


if __name__ == '__main__':
    train_df = pd.read_csv('train_labels.csv')
    test_df  = pd.read_csv('test_labels.csv')

    train_dataset = PokemonDataset(train_df, 'train', train_transforms)
    test_dataset  = PokemonDataset(test_df,  'test',  val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    # Phase 1: train head only
    print('--- Phase 1: head only ---')
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, test_loader,  criterion, device)
        print(f'Epoch {epoch+1}/10 | '
              f'Train loss: {train_loss:.4f} acc: {train_acc:.3f} | '
              f'Val loss: {val_loss:.4f} acc: {val_acc:.3f}')

    # Phase 2: unfreeze everything, fine-tune at low lr with early stopping
    print('--- Phase 2: full fine-tune ---')
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(20):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, test_loader,  criterion, device)
        print(f'Epoch {epoch+1}/20 | '
              f'Train loss: {train_loss:.4f} acc: {train_acc:.3f} | '
              f'Val loss: {val_loss:.4f} acc: {val_acc:.3f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Saved best model (val loss: {best_val_loss:.4f})')