import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import optuna

from src.model import build_model
from src.dataset import PokemonDataset, CropCard
from train import train_epoch, eval_epoch


def make_loaders(batch_size, jitter, erasing_p):
    train_transforms = transforms.Compose([
        CropCard(),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter * 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=erasing_p, scale=(0.02, 0.15)),
    ])
    val_transforms = transforms.Compose([
        CropCard(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_df = pd.read_csv('train_labels.csv')
    test_df  = pd.read_csv('test_labels.csv')
    train_loader = DataLoader(PokemonDataset(train_df, 'train', train_transforms), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(PokemonDataset(test_df,  'test',  val_transforms),   batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lr1        = trial.suggest_float('lr1',        1e-4, 1e-2, log=True)
    lr2        = trial.suggest_float('lr2',        1e-5, 1e-3, log=True)
    pos_weight = trial.suggest_float('pos_weight', 1.0,  4.0)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    jitter     = trial.suggest_float('jitter',     0.1,  0.5)
    erasing_p  = trial.suggest_float('erasing_p',  0.2,  0.7)

    train_loader, test_loader = make_loaders(batch_size, jitter, erasing_p)

    model     = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    # phase 1: head only
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr1)
    for _ in range(10):
        train_epoch(model, train_loader, criterion, optimizer, device)

    # phase 2: full fine-tune with early stopping
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr2)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(20):
        train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = eval_epoch(model, test_loader, criterion, device)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break

    return best_val_loss


if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print('\n=== Best trial ===')
    t = study.best_trial
    print(f'  Val loss: {t.value:.4f}')
    for k, v in t.params.items():
        print(f'  {k}: {v}')
