"""
PlantTraits2024 - FGVC11 Competition Pipeline
==============================================
Based on top Kaggle solutions (PlantHydra 1st place patterns).
Key techniques:
- EfficientNet-B0 backbone with late fusion (image + tabular)
- Log-scale label encoding/decoding for stable training
- StandardScaler for ancillary features
- R2Loss + Cosine Similarity loss
- Response variation augmentation using trait SD
- Proper outlier clipping and NaN handling
- 5-Fold cross-validation
"""

import os
import gc
import cv2
import math
import random
import warnings
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import torchvision.transforms as T
import timm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
class CFG:
    # Paths - UPDATE THESE FOR YOUR KAGGLE ENVIRONMENT
    COMPETITION_DIR = '/kaggle/input/planttraits2024'
    TRAIN_IMAGES = '/kaggle/input/planttraits2024/train_images'
    TEST_IMAGES = '/kaggle/input/planttraits2024/test_images'
    OUTPUT_DIR = '/kaggle/working'

    # Model
    BACKBONE = 'efficientnet_b0'  # timm model name
    IMG_SIZE = 224
    NUM_TARGETS = 6
    TABULAR_DIM = 163  # ancillary feature columns

    # Training
    EPOCHS = 12
    BATCH_SIZE = 32
    LR = 1e-3
    BACKBONE_LR = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    SEED = 42
    N_FOLDS = 5
    TRAIN_FOLDS = [0]  # which folds to train

    # Loss weights
    R2_LOSS_WEIGHT = 1.0
    COSINE_LOSS_WEIGHT = 0.3
    MSE_LOSS_WEIGHT = 0.1

    # Augmentation
    USE_RESPONSE_VARIATION = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


TRAIT_COLS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
TRAIT_SD_COLS = ['X4_sd', 'X11_sd', 'X18_sd', 'X50_sd', 'X26_sd', 'X3112_sd']


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LABEL ENCODER (Log-scale, from PlantHydra)
# ============================================================
class LabelEncoder:
    """Log10 transform + StandardScaler for targets.
    Critical for stable training - raw targets span orders of magnitude."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, y):
        log_y = np.log10(y + 1e-6)
        self.mean = log_y.mean(axis=0)
        self.std = log_y.std(axis=0)
        self.fitted = True
        print(f"LabelEncoder fitted: mean={self.mean}, std={self.std}")
        return self

    def transform(self, y):
        log_y = np.log10(y + 1e-6)
        return (log_y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        return 10 ** (y_scaled * self.std + self.mean)

    def transform_torch(self, y):
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=y.device)
        std_t = torch.tensor(self.std, dtype=torch.float32, device=y.device)
        log_y = torch.log10(y + 1e-6)
        return (log_y - mean_t) / std_t

    def inverse_transform_torch(self, y_scaled):
        mean_t = torch.tensor(self.mean, dtype=torch.float32, device=y_scaled.device)
        std_t = torch.tensor(self.std, dtype=torch.float32, device=y_scaled.device)
        return 10 ** (y_scaled * std_t + mean_t)


# ============================================================
# DATA PREPROCESSING
# ============================================================
def preprocess_data(cfg):
    """Load and preprocess train/test data with outlier handling."""
    train_df = pd.read_csv(os.path.join(cfg.COMPETITION_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(cfg.COMPETITION_DIR, 'test.csv'))

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Build image paths
    train_df['path'] = train_df['id'].apply(
        lambda x: os.path.join(cfg.TRAIN_IMAGES, str(x) + '.jpeg')
    )
    test_df['path'] = test_df['id'].apply(
        lambda x: os.path.join(cfg.TEST_IMAGES, str(x) + '.jpeg')
    )

    # Check which images exist
    train_df['exists'] = train_df['path'].apply(os.path.exists)
    print(f"Train images found: {train_df['exists'].sum()}/{len(train_df)}")
    train_df = train_df[train_df['exists']].reset_index(drop=True)

    # Identify ancillary columns (first 163 non-id columns)
    all_cols = train_df.columns.tolist()
    ancillary_cols = [c for c in all_cols if c not in
                      ['id', 'path', 'exists'] + TRAIT_COLS + TRAIT_SD_COLS
                      and train_df[c].dtype in ['float64', 'float32', 'int64']]
    # Take first 163 numeric columns as ancillary
    ancillary_cols = ancillary_cols[:cfg.TABULAR_DIM]
    print(f"Ancillary columns: {len(ancillary_cols)}")

    # Handle NaN in ancillary data
    for col in ancillary_cols:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    # Clip outliers in targets (1st-99th percentile)
    for col in TRAIT_COLS:
        if col in train_df.columns:
            low = train_df[col].quantile(0.01)
            high = train_df[col].quantile(0.99)
            train_df[col] = train_df[col].clip(low, high)

    # Handle NaN in targets - fill with median
    for col in TRAIT_COLS + TRAIT_SD_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(train_df[col].median())

    # Ensure targets are positive for log transform
    for col in TRAIT_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].clip(lower=1e-6)

    # StandardScaler for ancillary features
    scaler = StandardScaler()
    train_df[ancillary_cols] = scaler.fit_transform(train_df[ancillary_cols])
    test_df[ancillary_cols] = scaler.transform(test_df[ancillary_cols])

    # Replace any remaining NaN/inf
    train_df[ancillary_cols] = train_df[ancillary_cols].replace(
        [np.inf, -np.inf], 0).fillna(0)
    test_df[ancillary_cols] = test_df[ancillary_cols].replace(
        [np.inf, -np.inf], 0).fillna(0)

    # Fit label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[TRAIT_COLS].values)

    # KFold split
    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    train_df['fold'] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, 'fold'] = fold

    return train_df, test_df, ancillary_cols, label_encoder, scaler


# ============================================================
# DATASET
# ============================================================
class PlantDataset(Dataset):
    def __init__(self, df, ancillary_cols, label_encoder=None,
                 transform=None, is_test=False, use_response_var=False):
        self.df = df.reset_index(drop=True)
        self.ancillary_cols = ancillary_cols
        self.label_encoder = label_encoder
        self.transform = transform
        self.is_test = is_test
        self.use_response_var = use_response_var
        self.paths = self.df['path'].values
        self.metadata = self.df[ancillary_cols].values.astype(np.float32)

        if not is_test:
            self.targets = self.df[TRAIT_COLS].values.astype(np.float32)
            if all(c in self.df.columns for c in TRAIT_SD_COLS):
                self.targets_sd = self.df[TRAIT_SD_COLS].values.astype(np.float32)
            else:
                self.targets_sd = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image
        img_path = self.paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (CFG.IMG_SIZE, CFG.IMG_SIZE))
        except Exception:
            image = np.zeros((CFG.IMG_SIZE, CFG.IMG_SIZE, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        # Tabular data
        tab = torch.tensor(self.metadata[idx], dtype=torch.float32)
        tab = torch.nan_to_num(tab, nan=0.0)

        if self.is_test:
            return {'image': image, 'metadata': tab}

        # Targets
        target = self.targets[idx].copy()

        # Response variation augmentation (from PlantHydra)
        if self.use_response_var and self.targets_sd is not None:
            sd = self.targets_sd[idx]
            noise = np.random.normal(0, sd)
            target = np.clip(target + noise, 1e-6, None)

        target = torch.tensor(target, dtype=torch.float32)
        return {'image': image, 'metadata': tab, 'label': target}


# ============================================================
# TRANSFORMS
# ============================================================
def get_train_transforms(img_size):
    return T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transforms(img_size):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# MODEL - Late Fusion with EfficientNet-B0
# ============================================================
class TabularBranch(nn.Module):
    """MLP for ancillary/tabular features with self-attention."""
    def __init__(self, input_dim=163, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PlantModel(nn.Module):
    """Late Fusion: EfficientNet-B0 (image) + MLP (tabular) -> regression head"""
    def __init__(self, cfg):
        super().__init__()
        # Image backbone
        self.backbone = timm.create_model(
            cfg.BACKBONE, pretrained=True, num_classes=0, global_pool='avg'
        )
        backbone_dim = self.backbone.num_features  # 1280 for efficientnet_b0

        # Freeze early layers for stability
        for name, param in self.backbone.named_parameters():
            if 'blocks.0' in name or 'blocks.1' in name or 'conv_stem' in name or 'bn1' in name:
                param.requires_grad = False

        # Tabular branch
        self.tabular = TabularBranch(cfg.TABULAR_DIM, 128, 128)

        # Fusion head
        fusion_dim = backbone_dim + 128
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, cfg.NUM_TARGETS),
        )

    def forward(self, image, metadata):
        img_feat = self.backbone(image)       # [B, 1280]
        tab_feat = self.tabular(metadata)     # [B, 128]
        fused = torch.cat([img_feat, tab_feat], dim=1)  # [B, 1408]
        out = self.head(fused)                # [B, 6]
        return out


# ============================================================
# LOSSES
# ============================================================
class R2Loss(nn.Module):
    """Direct R2 optimization loss (from PlantHydra)."""
    def forward(self, y_pred, y_true):
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
        ss_tot = torch.sum((y_true - y_true.mean(dim=0)) ** 2, dim=0)
        r2 = ss_res / (ss_tot + 1e-6)
        return r2.mean()


class CombinedLoss(nn.Module):
    """Combined loss: R2Loss + Cosine Similarity + MSE."""
    def __init__(self, r2_w=1.0, cos_w=0.3, mse_w=0.1):
        super().__init__()
        self.r2_loss = R2Loss()
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.mse_loss = nn.MSELoss()
        self.r2_w = r2_w
        self.cos_w = cos_w
        self.mse_w = mse_w

    def forward(self, y_pred, y_true):
        loss_r2 = self.r2_loss(y_pred, y_true)
        loss_cos = (1 - self.cos_sim(y_pred, y_true)).mean()
        loss_mse = self.mse_loss(y_pred, y_true)
        return self.r2_w * loss_r2 + self.cos_w * loss_cos + self.mse_w * loss_mse


# ============================================================
# METRICS
# ============================================================
def compute_r2(y_true, y_pred):
    """Compute mean R2 across all traits (competition metric)."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2_per_trait = 1 - ss_res / (ss_tot + 1e-6)
    # Clip negative R2 to 0 (as per competition rules)
    r2_per_trait = np.clip(r2_per_trait, 0, None)
    return r2_per_trait.mean(), r2_per_trait


# ============================================================
# TRAINING
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    label_encoder, device):
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    pbar = tqdm(loader, desc='Train')
    for batch in pbar:
        images = batch['image'].to(device)
        metadata = batch['metadata'].to(device)
        targets = batch['label'].to(device)

        # Encode targets to log-scale
        targets_enc = label_encoder.transform_torch(targets)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, targets_enc)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Decode predictions for R2 calculation
        with torch.no_grad():
            preds = label_encoder.inverse_transform_torch(outputs)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mean_r2, per_trait = compute_r2(all_targets, all_preds)

    return total_loss / len(loader), mean_r2, per_trait


@torch.no_grad()
def validate(model, loader, criterion, label_encoder, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc='Valid'):
        images = batch['image'].to(device)
        metadata = batch['metadata'].to(device)
        targets = batch['label'].to(device)

        targets_enc = label_encoder.transform_torch(targets)
        outputs = model(images, metadata)
        loss = criterion(outputs, targets_enc)
        total_loss += loss.item()

        preds = label_encoder.inverse_transform_torch(outputs)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mean_r2, per_trait = compute_r2(all_targets, all_preds)

    return total_loss / len(loader), mean_r2, per_trait


@torch.no_grad()
def predict_test(model, loader, label_encoder, device):
    model.eval()
    all_preds = []
    for batch in tqdm(loader, desc='Predict'):
        images = batch['image'].to(device)
        metadata = batch['metadata'].to(device)
        outputs = model(images, metadata)
        preds = label_encoder.inverse_transform_torch(outputs)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    seed_everything(CFG.SEED)
    print(f"Device: {CFG.DEVICE}")

    # 1. Preprocess data
    print("\n" + "="*60)
    print("STEP 1: Data Preprocessing")
    print("="*60)
    train_df, test_df, ancillary_cols, label_encoder, scaler = preprocess_data(CFG)

    # Adjust TABULAR_DIM based on actual columns
    actual_tab_dim = len(ancillary_cols)
    CFG.TABULAR_DIM = actual_tab_dim
    print(f"Actual tabular dimension: {actual_tab_dim}")

    # 2. Transforms
    train_transform = get_train_transforms(CFG.IMG_SIZE)
    val_transform = get_val_transforms(CFG.IMG_SIZE)

    # 3. Test dataset
    test_dataset = PlantDataset(
        test_df, ancillary_cols, label_encoder,
        transform=val_transform, is_test=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True
    )

    # Store all fold predictions
    all_test_preds = []
    all_oof_preds = np.zeros((len(train_df), CFG.NUM_TARGETS))

    # 4. Train per fold
    for fold in CFG.TRAIN_FOLDS:
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        trn_df = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_df = train_df[train_df['fold'] == fold].reset_index(drop=True)
        print(f"Train: {len(trn_df)}, Val: {len(val_df)}")

        train_dataset = PlantDataset(
            trn_df, ancillary_cols, label_encoder,
            transform=train_transform, is_test=False,
            use_response_var=CFG.USE_RESPONSE_VARIATION
        )
        val_dataset = PlantDataset(
            val_df, ancillary_cols, label_encoder,
            transform=val_transform, is_test=False
        )

        train_loader = DataLoader(
            train_dataset, batch_size=CFG.BATCH_SIZE,
            shuffle=True, num_workers=CFG.NUM_WORKERS,
            pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=CFG.BATCH_SIZE * 2,
            shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True
        )

        # Model
        model = PlantModel(CFG).to(CFG.DEVICE)

        # Optimizer with different LR for backbone vs head
        backbone_params = [p for n, p in model.named_parameters()
                          if 'backbone' in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                      if 'backbone' not in n and p.requires_grad]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': CFG.BACKBONE_LR},
            {'params': head_params, 'lr': CFG.LR},
        ], weight_decay=CFG.WEIGHT_DECAY)

        total_steps = len(train_loader) * CFG.EPOCHS
        scheduler = OneCycleLR(
            optimizer, max_lr=[CFG.BACKBONE_LR, CFG.LR],
            total_steps=total_steps, pct_start=0.1,
            anneal_strategy='cos', div_factor=25,
        )

        criterion = CombinedLoss(
            r2_w=CFG.R2_LOSS_WEIGHT,
            cos_w=CFG.COSINE_LOSS_WEIGHT,
            mse_w=CFG.MSE_LOSS_WEIGHT
        )

        best_r2 = -999
        best_model_path = os.path.join(CFG.OUTPUT_DIR, f'best_model_fold{fold}.pth')

        for epoch in range(CFG.EPOCHS):
            print(f"\nEpoch {epoch+1}/{CFG.EPOCHS}")

            train_loss, train_r2, train_per = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, label_encoder, CFG.DEVICE
            )
            val_loss, val_r2, val_per = validate(
                model, val_loader, criterion, label_encoder, CFG.DEVICE
            )

            print(f"  Train Loss: {train_loss:.4f} | Train R2: {train_r2:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Val   R2: {val_r2:.4f}")
            print(f"  Per-trait R2: {', '.join(f'{r:.3f}' for r in val_per)}")

            if val_r2 > best_r2:
                best_r2 = val_r2
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved! R2={best_r2:.4f}")

        # Load best and predict
        print(f"\nBest Val R2 for Fold {fold}: {best_r2:.4f}")
        model.load_state_dict(torch.load(best_model_path))

        # OOF predictions
        val_indices = train_df[train_df['fold'] == fold].index.values
        val_preds_dataset = PlantDataset(
            train_df[train_df['fold'] == fold].reset_index(drop=True),
            ancillary_cols, label_encoder, transform=val_transform
        )
        val_preds_loader = DataLoader(
            val_preds_dataset, batch_size=CFG.BATCH_SIZE * 2,
            shuffle=False, num_workers=CFG.NUM_WORKERS
        )
        oof_preds = predict_test(model, val_preds_loader, label_encoder, CFG.DEVICE)
        all_oof_preds[val_indices] = oof_preds

        # Test predictions
        test_preds = predict_test(model, test_loader, label_encoder, CFG.DEVICE)
        all_test_preds.append(test_preds)

        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Evaluate OOF
    print(f"\n{'='*60}")
    print("OOF EVALUATION")
    print(f"{'='*60}")
    used_indices = train_df[train_df['fold'].isin(CFG.TRAIN_FOLDS)].index.values
    if len(used_indices) > 0:
        oof_r2, oof_per = compute_r2(
            train_df.loc[used_indices, TRAIT_COLS].values,
            all_oof_preds[used_indices]
        )
        print(f"OOF Mean R2: {oof_r2:.4f}")
        for i, col in enumerate(TRAIT_COLS):
            print(f"  {col}: R2 = {oof_per[i]:.4f}")

    # 6. Create submission
    print(f"\n{'='*60}")
    print("CREATING SUBMISSION")
    print(f"{'='*60}")
    test_preds_mean = np.mean(all_test_preds, axis=0)

    submission = pd.read_csv(
        os.path.join(CFG.COMPETITION_DIR, 'sample_submission.csv')
    )
    for i, col in enumerate(TRAIT_COLS):
        # Clip predictions to reasonable ranges
        submission[col] = np.clip(test_preds_mean[:, i], 1e-6, None)

    submission.to_csv(os.path.join(CFG.OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"Submission saved! Shape: {submission.shape}")
    print(submission.head())


if __name__ == '__main__':
    main()
