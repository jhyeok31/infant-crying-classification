"""
Baby Cry Classification - Improved CRNN Training Script (Mel Spectrogram Only)
=====================================================================
기존 train.py의 CRNN 아키텍처에 착안하여(Squeeze-and-Excitation 블록 추가 및 Temporal Average Pooling 도입)
정확도를 개선한 train_CRNN.py 입니다. 입력으로 멜 스펙트로그램만 사용합니다.

클래스 통합 + 균등 660개 샘플링:
  belly_pain ← belly_pain(330) + scared(330)
  burping    ← burping(660)
  discomfort ← discomfort(220) + cold_hot(220) + lonely(220)
  hungry     ← hungry(660)
  tired      ← tired(660)
"""

import os
import re
import json
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


# ── 클래스 통합 + 샘플링 설정 ──
SAMPLING_CONFIG = [
    ('belly_pain', 'belly_pain', 330),
    ('scared',     'belly_pain', 330),
    ('burping',    'burping',    660),
    ('cold_hot',   'discomfort', 220),
    ('discomfort', 'discomfort', 220),
    ('lonely',     'discomfort', 220),
    ('hungry',     'hungry',     660),
    ('tired',      'tired',      660),
]

NEW_CLASSES = sorted(set(cfg[1] for cfg in SAMPLING_CONFIG))
NEW_CLASS_TO_IDX = {name: idx for idx, name in enumerate(NEW_CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(description='Baby Cry Improved CRNN Training (5 Classes)')
    parser.add_argument('--data_dir', type=str, default='./all_preprocessing_files')
    parser.add_argument('--output_dir', type=str, default='./models_crnn')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# =============================================================================
# 1. 데이터 로드 + 균등 샘플링
# =============================================================================

def extract_group_key(filename):
    name = filename.replace('.npy', '')
    name = re.sub(r'_aug_\d+$', '', name)
    return name


def pad_to_length(data, target_length, axis=1):
    if data.shape[axis] >= target_length:
        return data[:, :target_length] if axis == 1 else data
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, target_length - data.shape[axis])
    return np.pad(data, pad_width, mode='constant', constant_values=0)


def load_data_balanced(data_dir, seed=42, target_len=299):
    np.random.seed(seed)
    mel_base = os.path.join(data_dir, 'FEATURES_MEL_SPECTROGRAMS_DATA')

    print("=" * 60)
    print("데이터 로드 (균등 660개, Mel Spectrogram Only)")
    print("=" * 60)

    all_mel, all_labels, all_groups = [], [], []
    n_padded = 0

    for old_cls, new_cls, target_count in SAMPLING_CONFIG:
        mel_cls_dir = os.path.join(mel_base, old_cls)
        new_idx = NEW_CLASS_TO_IDX[new_cls]

        files = sorted(os.listdir(mel_cls_dir))
        orig_files = [f for f in files if '_aug_' not in f]
        aug_files = [f for f in files if '_aug_' in f]

        selected = list(orig_files)
        need = target_count - len(selected)
        if need > 0:
            if need <= len(aug_files):
                sampled_aug = list(np.random.choice(aug_files, size=need, replace=False))
            else:
                sampled_aug = list(aug_files)
            selected.extend(sampled_aug)
        elif need < 0:
            selected = list(np.random.choice(orig_files, size=target_count, replace=False))

        for fname in selected:
            mel_data = np.load(os.path.join(mel_cls_dir, fname))
            if mel_data.shape[1] != target_len:
                mel_data = pad_to_length(mel_data, target_len, axis=1)
                n_padded += 1
            group_key = f"{old_cls}/{extract_group_key(fname)}"
            all_mel.append(mel_data)
            all_labels.append(new_idx)
            all_groups.append(group_key)

        n_orig = sum(1 for f in selected if '_aug_' not in f)
        n_aug = len(selected) - n_orig
        print(f"  {old_cls:12s} → {new_cls:12s}: {len(selected):3d}개 "
              f"(원본:{n_orig}, 증강:{n_aug})")

    all_mel = np.array(all_mel, dtype=np.float32)
    all_labels = np.array(all_labels)
    all_groups = np.array(all_groups)

    print(f"\n  전체: {len(all_labels)}개, 패딩: {n_padded}개")
    print(f"  Mel shape: {all_mel.shape}")
    for idx, name in enumerate(NEW_CLASSES):
        mask = all_labels == idx
        print(f"    [{idx}] {name:12s}: {mask.sum():4d}개")
    print()

    return all_mel, all_labels, all_groups


# =============================================================================
# 2. Group-Aware Split
# =============================================================================

def group_aware_split(mel, labels, groups, test_ratio=0.2, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    print("=" * 60)
    print(f"Group-Aware 분할 (test={test_ratio}, val={val_ratio})")
    print("=" * 60)

    for cls_idx in range(len(NEW_CLASSES)):
        cls_indices = np.where(labels == cls_idx)[0]
        cls_groups = groups[cls_indices]
        unique_groups = list(set(cls_groups))
        np.random.shuffle(unique_groups)

        n_test = max(3, int(len(unique_groups) * test_ratio))
        remaining = unique_groups[n_test:]
        n_val = max(2, int(len(remaining) * val_ratio))

        test_grps = set(unique_groups[:n_test])
        val_grps = set(remaining[:n_val])

        for idx in cls_indices:
            g = groups[idx]
            if g in test_grps:
                test_mask[idx] = True
            elif g in val_grps:
                val_mask[idx] = True
            else:
                train_mask[idx] = True

        print(f"  [{cls_idx}] {NEW_CLASSES[cls_idx]:12s}: "
              f"학습={train_mask[cls_indices].sum():4d} "
              f"검증={val_mask[cls_indices].sum():4d} "
              f"테스트={test_mask[cls_indices].sum():4d}")

    t_g, v_g, te_g = set(groups[train_mask]), set(groups[val_mask]), set(groups[test_mask])
    leak = len(t_g & te_g) + len(t_g & v_g) + len(v_g & te_g)
    print(f"\n  학습:{train_mask.sum()}, 검증:{val_mask.sum()}, 테스트:{test_mask.sum()}")
    print(f"  데이터 누수: {'없음' if leak == 0 else f'{leak}건!'}\n")

    return (mel[train_mask], mel[val_mask], mel[test_mask],
            labels[train_mask], labels[val_mask], labels[test_mask])


# =============================================================================
# 3. PyTorch Dataset
# =============================================================================

class MelDataset(Dataset):
    def __init__(self, mel_data, labels, augment=False):
        self.mel = torch.FloatTensor(mel_data)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mel[idx]

        if self.augment:
            # 시간/주파수 마스킹
            if torch.rand(1).item() < 0.5:
                t = torch.randint(0, mel.shape[1], (1,)).item()
                t_len = torch.randint(1, min(20, mel.shape[1]//4), (1,)).item()
                mel[:, t:t+t_len] = 0

            if torch.rand(1).item() < 0.5:
                f = torch.randint(0, mel.shape[0], (1,)).item()
                f_len = torch.randint(1, min(10, mel.shape[0]//4), (1,)).item()
                mel[f:f+f_len, :] = 0

        mel = mel.unsqueeze(0)  # (1, 128, 299)
        return mel, self.labels[idx]


# =============================================================================
# 4. 개선된 CRNN 모델
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)     # SE Block 적용
        x = self.pool(x)
        x = self.dropout(x)
        return x

class BabyCryCRNN(nn.Module):
    """
    CNN (with SE blocks) → BiLSTM → Classifier
    train.py의 아이디어를 도입해 정확도를 개선한 구조입니다.
    """
    def __init__(self, num_classes=5):
        super(BabyCryCRNN, self).__init__()

        # CNN feature extractor
        self.features = nn.Sequential(
            CNNBlock(1, 64),
            CNNBlock(64, 128),
            CNNBlock(128, 256)
        )

        # CNN 출력 크기 계산: (128, 299) → MaxPool2d ×3 → (16, 37)
        self.lstm_input_size = 256 * 16

        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 = 256 * 2 (bidirectional)
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        x = self.features(x)  # (batch, 256, 16, 37)

        # Reshape: 시간축을 sequence로
        x = x.permute(0, 3, 1, 2)  # (batch, time=37, channels=256, freq=16)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, 37, 256*16)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, 37, 512)
        
        # Temporal Average Pooling: 단순히 마지막 time step을 취하는 대신, 
        # sequence 전체를 평균내어 더 강건한 feature 도출
        x = lstm_out.mean(dim=1) 

        # Classifier
        x = self.classifier(x)
        return x


# =============================================================================
# 5. 학습 루프
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, 100. * correct / total, all_preds, all_labels


# =============================================================================
# 6. 시각화
# =============================================================================

def plot_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label='Train Loss', lw=2)
    axes[0].plot(val_losses, label='Val Loss', lw=2)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Acc', lw=2)
    axes[1].plot(val_accs, label='Val Acc', lw=2)
    axes[1].set_title('Accuracy (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()


def plot_cm(y_true, y_pred, names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=names, yticklabels=names, ax=ax, linewidths=0.5)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


# =============================================================================
# 7. 메인
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch {torch.__version__}, Device: {device}\n")

    # ─── 1. 데이터 로드 ───
    mel, labels, groups = load_data_balanced(args.data_dir, seed=args.seed)

    # ─── 2. Group-Aware Split ───
    (mel_tr, mel_v, mel_te,
     y_tr, y_v, y_te) = group_aware_split(
        mel, labels, groups,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # ─── 3. Z-score 정규화 ───
    m_mean, m_std = mel_tr.mean(), mel_tr.std()
    mel_tr = (mel_tr - m_mean) / (m_std + 1e-8)
    mel_v = (mel_v - m_mean) / (m_std + 1e-8)
    mel_te = (mel_te - m_mean) / (m_std + 1e-8)
    print(f"정규화: mean={m_mean:.2f}, std={m_std:.2f}")
    print(f"입력: 학습={mel_tr.shape[0]}, 검증={mel_v.shape[0]}, 테스트={mel_te.shape[0]}\n")

    # ─── 4. DataLoader ───
    train_ds = MelDataset(mel_tr, y_tr, augment=True)
    val_ds = MelDataset(mel_v, y_v, augment=False)
    test_ds = MelDataset(mel_te, y_te, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ─── 5. Class Weights ───
    cw = compute_class_weight('balanced', classes=np.arange(len(NEW_CLASSES)), y=y_tr)
    weights = torch.FloatTensor(cw).to(device)
    print("클래스 가중치:")
    for i, name in enumerate(NEW_CLASSES):
        print(f"  {name:12s}: {cw[i]:.3f}")
    print()

    # ─── 6. 모델 ───
    model = BabyCryCRNN(num_classes=len(NEW_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6, verbose=True
    )

    # 모델 파라미터 수
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터: {total_params:,}개\n")

    # ─── 7. 학습 ───
    print("=" * 60)
    print(f"학습 시작 (최대 {args.epochs} 에폭, patience={args.patience})")
    print("=" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    model_path = os.path.join(args.output_dir, 'best_model.pth')

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
              f"Train Loss={tr_loss:.4f} Acc={tr_acc:.1f}% | "
              f"Val Loss={vl_loss:.4f} Acc={vl_acc:.1f}% | "
              f"LR={current_lr:.2e}", end="")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(" ★ 저장")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print(f"\n  Early Stopping at epoch {epoch+1}")
            break

    # ─── 8. Best 모델 로드 후 테스트 ───
    print("\n" + "=" * 60)
    print("테스트 세트 평가 (Best 모델)")
    print("=" * 60)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"  테스트 손실    : {test_loss:.4f}")
    print(f"  테스트 정확도  : {test_acc:.2f}%")

    report = classification_report(y_true, y_pred, target_names=NEW_CLASSES, zero_division=0)
    print(f"\n분류 리포트:\n{report}")

    # ─── 9. 결과 저장 ───
    hist_path = os.path.join(args.output_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_curves(history['train_loss'], history['val_loss'],
                history['train_acc'], history['val_acc'], args.output_dir)
    plot_cm(y_true, y_pred, NEW_CLASSES, args.output_dir)

    rpt_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(f"테스트 손실    : {test_loss:.4f}\n")
        f.write(f"테스트 정확도  : {test_acc:.2f}%\n\n")
        f.write(f"모델: CRNN (CNN + SEBlock + BiLSTM + Temporal Avg Pool)\n")
        f.write(f"입력: Mel Spectrogram Only (128x299)\n\n")
        f.write(f"분류 리포트:\n{report}")

    le_path = os.path.join(args.output_dir, 'label_encoder_5class.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump({'classes': NEW_CLASSES, 'sampling_config': SAMPLING_CONFIG}, f)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"  모델          : {model_path}")
    print(f"  히스토리      : {hist_path}")
    print(f"  리포트        : {rpt_path}")
    print(f"  테스트 정확도 : {test_acc:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
