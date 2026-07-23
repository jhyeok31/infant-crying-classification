"""
FYP Dataset - 아기 울음소리 6클래스 CRNN 분류 모델
Classes: belly_pain, burping, discomfort, hungry, non-crying, tired

- 데이터 증강 없음 (이미 증강된 데이터셋 사용)
- 전체 데이터를 학습에 사용 (train/val 분리 없음)
- 목표: 훈련 정확도 최대화
"""

import os
import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
import os

# Docker 컨테이너 내부 마운트 경로 또는 로컬 경로 자동 선택
_container_path = Path("/workspace/FYP dataset")
_host_path     = Path("C:/Users/SPL_1/Documents/3.18/FYP dataset")
DATASET_DIR    = _container_path if _container_path.exists() else _host_path

# 모델 저장 경로
MODEL_SAVE_PATH = os.environ.get("MODEL_SAVE_PATH", "fyp_model.pth")

SAMPLE_RATE = 16000
DURATION = 7          # seconds
TARGET_SAMPLES = SAMPLE_RATE * DURATION

# Mel spectrogram parameters
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 20
FMAX = 8000

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# 클래스 매핑 (폴더명 → 라벨 인덱스)
CLASS_MAP = {
    "belly pain":   0,
    "burping":      1,
    "discomfort":   2,
    "hungry":       3,
    "non-crying":   4,
    "tired":        5,
}
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class FYPDataset(Dataset):
    """
    FYP dataset 로더.
    폴더 내 모든 .wav / .3gp 파일을 읽어
    Mel Spectrogram으로 변환하여 반환합니다.
    """
    EXTENSIONS = {".wav", ".3gp"}

    def __init__(self, dataset_dir: Path):
        self.samples = []   # (file_path, label)

        class_counts = {}
        for folder_name, label in CLASS_MAP.items():
            folder = dataset_dir / folder_name
            if not folder.exists():
                print(f"[WARNING] 폴더 없음: {folder}")
                continue

            files = [
                f for f in folder.iterdir()
                if f.suffix.lower() in self.EXTENSIONS
            ]
            class_counts[folder_name] = len(files)
            for f in files:
                self.samples.append((str(f), label))

        print("\n=== 클래스별 파일 수 ===")
        total = 0
        for name, cnt in class_counts.items():
            print(f"  {name:15s}: {cnt:5d}")
            total += cnt
        print(f"  {'TOTAL':15s}: {total:5d}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = self._load_mel(path)
        return mel, label

    def _load_mel(self, path: str) -> torch.Tensor:
        try:
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            print(f"[ERROR] 로드 실패: {path} -> {e}")
            y = np.zeros(TARGET_SAMPLES, dtype=np.float32)

        # 길이 맞추기 (pad / truncate)
        if len(y) >= TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        else:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        # 정규화: [0, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max - mel_min > 1e-6:
            mel = (mel - mel_min) / (mel_max - mel_min)

        return torch.FloatTensor(mel).unsqueeze(0)  # (1, n_mels, time)


# ─────────────────────────────────────────────
# Model: CRNN
# ─────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=(2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.2),
        )

    def forward(self, x):
        return self.net(x)


class CRNNModel(nn.Module):
    """
    CNN (특징 추출) + Bidirectional GRU (시계열 모델링) + FC (분류)
    입력: (batch, 1, 80, T)  -> Mel Spectrogram
    출력: (batch, NUM_CLASSES)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # CNN Encoder
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 2)),   # (1,80,T) -> (32,40,T//2)
            ConvBlock(32,  64,  pool=(2, 2)),   # -> (64,20,T//4)
            ConvBlock(64,  128, pool=(2, 2)),   # -> (128,10,T//8)
            ConvBlock(128, 256, pool=(1, 2)),   # -> (256,10,T//16)
        )

        # Frequency dim이 10이 됨 → GRU input_size = 256 * 10 = 2560
        gru_input_size = 256 * 10

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),   # 512 = 256 * 2 (bidirectional)
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        b = x.size(0)
        x = self.cnn(x)                 # (B, 256, F', T')
        # (B, C, F', T') -> (B, T', C*F')
        x = x.permute(0, 3, 1, 2)      # (B, T', C, F')
        x = x.reshape(b, x.size(1), -1) # (B, T', C*F')
        x, _ = self.gru(x)              # (B, T', 512)
        x = x[:, -1, :]                 # last time step (B, 512)
        x = self.classifier(x)          # (B, num_classes)
        return x


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset & DataLoader
    dataset = FYPDataset(DATASET_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model
    model = CRNNModel(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {total_params:,}")

    # Loss / Optimizer / Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training
    best_acc = 0.0
    print("=" * 60)
    print("학습 시작")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        scheduler.step()

        print(
            f"[Epoch {epoch:3d}/{NUM_EPOCHS}] "
            f"Loss: {loss:.4f}  |  Train Acc: {acc:.2f}%"
            f"  |  LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # 최고 정확도 모델 저장
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": acc,
                "loss": loss,
                "class_map": CLASS_MAP,
            }, MODEL_SAVE_PATH)
            print(f"  ★ 모델 저장: {MODEL_SAVE_PATH} (acc={acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"학습 완료! 최고 Train Accuracy: {best_acc:.2f}%")
    print(f"저장된 모델: {MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
