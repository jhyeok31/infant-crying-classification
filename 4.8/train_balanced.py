"""
FYP Dataset - 아기 울음소리 다중 클래스 CRNN 분류 모델 (자동 라벨 인식)

- 통합 .npy 파일 (mel_spectrograms.npy, labels_encoded.npy) 사용
- label_encoder.pkl을 사용하여 클래스 개수와 매핑을 자동으로 인식
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
import pickle  # Label Encoder 로드를 위해 추가됨
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 설정 (경로 확인 필수!)
# ─────────────────────────────────────────────
# 통합 데이터 파일 및 인코더 경로
MEL_PATH = r"../all_preprocessing_files/Data_Explorer/mel_spectrograms.npy"
LABEL_PATH = r"../all_preprocessing_files/Data_Explorer/labels_encoded.npy"
ENCODER_PATH = r"../all_preprocessing_files/Data_Explorer/label_encoder.pkl"  # ★ 추가됨

# 모델 저장 경로
MODEL_SAVE_PATH = "balanced_model.pth"

# Mel spectrogram parameters
N_MELS = 128

# Training hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 4

# ─────────────────────────────────────────────
# ★ 핵심 수정: 클래스 정보 자동 로드
# ─────────────────────────────────────────────
print("[INFO] Label Encoder를 로드하여 클래스 정보를 파악합니다...")
try:
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # label_encoder.classes_ 에는 알파벳 순으로 정렬된 클래스 이름들이 들어있음
    class_names_list = label_encoder.classes_
    CLASS_MAP = {name: idx for idx, name in enumerate(class_names_list)}
    CLASS_NAMES = {idx: name for idx, name in enumerate(class_names_list)}
    NUM_CLASSES = len(CLASS_MAP)
    
    print(f"[성공] 총 {NUM_CLASSES}개의 클래스가 자동 인식되었습니다.")
    print(f"       매핑 정보: {CLASS_MAP}")
except Exception as e:
    print(f"[에러] label_encoder.pkl 파일을 불러오는 데 실패했습니다: {e}")
    print("경로를 다시 확인해 주세요. 실행을 중단합니다.")
    exit()


# ─────────────────────────────────────────────
# Dataset (초고속 NPY 로더)
# ─────────────────────────────────────────────
class NpyDataset(Dataset):
    def __init__(self, mel_path, label_path):
        print(f"[INFO] 통합 텐서 데이터 로딩 중...")
        self.mels = np.load(mel_path)
        self.labels = np.load(label_path)
        print(f"[INFO] 로드 완료! Mels shape: {self.mels.shape}, Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel_tensor = torch.FloatTensor(self.mels[idx]).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_tensor, label_tensor


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
            nn.Dropout2d(0.3),
        )

    def forward(self, x):
        return self.net(x)

class CRNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 2)),
            ConvBlock(32,  64,  pool=(2, 2)),
            ConvBlock(64,  128, pool=(2, 2)),
            ConvBlock(128, 256, pool=(1, 2)),
        )

        # N_MELS=128 기준 GRU 입력 사이즈
        gru_input_size = 256 * 16

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.4,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),  # 자동 인식된 8개 클래스가 여기에 적용됨
        )

    def forward(self, x):
        b = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, x.size(1), -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# Training & Validation Loops
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Valid", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def evaluate_and_plot_cm(model, loader, device, class_names, save_path):
    import os
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[INFO] matplotlib, seaborn 패키지가 없어 설치를 시도합니다...")
        os.system('pip install matplotlib seaborn')
        import matplotlib.pyplot as plt
        import seaborn as sns
    from sklearn.metrics import confusion_matrix

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Confusion Matrix", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    labels_list = [class_names[i] for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_list, 
                yticklabels=labels_list)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[성공] 혼동행렬 이미지가 성공적으로 저장되었습니다: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 초고속 NpyDataset 로드
    dataset = NpyDataset(MEL_PATH, LABEL_PATH)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    print(f"--- 데이터 분할 ---")
    print(f"Train Dataset Size: {train_size}")
    print(f"Valid Dataset Size: {val_size}")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = CRNNModel(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델 파라미터 수: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training
    best_val_acc = 0.0
    print("=" * 60)
    print("학습 시작")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        print(
            f"[Epoch {epoch:3d}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "class_map": CLASS_MAP,
            }, MODEL_SAVE_PATH)
            print(f"  ★ 최고 성능 모델 저장: {MODEL_SAVE_PATH} (Val Acc: {val_acc:.2f}%)")

    print("\n" + "=" * 60)
    print(f"학습 완료! 최고 Valid Accuracy: {best_val_acc:.2f}%")
    print(f"저장된 모델: {MODEL_SAVE_PATH}")

    # Validation
    print("\n" + "=" * 60)
    print("최종 최고 성능 모델 로드 및 재검증 시작")
    print("=" * 60)

    if os.path.exists(MODEL_SAVE_PATH):
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"최고 성능 모델 로드 완료 (Epoch: {checkpoint.get('epoch', 'N/A')})")

    val_loss, val_acc = validate(model, val_loader, criterion, device)
    print(f"\n[최종 최고 성능 검증 결과]")
    print(f"Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.2f}%")
    print("=" * 60)
    
    cm_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH) if os.path.dirname(MODEL_SAVE_PATH) else ".", "confusion_matrix.png")
    evaluate_and_plot_cm(model, val_loader, device, CLASS_NAMES, cm_path)

if __name__ == "__main__":
    main()