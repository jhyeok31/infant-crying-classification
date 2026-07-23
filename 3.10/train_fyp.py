import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DeepInfantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        self.label_map = {
            'bp': 0,  # belly pain
            'bu': 1,  # burping
            'dc': 2,  # discomfort
            'hu': 3,  # hungry
            'nc': 4,  # non-crying
            'ti': 5,  # tired
        }
        
        metadata_file = Path(data_dir).parent / 'metadata.csv'
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row['split'] == self.data_dir.name:  # 'train' or 'test'
                audio_path = self.data_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    self.labels.append(self.label_map[row['class_code']])
    
    def _load_dataset(self):
        for audio_file in self.data_dir.glob('*.*'):
            if audio_file.suffix in ['.wav', '.caf', '.3gp', '.ogg']:
                # Parse filename for label (assuming format stem-code.wav)
                label = audio_file.stem.split('-')[-1][:2]
                if label in self.label_map:
                    self.samples.append(str(audio_file))
                    self.labels.append(self.label_map[label])
    
    def _process_audio(self, audio_path):
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        if self.transform:
            # Random time shift
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[:len(waveform)]
            else:
                waveform = np.pad(waveform, (0, -shift))[(-shift):]
            
            # Random noise injection
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(waveform))
                waveform = waveform + noise
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generated mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=20,
            fmax=8000
        )
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]
        
        mel_spec = self._process_audio(audio_path)
        
        return mel_spec, label

class DeepInfantModel(nn.Module):
    def __init__(self, num_classes=6):
        super(DeepInfantModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Bi-directional LSTM for better temporal modeling
        # After 3 MaxPool2d(2) layers, the freq dimension 80 becomes 80/8 = 10
        # and the channel dimension is 256. So input_size = 256 * 10
        self.lstm = nn.LSTM(
            input_size=256 * 10,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for LSTM
        # x is currently (batch, channels=256, freq=10, time_steps)
        # We permute it so time is the 2nd dimension: (batch, time, channels, freq)
        x = x.permute(0, 3, 1, 2)
        # Then flatten the channels and freq features into one vector
        # Size after permute: (batch, time_steps, 256, 10)
        # We flatten the last two dims -> (batch, time_steps, 256 * 10)
        batch_size, time_steps, channels, freq = x.size()
        x = x.contiguous().view(batch_size, time_steps, channels * freq)  # (batch, time, features)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1) 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'deepinfant_fyp.pth')
            print(f'Saved Best Model with Val Acc: {best_val_acc:.2f}%')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_dataset = DeepInfantDataset('/app/processed_fyp_dataset/train', transform=False)
    val_dataset = DeepInfantDataset('/app/processed_fyp_dataset/test', transform=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("Error: Dataset empty. Run prepare_fyp_dataset.py first.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = DeepInfantModel(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, device=device)

if __name__ == '__main__':
    main()
