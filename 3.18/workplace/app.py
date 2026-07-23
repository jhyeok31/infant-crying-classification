import os
import uuid
import numpy as np
import torch
import torch.nn as nn
import librosa
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────
# 1. 모델 구조 정의 (train_fyp.py와 완전히 동일해야 함)
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
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(1,   32,  pool=(2, 2)),   
            ConvBlock(32,  64,  pool=(2, 2)),   
            ConvBlock(64,  128, pool=(2, 2)),   
            ConvBlock(128, 256, pool=(1, 2)),   
        )
        gru_input_size = 256 * 10
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),   
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
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
# 2. 서버 및 모델 설정
# ─────────────────────────────────────────────
app = Flask(__name__)

# 학습 설정과 동일한 파라미터
SAMPLE_RATE = 16000
DURATION = 7          # seconds
TARGET_SAMPLES = SAMPLE_RATE * DURATION

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 20
FMAX = 8000

# 모델 로드 (서버 구동 시 1회)
MODEL_PATH = "output/fyp_model.pth" # run_train.sh가 저장한 폴더 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 서버 디바이스: {device}")

model = None
CLASS_NAMES = {}

try:
    print(f"⏳ 모델 가중치 로딩 중... ({MODEL_PATH})")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    CLASS_MAP = checkpoint['class_map']
    CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}
    NUM_CLASSES = len(CLASS_MAP)
    
    model = CRNNModel(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # 추론 모드로 변경 (dropout 등 비활성화)
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 오류: 모델 가중치를 불러올 수 없습니다. 경로를 확인하세요. ({e})")

# ─────────────────────────────────────────────
# 3. 오디오 전처리 함수 (Mel Spectrogram)
# ─────────────────────────────────────────────
def process_audio(file_path):
    try:
        # 1. 오디오 로드 및 리샘플링
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # 2. 길이 맞추기 (학습할 때와 동일하게 7초로 고정)
        if len(y) >= TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        else:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")

        # 3. Mel Spectrogram 변환
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX
        )
        mel = librosa.power_to_db(mel, ref=np.max)

        # 4. 정규화: [0, 1]
        mel_min, mel_max = mel.min(), mel.max()
        if mel_max - mel_min > 1e-6:
            mel = (mel - mel_min) / (mel_max - mel_min)

        # 5. Tensor 변환 (Batch=1, Channel=1, Mels=80, Time)
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0) 
        return tensor.to(device)

    except Exception as e:
        print(f"❌ 전처리 에러: {e}")
        return None

# ─────────────────────────────────────────────
# 4. 웹 서버 라우트
# ─────────────────────────────────────────────
@app.route('/')
def home():
    # templates 폴더 안에 index.html이 있어야 합니다.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '전송된 파일이 없습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    # 동시 접속 꼬임 방지를 위한 고유 임시 파일명 생성
    ext = file.filename.split('.')[-1]
    if ext not in ['wav', 'webm', 'ogg', 'mp4', '3gp']:
        ext = 'webm'  
        
    temp_filename = f"temp_{uuid.uuid4().hex}.{ext}"
    
    try:
        file.save(temp_filename)
        print(f"\n📥 파일 수신: {temp_filename}")

        if model is None:
            return jsonify({'error': '서버에 AI 분류 모델이 준비되지 않았습니다.'}), 500

        # 전처리
        input_tensor = process_audio(temp_filename)
        if input_tensor is None:
            return jsonify({'error': '오디오 분석 실패 (무음이거나 지원하지 않는 포맷입니다).'}), 500

        # 추론
        with torch.no_grad():
            outputs = model(input_tensor)
            # Softmax를 적용하여 확률 값으로 변환
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 결과 추출
        max_idx = torch.argmax(probabilities).item()
        confidence = probabilities[max_idx].item()
        result_label = CLASS_NAMES[max_idx]

        print(f"💡 예측 결과: {result_label} ({confidence*100:.1f}%)")

        return jsonify({
            'result': result_label,
            'confidence': confidence,
            'scores': probabilities.cpu().numpy().tolist()
        })

    except Exception as e:
        print(f"❌ 서버 내부 에러: {e}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500
        
    finally:
        # 처리 완료 후 임시 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    print("🚀 Flask 서버를 시작합니다...")
    # 윈도우 환경 테스트 시
    app.run(host='0.0.0.0', port=5000, debug=True)