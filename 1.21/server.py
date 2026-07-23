import os
import numpy as np
import torch
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. 모델 & 엔진 로딩 (서버 켤 때 한 번만 함) ---
print("⏳ Wav2Vec2(특징 추출기) 로딩 중...")
device = 'cpu' # 서버는 보통 CPU로 돌림 (GPU 있으면 'cuda')
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)

print("⏳ Keras(분류기) 로딩 중...")
try:
    classifier_model = load_model("cry_classifier_model.h5")
    print("✅ 모든 모델 로드 완료!")
except:
    print("❌ 오류: cry_classifier_model.h5 파일이 없어요! 같은 폴더에 넣어주세요.")

LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# --- 2. 오디오 처리 함수 ---
def extract_features_from_audio(file_path):
    # (1) 오디오 로드 (16000Hz, 1초)
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=1.0)
    except Exception as e:
        return None
    
    # (2) 길이 맞추기 (Padding or Truncating)
    target_len = 16000
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # [Debug] 오디오 볼륨 체크
    max_amp = np.max(np.abs(audio))
    print(f"📊 Audio Max Amplitude: {max_amp:.4f} (너무 낮으면 무음)")

    # (3) Wav2Vec2에 넣기 위해 차원 변경
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # (4) 특징 추출 (PyTorch)
    with torch.no_grad():
        outputs = feature_extractor(input_values)
        embeddings = outputs.last_hidden_state.cpu().numpy()
    
    # (5) 평균 내기 (768차원 벡터 생성)
    avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
    
    # (6) Keras 모델에 넣기 위해 차원 확장 (1, 768, 1)
    # FCN 모델은 (Batch, Time, Channel) 형태를 원함
    final_input = avg_embeddings.reshape(1, 768, 1)
    
    return final_input

# --- 3. 웹 서버 라우트 ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '파일 없음'}), 400
    
    file = request.files['file']
    filename = "temp_audio.wav"
    file.save(filename) # 임시 저장

    try:
        # 1. 특징 추출 (Wav2Vec2)
        input_data = extract_features_from_audio(filename)
        
        if input_data is None:
            return jsonify({'error': '오디오 변환 실패'}), 500

        # 2. 분류 (Keras FCN)
        prediction = classifier_model.predict(input_data)
        
        # 3. 결과 정리
        max_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][max_idx])
        result_label = LABELS[max_idx]

        return jsonify({
            'result': result_label,
            'confidence': confidence,
            'scores': prediction[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename) # 청소

if __name__ == '__main__':
    # 형 컴퓨터에서 돌릴 때
    app.run(host='0.0.0.0', port=5000)