# PyTorch + CUDA 베이스 이미지 (GPU 있으면 CUDA, 없으면 CPU fallback)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# 필요한 패키지 설치
RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile \
    tqdm \
    numpy \
    scikit-learn

# 학습 스크립트 복사
COPY train_fyp.py .

# 로그를 실시간으로 출력
ENV PYTHONUNBUFFERED=1

CMD ["python", "train_fyp.py"]
