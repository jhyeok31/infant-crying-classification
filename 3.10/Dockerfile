FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# 기본 시스템 패키지 업데이트 및 오디오 처리 라이브러리 추가
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 기본 실행 명령어 (스크립트 이름은 실행 시 지정 가능)
CMD ["python", "/app/train_fyp.py"]
