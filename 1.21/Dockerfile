# 1. 베이스 이미지
FROM python:3.10-slim

# 2. 작업 폴더 설정
WORKDIR /app

# 3. 필수 시스템 프로그램 설치
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4-1. [수정됨] 일반 라이브러리 먼저 설치 (기본 PyPI에서 다운)
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pandas \
    librosa \
    transformers \
    tensorflow-cpu

# 4-2. [수정됨] 파이토치만 따로 설치 (CPU 전용 사이트에서 다운)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. 파일 복사
COPY server.py .
COPY cry_classifier_model.h5 .

# 6. 포트 열기
EXPOSE 5000

# 7. 실행
CMD ["python", "server.py"]