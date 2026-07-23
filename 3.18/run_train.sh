#!/bin/bash
set -e

echo "========================================"
echo "  FYP 아기 울음소리 분류 모델 학습"
echo "========================================"

DATASET_DIR="C:/Users/SPL_1/Documents/3.18/FYP dataset"
SCRIPT_DIR="C:/Users/SPL_1/Documents/3.18"
IMAGE_NAME="cry-classifier"

# 이전 stale 컨테이너 정리
docker rm -f cry-train 2>/dev/null && echo "이전 컨테이너 제거됨" || true

# Docker 이미지 빌드
echo ""
echo "[1/2] Docker 이미지 빌드 중..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# GPU 사용 가능 여부 확인
if docker info 2>/dev/null | grep -q "Runtimes.*nvidia" || docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 true 2>/dev/null; then
    GPU_FLAG="--gpus all"
    echo ""
    echo "✅ GPU 감지됨 - CUDA 학습 모드"
else
    GPU_FLAG=""
    echo ""
    echo "⚠️  GPU 없음 - CPU 학습 모드 (느릴 수 있음)"
fi

# 출력 디렉토리 준비
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# 컨테이너 실행
echo ""
echo "[2/2] 학습 컨테이너 실행 중..."
echo "  - 로그: $SCRIPT_DIR/train_log.txt"
echo "  - 모델 저장: $OUTPUT_DIR/fyp_model.pth"
echo ""

docker run \
    --name cry-train \
    $GPU_FLAG \
    -v "$DATASET_DIR:/workspace/FYP dataset:ro" \
    -v "$OUTPUT_DIR:/workspace/output" \
    -e MODEL_SAVE_PATH=/workspace/output/fyp_model.pth \
    --rm \
    "$IMAGE_NAME" 2>&1 | tee "$SCRIPT_DIR/train_log.txt"

echo ""
echo "학습 완료! 결과: $OUTPUT_DIR/fyp_model.pth"
