FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies for audio processing (librosa needs libsndfile)
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir librosa tqdm pandas scikit-learn
