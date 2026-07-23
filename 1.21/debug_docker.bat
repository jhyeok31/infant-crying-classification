docker build -t baby-server .
docker run -it --rm -p 5000:5000 -v "%cd%:/app" baby-server