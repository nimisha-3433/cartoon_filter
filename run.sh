#!/usr/bin/env bash
# One script for everything.
#   ./run.sh build        compile both programs (do this once)
#   ./run.sh camera       cartoon window
#   ./run.sh meet         cartoon window + CartoonCam for Google Meet / Zoom
#   ./run.sh benchmark    CPU vs CUDA vs cuDNN comparison -> results/
set -e
cd "$(dirname "$0")"
OPENCV="$(pkg-config --cflags --libs opencv4)"

case "${1:-}" in
  build)
    echo "building cartoon (live camera)..."
    nvcc -O2 -arch=native cartoon.cu -o cartoon $OPENCV
    echo "building benchmark (CPU vs CUDA vs cuDNN)..."
    if nvcc -O2 -arch=native benchmark.cu -o benchmark $OPENCV -lcudnn; then
      echo "done: ./run.sh camera | meet | benchmark"
    else
      echo "benchmark needs cuDNN:  sudo apt install cudnn9-cuda-13   (then ./run.sh build again)"
    fi
    ;;
  camera)
    ./cartoon
    ;;
  meet)
    [[ -e /dev/video10 ]] || ./setup_vcam.sh
    echo "now open Google Meet in Chrome and choose the camera 'CartoonCam'"
    ./cartoon /dev/video10
    ;;
  benchmark)
    ./benchmark ${2:+"$2"}
    ;;
  *)
    echo "usage: ./run.sh build | camera | meet | benchmark [image.jpg]"
    ;;
esac
