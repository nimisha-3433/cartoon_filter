# CartoonCam - simple version

Real-time cartoon filter for your webcam on an NVIDIA GPU, usable as a camera in
Google Meet, plus a CPU vs CUDA vs cuDNN performance comparison.

## Files
| File | What it is |
|---|---|
| `cartoon.cu` | the live cartoon camera (CUDA): webcam -> GPU filter -> window + virtual camera |
| `benchmark.cu` | the same filter written for CPU, CUDA and cuDNN, timed and compared |
| `setup_vcam.sh` | creates the virtual camera "CartoonCam" (/dev/video10) |
| `run.sh` | one script to build and run everything |

## Needs (once)
```bash
sudo apt install build-essential libopencv-dev v4l2loopback-dkms v4l-utils
sudo apt install cudnn9-cuda-13        # for the benchmark (use cudnn9-cuda-12 on CUDA 12)
```
plus the NVIDIA driver and CUDA toolkit (`nvcc --version` must work).

## Use
```bash
./run.sh build          # once
./run.sh camera         # cartoon window
./run.sh meet           # then pick "CartoonCam" in Google Meet (Chrome)
./run.sh benchmark      # performance comparison
```
Keys in the cartoon window: `+ -` colours, `] [` outlines, `. ,` smoothing, `h` hide text, `q` quit.
To change the starting look, edit `struct Settings` at the top of `cartoon.cu` and run `./run.sh build`.

## Benchmark results
`./run.sh benchmark` prints a table and saves to `results/`:
- `table.txt` - time per frame at 480p / 720p / 1080p, speed-up, correctness
- `results.csv` - the same numbers for a spreadsheet
- `chart.png` - bar chart of the times
- `outputs.png` - input, CPU, CUDA and cuDNN results side by side

The benchmark filter is Gaussian blur -> Sobel edges -> 4 colours + outlines.
It uses a Gaussian blur instead of the camera's bilateral filter because cuDNN can only
run fixed-weight convolutions; the bilateral filter changes its weights for every pixel.
