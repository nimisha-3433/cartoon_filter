# CartoonCam - simple version

Real-time cartoon filter for your webcam on an NVIDIA GPU, usable as a live filter in
Google Meet

## Files
| File | Description |

| `cartoon.cu` | the live cartoon camera |
| `benchmark.cu` | comparison file for CPU, CUDA and cuDNN|
| `setup_vcam.sh` | creates the virtual camera "CartoonCam" |
| `run.sh` | run file |

## Initial Setup
```bash
sudo apt install build-essential libopencv-dev v4l2loopback-dkms v4l-utils
sudo apt install cudnn9-cuda-13       

## Use
```bash
./run.sh build          
./run.sh camera         
./run.sh meet           
./run.sh benchmark      # performance comparison
```
Keys in the cartoon window: `+ -` colours, `] [` outlines, `. ,` smoothing, `h` hide text, `q` quit.

## Benchmark results
`./run.sh benchmark` prints a table and saves to `results/`:
- `table.txt` - time per frame at 480p / 720p / 1080p, speed-up, correctness
- `results.csv` - the same numbers for a spreadsheet
- `chart.png` - bar chart of the times
- `outputs.png` - input, CPU, CUDA and cuDNN results side by side
