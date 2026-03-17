# Omni-PVM

## Overview

The system works in two steps:

1. **Build Visual Map** — Extract multi-scale ORB features from reference images with known poses (provided as a CSV file). The map can be cached to a binary file for reuse.
2. **Run Matching** — Match query images against the visual map using the Viterbi algorithm, then optionally refine the estimated pose with a PnP solver.

```
CSV (poses) + Images → Visual Map → ORB Features (3 scales) → Binary Cache
                                                                     ↓
Query CSV + Images → Query Loader → Viterbi Matcher → Results CSV
```

## Prerequisites

### System Requirements

- Ubuntu 20.04 / 22.04
- C++17 compiler (GCC 9+)
- CMake 3.10+
- ROS Noetic or ROS2 (catkin build system)

### Required Libraries

| Library | Version | Install |
|---------|---------|---------|
| ROS | Noetic | [ros.org](http://wiki.ros.org/noetic/Installation) |
| OpenCV | 4.x | `sudo apt install libopencv-dev` |
| Eigen3 | 3.x | `sudo apt install libeigen3-dev` |
| PCL | 1.x | `sudo apt install libpcl-dev` |
| Boost | 1.71+ | `sudo apt install libboost-all-dev` |
| VTK | 9.2 | See below |
| G2O | latest | See below |

### Installing VTK 9.2

```bash
# Download and build from source, or install to /opt/vtk-9.2.2
# The CMakeLists.txt expects VTK at /opt/vtk-9.2.2
wget https://www.vtk.org/files/release/9.2/VTK-9.2.2.tar.gz
tar xf VTK-9.2.2.tar.gz && cd VTK-9.2.2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/vtk-9.2.2
make -j$(nproc)
sudo make install
```

### Installing G2O

```bash
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

## Build

```bash
cd ~/3DGS_match_ws
catkin_make
source devel/setup.bash
```

## Data Format

### Map / Query CSV

Each row represents one frame:

```
frame_id, x, y, z, qx, qy, qz, qw
0, 1.23, 4.56, 0.0, 0.0, 0.0, 0.0, 1.0
...
```

### Image Folder

Images should be named by their `frame_id`:

```
images/
├── 0.jpg
├── 1.jpg
└── ...
```

## Running

```bash
catkin_make
./devel/lib/3DGS_match/3DGS_match
```


