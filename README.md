# Quickshift
A working implementation of Quickshift algorithm in CUDA, GPU-compatible.

## Usage
`Quickshift <input> <mode> <sigma> <dist>`
* `input`: input image with a compatible formats `JPG`, `PNG`, `PNM` or `BMP`.
* `mode`: computing mode, `GPU` or `CPU`.
* `sigma`: approximation value to control the scale of the local density.
* `dist`: max distance level in the hierarchical segmentation that is produced.

## Examples
`Quickshift peppers.jpg gpu 5 8`  
GPU algorithm with a JPG image, density 5 and max distance 8.  
  
`Quickshift lena.png cpu 3 0.5`  
CPU algorithm with a PNG image, density 3 and max distance 0.5.  
