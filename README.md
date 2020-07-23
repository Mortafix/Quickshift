# Quickshift
A working implementation of Quickshift algorithm in CUDA, GPU-compatible.

## Usage
`Quickshift <input> <mode> <sigma> <dist> <texture>`
* `input`: input image with a compatible formats `JPG`, `PNG`, `PNM` or `BMP`.
* `mode`: computing mode, `GPU` or `CPU`. (deafult `GPU`)
* `sigma`: approximation value to control the scale of the local density. (integer)
* `dist`: max distance level in the hierarchical segmentation that is produced. (integer)
* `texture`: adding texture memory support, only for `GPU` mode, `y` or `n`. (default `n`)

## Examples
`Quickshift peppers.jpg gpu 5 8 y`  
GPU algorithm with texture memory, JPG image as input, density 5 and distance 8.  
  
`Quickshift lena.png cpu 3 7`  
CPU algorithm, PNG image as input, density 3 and distance 7.  
