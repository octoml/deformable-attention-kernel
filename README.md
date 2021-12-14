# Deformable Attention Kernel Implemented in TVMScript

This is a comparison of a deformabel attention kernel written in TVM's
TVMScript to the pytorch and CUDA kernels in
https://github.com/fundamentalvision/Deformable-DETR.

## Setup

You need to install:

- TVM: https://github.com/apache/tvm
- PyTorch
- The deformable attention cuda kernels from
  https://github.com/fundamentalvision/Deformable-DETR. Run
  `python3 setup.py build install` in the `models/ops` directory.

## Running

`python3 kernel.py "llvm -mcpu=skylake"` will run the cpu kernels targeted for
an Intel Skylake cpu. YOu can determine what cpu you have by running `llc --version`
and looking for "Host CPU".

`python3 kernel.py cuda` will run the gpu kernels.

## Example Output

```
# On AMD 5900X cpu
> python3 kernel.py "llvm -mcpu=znver3"
BATCH SIZE 1
TVMScript CPU: 0.33ms
PyTorch: 35.48ms
Speedup: 106.78x

BATCH SIZE 4
TVMScript CPU: 4.08ms
PyTorch: 128.69ms
Speedup: 31.53x

BATCH SIZE 8
TVMScript CPU: 11.30ms
PyTorch: 232.38ms
Speedup: 20.56x

# on NVIDIA Geforce RTX 3070
> python3 kernel.py "nvidia/geforce-rtx-3070"
BATCH SIZE 1
TVMScript GPU: 0.085ms
Handwritten CUDA: 0.127ms
Speedup: 1.49x

BATCH SIZE 4
TVMScript GPU: 0.333ms
Handwritten CUDA: 0.444ms
Speedup: 1.33x

BATCH SIZE 8
TVMScript GPU: 0.669ms
Handwritten CUDA: 0.870ms
Speedup: 1.30x
```
