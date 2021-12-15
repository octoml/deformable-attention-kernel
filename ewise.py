import torch
import torch.nn.functional as F
import tvm
from tvm.script import tir as T
import numpy as np
import timeit
import argparse


@T.prim_func
def ewise_add(
    lhs: T.handle,
    rhs: T.handle,
    output: T.handle,
) -> None:
    n = T.var("int32")
    lhs_ = T.match_buffer(lhs, [n], dtype="float32")
    rhs_ = T.match_buffer(rhs, [n], dtype="float32")
    output_ = T.match_buffer(output, [n], dtype="float32")

    for i in range(n):
        output_[i] = lhs_[i] + rhs_[i]

# Compile the kernel into an executable module.
ewise_add_compiled = tvm.build(ewise_add, target="llvm", name="ewise_add")

# Generate random numpy inputs.
x = np.random.rand(1000).astype("float32")
out = np.random.rand(1000).astype("float32")

tvm_x = tvm.nd.array(x)
tvm_out = tvm.nd.array(out)

ewise_add_compiled(tvm_x, tvm_x, tvm_out)
np.testing.assert_allclose(np.add(x, x), tvm_out.numpy())

# Generate random PyTorch inputs.
x = torch.randn(1000, dtype=torch.float32)
out = torch.randn(1000, dtype=torch.float32)

tvm_x = tvm.nd.array(x)
tvm_out = tvm.nd.array(out)
ewise_add_compiled(tvm_x, tvm_x, tvm_out)
assert torch.allclose(x + x, torch.from_numpy(tvm_out.numpy()))
