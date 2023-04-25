from dataset import Arithmetic
import torch

dset = Arithmetic(10)

for _ in range(10):
    exp, res, res_ = dset.generate_expression()

    print(exp)
    print(res)
    print(res_)
    print(dset.get_str(exp))
    print(dset.get_str(res))
    print("=======================")

x, y, y_ = dset.get_batch(10)
for _ in range(100):
    x,y, y_ = dset.get_batch(10)
print(x)
print(y)
print(y_)

print(x.shape)
print(dset.max_len)

