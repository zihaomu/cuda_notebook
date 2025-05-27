import torch

# set system environment variable to use the inductor backend
import os
# os.environ["TORCH_COMPILE_DEBUG"] = "1"
# os.environ["TORCH_LOGS"] = "dynamo,aot,+inductor"

torch._dynamo.config.verbose = True
def toy_example(x):
    y = x.sin()
    z = y.cos()
    
    return z

DEVICE = torch.device('cuda:0')
x = torch.randn(1000, device=DEVICE, requires_grad=True)

compile_f = torch.compile(toy_example, backend='inductor')

output = compile_f(x)