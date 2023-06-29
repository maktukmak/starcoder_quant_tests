import torch
import intel_extension_for_pytorch as ipex
import os
from torch.profiler import ProfilerActivity, profile
import time
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, HistogramObserver
from intel_extension_for_pytorch.quantization import prepare, convert
import torch.fx as fx
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.data import Dataset, DataLoader
from torch.profiler import ProfilerActivity, profile, record_function
from contextlib import nullcontext
import os
import random
os.environ['ONEDNN_VERBOSE'] = '0'
os.environ['DNNL_GRAPH_VERBOSE'] = '2'
os.environ['ONEDNN_GRAPH_VERBOSE'] = '1'

precision = 'int8' #bf16, fp32, int8
device = "cpu"  # cuda:0,  cpu

dim = 6144

print(device)


class Model(torch.nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        x = self.lin(x)
        x.split()
        return x
    

class RandomDataset(Dataset):
    def __init__(self, dim):
        self.dim = dim

    def __getitem__(self, index):
            
        #seq_len = random.randint(30, 100)
        seq_len = 40
        data = torch.rand(seq_len, self.dim)

        return data
    
    def __len__(self):
        return 1000
    
class RandomCollator(object):
    def __init__(self):
        pass
    def  __call__(self, batch_list):

        batch = torch.nn.utils.rnn.pad_sequence(batch_list, batch_first=True)
        return batch


block = Model(dim)


collat = RandomCollator()
dataset = RandomDataset(dim = dim)

block.eval()

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collat)
inp = next(iter(dataloader))

output = block(inp)

if precision == 'bf16' or precision == 'fp32':
    block = ipex.optimize(block,
                        auto_kernel_selection=False,
                        dtype=(torch.bfloat16 if precision == 'bf16' else torch.float32))
    with torch.no_grad():
        with torch.cpu.amp.autocast() if precision == 'bf16' else nullcontext():
            block = torch.jit.trace(block, example_inputs=inp, check_trace=False, strict=False)
            block = torch.jit.freeze(block)

elif precision == 'int8':

    recipes = {
        "smooth_quant": True,
        "smooth_quant_args": {
            "alpha": 0.5,
            "folding": False,
        },
    }
    op_type_dict = {
        "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        "linear": {
            "weight": {
                "dtype": ["int8"],
                "scheme": ["sym"],
                "granularity": ["per_channel"],
                "algorithm": ["minmax"],
            },
            "activation": {
                "dtype": ["uint8"],
                "scheme": ["asym"],
                "granularity": ["per_tensor"],
                "algorithm": ["minmax"],
            }}
    }

    conf = PostTrainingQuantConfig( backend="ipex",
                                    recipes = recipes,
                                    calibration_sampling_size = 16,
                                    op_type_dict=op_type_dict,) 

    block = quantization.fit(model=block,
                            conf=conf,
                            calib_dataloader=dataloader)


if True:
    wait = 5
    warmup = 5
    active = 10
    # Profile
    def trace_handler(p):
        output = p.key_averages().table(sort_by="cpu_time_total")#, row_limit=20)
        print(output)

    with profile(activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active),
            on_trace_ready=trace_handler,
            profile_memory=False, 
            with_stack = False,
            with_flops = False,
            with_modules = True,
            record_shapes=True
            ) as prof:

        for i, batch in enumerate(dataloader):  
            with torch.no_grad():
                output = block(batch)
            prof.step()
            print(i)
            if i == wait + warmup + active - 1:
                break