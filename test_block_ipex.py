import torch
import intel_extension_for_pytorch as ipex
import os
from torch.profiler import ProfilerActivity, profile
import time
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig, HistogramObserver
from intel_extension_for_pytorch.quantization import prepare, convert
import torch.fx as fx
from utils import BlockTwoLayers, RandomDataset, RandomCollator
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from torch.utils.data import Dataset, DataLoader
from torch.profiler import ProfilerActivity, profile, record_function
from contextlib import nullcontext
from collections import defaultdict
import os
os.environ['ONEDNN_VERBOSE'] = '0'
os.environ['DNNL_GRAPH_VERBOSE'] = '2'
os.environ['ONEDNN_GRAPH_VERBOSE'] = '1'

device = "cpu"  # cuda:0,  cpu

dim = 6144

print(device)

class Model(torch.nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
        self.lin = torch.nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        x = self.lin(x)
        x,_ = x.split((self.dim//2, self.dim//2), dim=-1)
        x,_ = x.split((self.dim//4, self.dim//4), dim=-1)
        x = torch.matmul(x, x.transpose(-1, -2))
        return x
    

class RandomDataset(Dataset):
    def __init__(self, dim):
        self.dim = dim

    def __getitem__(self, index):
            
        seq_len = 40
        data = {}
        data['x'] = torch.rand(seq_len, self.dim)
        return data
    
    def __len__(self):
        return 1000
    
class RandomCollator(object):
    def __init__(self):
        pass
    def  __call__(self, batch_list):
        batch = defaultdict(list)
        for b in batch_list:
            for k in b:
                batch[k].append(b[k])

        batch['x'] = torch.nn.utils.rnn.pad_sequence(batch['x'], batch_first=True)
        return batch

block = Model(dim)


collat = RandomCollator()
dataset = RandomDataset(dim = dim)

block.eval()

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collat)
inp = next(iter(dataloader))
block(**inp)

qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
block = prepare(block, qconfig, example_kwarg_inputs=dict(inp), inplace=False)

block(**inp)
block = convert(block)

with torch.no_grad():
    block = torch.jit.trace(block, example_kwarg_inputs=dict(inp), check_trace=False, strict=False)
    block = torch.jit.freeze(block)

for _ in range(5):
    block(**inp)