from transformers import AutoModelForCausalLM, AutoTokenizer
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
import torch
from torch.utils.data import Dataset, DataLoader

checkpoint = "bigcode/starcoder"
device = "cpu" # for GPU usage or "cpu" for CPU usage


class RandomDataset(Dataset):
    def __init__(self, ):
        pass

    def __getitem__(self, index):
            
        #seq_len = random.randint(30, 100)
        seq_len = 40
        data = torch.randint(0, 10000, (seq_len,))

        return data
    
    def __len__(self):
        return 1000
    
class RandomCollator(object):
    def __init__(self):
        pass
    def  __call__(self, batch_list):

        batch = torch.nn.utils.rnn.pad_sequence(batch_list, batch_first=True)
        return batch
    
collat = RandomCollator()
dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collat)
inp = next(iter(dataloader))


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.config.return_dict=False
model.eval()



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
                                calibration_sampling_size = 4,
                                op_type_dict=op_type_dict,) 

block = quantization.fit(model=model,
                        conf=conf,
                        calib_dataloader=dataloader)

block(inp)