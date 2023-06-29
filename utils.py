import torch
from intel_extension_for_pytorch.quantization._quantization_state_utils import SeenQOpInfo, SeenNonQOpInfo, QTensorInfo
from torch.utils.data import Dataset, DataLoader
import pyarrow as pa
import io
from PIL import Image
import random
from typing import List, Optional
from collections import defaultdict



class BlockTwoLayers(torch.nn.Module):
    def __init__(self, modelA, modelB):
        super(BlockTwoLayers, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self,hidden_states, attention_mask):
        hidden_states = self.modelA(hidden_states = hidden_states, attention_mask = attention_mask)
        hidden_states = self.modelB(hidden_states = hidden_states[0], attention_mask = attention_mask)
        return hidden_states
    
class RandomDataset(Dataset):
    def __init__(self, dim, random_seq_len = False):
        self.dim = dim
        self.random_seq_len = random_seq_len

    def __getitem__(self, index):
    
        if self.random_seq_len:
            seq_len = random.randint(30, 100)
        else:
            seq_len = 40
        inp_dict = {'hidden_states': torch.rand(seq_len, self.dim)}
        inp_dict['attention_mask'] = torch.rand([seq_len,48,seq_len])  < 0.9

        return inp_dict
    
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

        batch['hidden_states'] = torch.nn.utils.rnn.pad_sequence(batch['hidden_states'], batch_first=True)
        batch['attention_mask'] = torch.transpose(torch.nn.utils.rnn.pad_sequence(batch['attention_mask'], batch_first=True), 1, 3)
        
        return batch
    


class MyCollator(object):
    def __init__(self, processor, return_ids_capt = True):
        self.processor = processor
        self.return_ids_capt = return_ids_capt
    def  __call__(self, batch_list):
        image_ids, captions, images = list(zip(*batch_list))
        with torch.no_grad():
            batch = self.processor(images, captions, padding=True, return_tensors="pt").to('cpu')

        if self.return_ids_capt:
            batch['image_ids'] = image_ids
            batch['captions'] = captions

        return batch