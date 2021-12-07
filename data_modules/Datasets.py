import numpy as np
from torch.utils.data import Dataset
import torch
from scripts.Evaluater import Evaluater

MODEL_ARGS = []

def build_all_dataset(config, tokenizer):
    train_set = MyDataset(config,  tokenizer)
    dev_set = MyDataset(config, tokenizer)
    test_set = MyDataset(config, tokenizer)

    Evaluater.train_set = train_set
    Evaluater.dev_set = dev_set
    Evaluater.test_set = test_set
    
    return train_set, dev_set, test_set

class MyDataset(Dataset):
    def __init__(self, config, instances, vocab, vocab_gnn, dialogs=None, dp_instance=None, tokenizer=None, is_training=True):
        super().__init__()
        self.config = config
        MODEL_ARGS.clear()
        
        assert len(MODEL_ARGS)!=0 , '请检查要做哪些任务，当前模型参数为空，其参数不能为空！'
        self.total_items = []
        
        

    def __getitem__(self, index):

        one_item = self.total_items[index]

        return one_item

    def __len__(self):
        return len(self.instances)



def my_data_collator(onebatch):
    return_data = {}
    for one_item in onebatch:
        for k in MODEL_ARGS:
            if k not in return_data.keys() and one_item[k] is not None:
                return_data[k] = [one_item[k]]
            elif k in one_item.keys() and  one_item[k] is not None:
                return_data[k].append(one_item[k])

    for k in return_data.keys():
        if type(return_data[k][0]) is torch.Tensor:
            return_data[k] = torch.stack(return_data[k])
    # assert len(return_data.keys()) == len(MODEL_ARGS)
    return return_data
