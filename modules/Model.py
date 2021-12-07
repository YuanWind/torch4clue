from torch.nn import Module
from transformers.models.bert.modeling_bert import BertForTokenClassification
import torch

def build_model(config, vocab):
    model = Model(config, vocab)
    
    return model
class Model(Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.model = BertForTokenClassification.from_pretrained(config.pretrained_model_name_or_path, num_labels=self.vocab.label_size)
        
    def optimizer_params(self):
        """
        获取优化器的输入参数，在这里可以设定各层不同的学习率等等
        :return: 优化器输入参数
        """
        params = [] 

        return params

    def forward(self, **kwargs):
        # inputs = {
        #     'input_ids': kwargs.pop('input_ids'),
        #     'attention_mask': kwargs.pop('attention_mask'),
        #     'token_type_ids': kwargs.pop('token_type_ids')
        # }
        outputs = self.model(**kwargs)
        logits = outputs.logits
        logits = torch.softmax(logits,dim = -1)
        topk_logits,pred_labels = torch.topk(logits,k=self.config.topk, dim=-1)
        
        model_outputs = {
            'pred_labels':pred_labels,
            'attention_mask':kwargs['attention_mask'],
            'topk_logits':topk_logits
        }
        if outputs.loss is not None:
            model_outputs['loss'] = outputs.loss
        torch.cuda.empty_cache()
        return model_outputs
