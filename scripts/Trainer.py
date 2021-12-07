from transformers.trainer import Trainer


class MyTrainer(Trainer):
    def __init__(self, model, configs, **kwargs):
        super().__init__(model, args=configs, **kwargs)
