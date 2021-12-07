import collections
import json
import os
from transformers import EvalPrediction, logging
import numpy as np
from sklearn.metrics import classification_report

logger = logging.get_logger(__name__.replace('_', ''))

class Evaluater:
    config = None
    train_set = None
    dev_set = None
    test_set = None
    stage = 'dev'
    @staticmethod
    def evaluate_cls(evalPrediction:EvalPrediction ):
        if type(evalPrediction[0]) is not tuple:
            evalPrediction_logits = [evalPrediction[0]]
        else:
            evalPrediction_logits = evalPrediction[0]
        if type(evalPrediction[1]) is not tuple:
            evalPrediction_labels = [evalPrediction[1]]
        else:
            evalPrediction_labels = evalPrediction[1]

        return_res = {}
        logit_idx = 0
        label_idx = 0
        
        return return_res


