from argparse import ArgumentParser
import torch
from transformers.utils import logging
from Config import  MyConfigs
from data_modules.MyTokenizer import MyTokenizer
from modules.Model import build_model
from data_modules.Datasets import  build_all_dataset, my_data_collator
from scripts.Trainer import MyTrainer
from scripts.utils import set_logger, set_seed
from scripts.Evaluater import Evaluater
import os 

logger = logging.get_logger(__name__.replace('_', ''))


if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): # 注释掉这几行即可使用分布式训练
    print(' 未指定使用的GPU，将使用 0 卡。')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def set_configs():
    argsParser = ArgumentParser()
    argsParser.add_argument('--config_file', type=str, default='configs/debug.cfg')
    args, extra_args = argsParser.parse_known_args()
    if not os.path.exists(args.config_file):
        print('Config file {} not found.'.format(args.config_file))
        raise FileNotFoundError
    
    # 解析参数
    configs = MyConfigs(args.config_file, extra_args)

    # 设置 root logger
    set_logger(to_console=True, log_file=configs.log_file)
    logger.info("------------  Process ID {}, Process Parent ID {}  --------------------\n".format(os.getpid(), os.getppid()))
    configs.save()

    # 可以在下方手动覆盖所有参数的值
    # configs.trainer_args.do_train = False

    return configs


if __name__ == '__main__':
    config = set_configs()
    set_seed(config.seed)
    Evaluater.config = config
    tokenizer = MyTokenizer.from_pretrained(config.pretrained_model_name_or_path, use_fast=False)
    train_set, dev_set, test_set, dp_vocab = build_all_dataset(config, tokenizer)
    model = build_model(config, dp_vocab, tokenizer)

    trainer = MyTrainer(model, config.trainer_args, 
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = my_data_collator,
                        compute_metrics=Evaluater.evaluate_cls,
                        
                        )
    
    if config.trainer_args.do_train:
        logger.info('start training...\n\n')
        trainer.train()
        torch.save(model.state_dict(), config.best_model_file)
    
    if config.trainer_args.do_eval:
        model.load_state_dict(torch.load(config.best_model_file))
        logger.info('Use best model to evaluate dev dataset:')
        trainer.evaluate(dev_set)
        logger.info('Use best model to predict test dataset ...')
        Evaluater.stage = 'test'
        pred_rtn = trainer.predict(test_set)
        logits_tuple, _, pred_info = pred_rtn[0], pred_rtn[1], pred_rtn[2]
        pred_labels, attention_mask, topk_logits = logits_tuple
    
    logger.info('---------------------------  Finish!  ----------------------------------\n\n')


