import json
from transformers.training_args import TrainingArguments
from transformers.utils import logging
import os
from configparser import ConfigParser, ExtendedInterpolation

logger = logging.get_logger(__name__.replace('_', ''))


class MyConfigs():
    def __init__(self, config_file, extra_args=None):
        self.config_file = config_file
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_file, encoding="utf-8")
        if extra_args:  # 如果命令行中有参数与config中的相同，则值使用命令行中传入的参数值
            extra_args = extra_args = dict([ (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
            for section in config.sections():
                for k, v in config.items(section):
                    if k in extra_args:
                        v = type(v)(extra_args[k])
                        config.set(section, k, v)
        
        self._config = config
        self.train_args_dict={}
        for section in config.sections():
            for k, v in config.items(section):
                v = self.get_type(v)
                if 'Trainer' == section:
                    self.train_args_dict[k] = v
                self.__setattr__(v)
        
        
        self.post_init()
    def get_type(self,v):
        if '[' == v[0] and ']' == v[-1]:
            v = v.replace('[', '')
            v = v.replace(']', '')
            tmp = v.split(',')
            v = []
            for t in tmp:
                v.append(self.get_one_type(t))
        else:
            v = self.get_one_type(v)
        return v

    def get_one_type(self, v):
        """
        设置值的类型
        """

        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'none':
            v = None
        else:
            try:
                v = eval(v)
            except:
                v = v
        return v
        
        

    def post_init(self):

        if self.temp_dir is not None:
            self.temp_dir = os.path.expanduser(self.temp_dir)
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

        self.output_dir = self.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.log_file = self.log_file
        if not os.path.exists(self.log_file.rsplit('/', 1)[0]):
            os.makedirs(self.log_file.rsplit('/', 1)[0])

        self.trainer_args = TrainingArguments(**self.train_args_dict)

    def save(self):
        logger.info('Loaded config file from {} sucessfully.'.format(self.config_file))
        self._config.write(open(self.output_dir+'/'+self.config_file.split('/')[-1], 'w'))
        logger.info('Write this config to {} sucessfully.'.format(self.output_dir+'/'+self.config_file.split('/')[-1]))
        out_str = '\n'
        for section in self._config.sections():
            for k, v in self._config.items(section):
                out_str +='{} = {}\n'.format(k,v)
        logger.info(out_str)
    
    def to_json_string(self):
        out_json = {}
        for section in self._config.sections():
            for k, v in self._config.items(section):
                out_json[k] = v
        return json.dumps(out_json)

if __name__ == '__main__':
    config = MyConfigs('debug.cfg')
    print(config.log_file)
