from transformers import AutoTokenizer

class MyTokenizer(AutoTokenizer):
    def __init__(self):
        
        # special_tokens_dict = {'additional_special_tokens': ['url', 'filepath', '<root>']}
        # self.add_special_tokens(special_tokens_dict)
        print("Load bert vocabulary finished.")








