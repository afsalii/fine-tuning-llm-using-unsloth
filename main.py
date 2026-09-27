
import sys
import bitsandbytes as bnb
print(sys.executable)
print(sys.version)

from unsloth import FastLanguageModel
import torch
import torch.nn as nn




#model loading function using unsloth library
def loading_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",max_seq_length: int = 248,load_in_4bit: bool = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    return model, tokenizer


#to check the total number of parameters in the model
def total_parametrs(model):
    model=model.parameters()
    total_params = sum(p.numel() for p in model)
    return total_params


def isit4bit(model):
    for name,model in model.named_modules():
        if isinstance(model,bnb.nn.Linear4bit ):
            return True
    return False


