
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


#the words or sentence are converted into tockens ,in some llm the pad tockens like eos are not defined so we define it here
def tockeniser(tokenizer):
    if tokenizer is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


#get_lora_adapters

def lora_get_adapters():
    return ["k_proj","q_proj","v_proj","o_proj"]



#attach lora adapters

def attach_lora_adapters(model,r=8,lora_alpha=16,target_modules=None):
    if target_modules is None:
        target_modules = lora_get_adapters()
    
    adapters=FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,  
            target_modules=target_modules
        )
    return adapters


def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fraction_of_parameters(trainable_params, total_params):
    return trainable_params / total_params if total_params > 0 else 0


model, tokenizer = loading_model(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=256,
    )

print("Total Parameters in the model:",total_parametrs(model))

if isit4bit(model) =="True":
    print('True')
else:
    print('False') 


print("Tokenizer:", tockeniser(tokenizer))

print("LoRA Adapters:", lora_get_adapters())


lora_adapters = attach_lora_adapters(model, r=8, lora_alpha=16)

print("Number of Parameters in LoRA Adapters:", number_of_parameters(model))

fraction = fraction_of_parameters(number_of_parameters(model), total_parametrs(model))
print("Fraction of Parameters in LoRA Adapters:", fraction)
