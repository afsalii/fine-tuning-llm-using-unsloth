
import sys

print(sys.executable)
print(sys.version)

from unsloth import FastLanguageModel
import torch


if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")

def loading_model(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",max_seq_length: int = 248,load_in_4bit: bool = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    return model, tokenizer
    





