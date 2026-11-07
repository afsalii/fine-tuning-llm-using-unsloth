
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



def training_data():
    dataset=[
  {
    "instruction": "What is InnoRoots AI?",
    "response": "InnoRoots AI is an AI and robotics education initiative based in Mannarkkad, Kerala. It focuses on empowering learners through practical training in Artificial Intelligence, Machine Learning, Robotics, and emerging technologies."
  },
  {
    "instruction": "What courses does InnoRoots AI offer?",
    "response": "InnoRoots AI provides training in Artificial Intelligence, Machine Learning, Robotics, and other technology-related fields. Learners can explore its official website for current course offerings."
  },
  {
    "instruction": "Where is InnoRoots AI located?",
    "response": "InnoRoots AI is located opposite MES Kalladi College, Mannarkkad, Palakkad, Kerala, India."
  },
  {
    "instruction": "What is the vision of InnoRoots AI?",
    "response": "InnoRoots AI aims to democratize intelligence and revolutionize technology through robotics by empowering learners with practical technological skills."
  },
  {
    "instruction": "Does InnoRoots AI provide robotics training?",
    "response": "Yes. Robotics is one of the key areas of InnoRoots AI's educational initiatives. Its learning approach emphasizes practical projects, experimentation, and hands-on engineering."
  }
]

    return dataset

def formatted_data(dataset):
    strings=( 
        '### Instruction:\n'
        f"{dataset['instruction']}\n\n"
        '### Response:\n'
         f"{dataset['response']}"
    )
    return strings

def format_all(examples):
    alle=[formatted_data(example) for example in examples]

    return alle

from datasets import Dataset

def build_text_dataset(texts):
    dataset = Dataset.from_dict({"text": texts})
    return dataset  


def tockenize(tokenizer, texts):
    tokenized_dataset = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256
    )
    return tokenized_dataset


def count_tokens(tokenized_dataset):
    return len(tokenized_dataset)


print(format_all(training_data())) 

tockenized_dataset = tockenize(tokenizer, format_all(training_data()))
print("Number of Tokens in Tokenized Dataset:", count_tokens(tockenized_dataset))


from transforms import TrainingArguments


def build_training_Arguments():
     training=TrainingArguments(
            output_dir=".",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            report_to="none",
            save_strategy="no",
            optim="adamw_torch",
     )
     return training

from trl import SFTTrainer

def sft_training():

    training_sft=SFTTrainer(
        model=model,    
        dataset=tockenized_dataset,
        tokenizer=tokenizer,
        args=training_Args,
        max_seq_length=256,
    )
    return training_sft


#run the sft training and resultant loss
def run_sft_training(trainer):
    
    train=trainer.train()
    metric=trainer.metrics
    resultant_loss = float(metric["train_loss"])
    return resultant_loss


