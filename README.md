# fine-tuning-llm-using-unsloth
fine tuning  qwen model of 0.5 billion parameters to a custom instruct dataset using unsloth


# process in finetuning
1.import FastLanguageModel from unsloth 

2.using from_pretrained function get model from huggingface

3.we are using Qwen model which has 0.5 billion parameters

3.set the sequence_length,load it in 4bit
