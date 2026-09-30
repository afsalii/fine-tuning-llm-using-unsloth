# Fine-Tuning-LLM-using-unsloth
fine tuning  qwen model of 0.5 billion parameters to a custom instruct dataset using unsloth


# Processes in Finetuning
1.import FastLanguageModel from unsloth 

2.using from_pretrained function get model from huggingface

3.we are using Qwen model which has 0.5 billion parameters

3.set the sequence_length,load it in 4bit

4.some qwen models comes with preset tockenizer or some decoder models comes with pad_tocken as None ,when the pad_tocken is None set it as <eos> tocken which is referred as end of sentence .
