# Fine-Tuning-LLM-using-unsloth
fine tuning  qwen model of 0.5 billion parameters to a custom instruct dataset using Lora
thanks to unsloth


# Processes in Finetuning
1.import FastLanguageModel from unsloth (helps us to import model from huggingface,and model tockenizer)

2.using from_pretrained function get model from huggingface

3.we are using Qwen model which has 0.5 billion parameters

3.set the sequence_length,load it in 4bit

4.some qwen models comes with preset tockenizer or some decoder models comes with pad_tocken as None ,when the pad_tocken is None set it as <eos> tocken which is referred as end of sentence .

5.some of the model comes with inbuild pad tocken(like eos,end of text ,here we check if there is any prebuild pad tocken)if not is change into eos tocken)
(in our model there is more pad tockens)

5.returning the projection layers(attention layers) where the lora adapters we need to train ,these process is reliable for every model ,but we are training on qwen model 


6.normaly qwen model projection are 4,if target_layers are None,we returns the default model project layers otherwise  we use unsloth function get_peft model 
