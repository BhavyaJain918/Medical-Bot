# Medical-Bot
A medical LLM bot that can answer queries related to medical terminologies. This code uses Qwen 1.5-0.5-Chat LLM model (open-source) and performs Few-Shot fine-tuning (16 examples) using a open-source dataset. The model is fine-tuned for 100 epochs , having a training loss of 2.228572254180908.

Requirements :
trl -- 0.21.0
peft -- 0.17.0
torch -- 2.6.0+cu124
datasets -- 4.0.0
transformers -- 4.55.0

Issues :
Can randomly generate some Chinese words/tokens/symbols instead of English , as the model was trained on both Chinese and English text.

It is advised to run the code on a GPU with atleast 8 GB VRAM , otherwise the training will take a long time. 
