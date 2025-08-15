! pip install accelerate peft bitsandbytes trl

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from transformers import BitsAndBytesConfig , AutoModelForCausalLM , AutoTokenizer , TrainingArguments , pipeline

dataset = load_dataset("gamino/wiki_medical_terms" , split = "train")   # Load dataset


lora_config = LoraConfig(r = 64 , lora_alpha = 16 , lora_dropout = 0.02 , task_type = "CAUSAL_LM")   # Using LoRA for Parameter Efficient Fine Tuning (PEFT)

def preprocessed(sample) :
  us = "Explain " + sample["page_title"]
  bt = ".".join(sample["page_text"].split("."))
  sy = "You are a helpful assistant. Please answer all queries truthfully in English and in context of the medical world."

  return {"chat" : [{"role" : "system" , "content" : sy} , {"role" : "user" , "content" : us} , {"role" : "assistant" , "content" : bt}]}   # Format expected by Qwen model

dataset_mapped = dataset.from_dict(dataset[0 : 16]).map(preprocessed , remove_columns = ["__index_level_0__" , "page_title" , "page_text"])   # Loading only the first 16 rows to perform Few-Shot training

quantization = BitsAndBytesConfig(load_in_4bit = True , bnb_4bit_compute_dtype = torch.float16 , bnb_4bit_quant_type = "nf4" , bnb_4bit_use_double_quant = True)   # Paramters for 4-bit quantization

model_llama = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat" , quantization_config = quantization , trust_remote_code = True)    # Loading model

model_llama.config.use_cache = False
model_llama.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")    # Loading model tokenizer

tokenizer.model_max_length = 131072
tokenizer.pad_token = tokenizer.eos_token

dataset_preprocess = dataset_mapped.map(lambda row : {"formatted_text" : tokenizer.apply_chat_template(row["chat"] , tokenize = False , add_generation_prompt = True)} , remove_columns = "chat")

llama_train_arguments = TrainingArguments(output_dir = "./Medical-Bot" , per_device_train_batch_size = 2 , logging_steps = 40 , max_steps = 800 , weight_decay = 1e-3 , report_to = "none")   # Arguments for training model

torch.cuda.empty_cache()
trainer = SFTTrainer(model = model_llama , args = llama_train_arguments , train_dataset = dataset.from_dict({"text" : dataset_preprocess["formatted_text"]}) , peft_config = lora_config)   # Actual supervised training object 

trainer.train()


instruction = "Explain Blastic plasmacytoid dendritic cell neoplasm"

instructions = tokenizer.apply_chat_template([{"role" : "user" , "content" : instruction}] , tokenize = False , add_generation_prompt = True)
model_pipeline = pipeline(task = "text-generation" , model = model_llama , tokenizer = tokenizer , max_length = 1000 , torch_dtype = torch.float16 , temperature = 0.2)   # Model testing

model_output = model_pipeline(instructions)   # Feeding properly formatted instruction into the model pipeline

print(model_output[0]["generated_text"])


token_saved = AutoTokenizer.from_pretrained("Medical-Bot/checkpoint-800")
model_saved = AutoModelForCausalLM.from_pretrained("Medical-Bot/checkpoint-800" , use_safetensors = True) # Code to load model and tokenizer from created checkpoint

saved_pipeline = pipeline(task = "text-generation" , model = model_saved , tokenizer = token_saved , max_length = 1000 , torch_dtype = torch.float16 , temperature = 0.2)

saved_output = saved_pipeline(token_saved.apply_chat_template([{"role" : "user" , "content" : "Explain Blastic plasmacytoid dendritic cell neoplasm"}] , tokenize = False , add_generation_prompt = True))   # Testing using the checkpoint

print(saved_output[0]["generated_text"])