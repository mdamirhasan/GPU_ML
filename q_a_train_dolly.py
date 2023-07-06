from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, AutoConfig, GPTJForCausalLM, DataCollatorForLanguageModeling
import os
import torch.nn as nn
import bitsandbytes as bnb
import transformers
# import deepspeed
from functools import partial
import numpy as np
from peft.tuners.lora import LoraLayer

# from transformers import AutoTokenizer,AutoModelForCausalLM
# from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch, time

starttime = time.time()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
  #  bnb_4bit_compute_dtype=torch.bfloat16
)

OUTPUT_DIR = "models/q_a/qlora_shorthills_dolly_12B_V2"
model_name = "databricks/dolly-v2-12b"

TRAIN_DATA_PATH = "q_a_train_data.json"


INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42
INTRO_BLURB = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.")

# Settings for A100 - For 3090  2600/4
MICRO_BATCH_SIZE = 3  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # paper uses 3
LEARNING_RATE = 2e-4  #2e-5, 2e-4
# CUTOFF_LEN =512

LORA_R = 64 #4, 8, 64
LORA_ALPHA = 16   #16,32,16
LORA_DROPOUT = 0.05


tokenizer = AutoTokenizer.from_pretrained(model_name)

data = load_dataset("json", data_files="q_a_train_data.json")

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Below input is from various reviews, blog articles, videos of a product from category {category}. Answer the question as truthfully as possible using the provided input, and if the answer is not contained within the given knowledge, say "I don't know
Question: {question}

### Input:
{context}

### Response:
{response}"""



class DataCollatorForCompletionOnlyLMV(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        response_token_ids = self.tokenizer.encode(RESPONSE_KEY)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch

def preprocess_batch(batch, tokenizer: AutoTokenizer, max_length: int):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def find_all_linear_names(model):

    cls = bnb.nn.Linear4bit
    
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
        

    return list(lora_module_names)

def load_training_dataset(path_or_dataset) -> Dataset:
    dataset = load_dataset("json",data_files=path_or_dataset)["train"]

    def _add_text(rec):
        
        question = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")
        category =  rec["category_slug"]
                

        if not question:
            raise ValueError(f"Expected an Question in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        if context:
            rec["text"] = prompt_template.format(category= category , question=question, context=context, response =response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, training_dataset=TRAIN_DATA_PATH):
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(training_dataset)

    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)

    return dataset

def freeze_layer_types(model):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            # if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                # if args.bf16 and module.weight.dtype == torch.float32:
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
    return model


def train():
    
   # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit= True, quantization_config=quant_config, device_map={"":0})
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit= True, quantization_config=quant_config, device_map="auto")    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    modules = find_all_linear_names(model)
    
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=modules, 
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)
    #model = freeze_layer_types(model)
        
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = 512
        

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=DEFAULT_SEED, training_dataset=TRAIN_DATA_PATH)
    split_dataset = processed_dataset.train_test_split(test_size=200, seed=42)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY, RESPONSE_KEY_NL]})

    model = model
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            warmup_steps=10,
            num_train_epochs=EPOCHS, #2
            learning_rate=1e-5,
            fp16=True, #True
            bf16=False, #False
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            save_steps=1828,
            logging_steps=100,
            output_dir=OUTPUT_DIR,
            save_total_limit=1,
            optim="paged_adamw_8bit"
            
        ),
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=DataCollatorForCompletionOnlyLMV(tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8),
    )
    
    
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

    

train()

timetaken = time.time()-starttime
print("time taken:", timetaken)


