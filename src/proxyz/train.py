import click
import torch
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from proxyz.utils import dict2object

@click.command()
@click.option(
    "--tokenizer_file",
    type=click.Path(),
    default="my_tokenizer.json",
    help="Path to the tokenizer directory, where the files saved."
)
@click.option("--model_hidden_size", type=int, default=2048, help="Model width.")
@click.option(
    "--model_intermediate_size",
    type=int,
    default=5632,
    help="Model SwiGLU hidden dimension (usually ~8/3 of hidden_size)."
)
@click.option("--model_num_hidden_layers", type=int, default=24, help="Model depth.")
@click.option(
    "--model_num_attention_heads", type=int, default=16, help="Model attention heads."
)
@click.option(
    "--model_num_key_value_heads",
    type=int,
    default=4,
    help="Model Grouped-Query Attention (GQA) for speed."
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    # ==========================================
    # 1. LOAD YOUR CUSTOM BPE TOKENIZER
    # ==========================================
    # Wrap your standalone BPE json file into the Hugging Face ecosystem
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        # bos_token="<｜begin of sentence｜>",  # DeepSeek style BOS
        # eos_token="<｜end of sentence｜>",    # DeepSeek style EOS
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    
    # Ensure the embedding layer matches this size exactly
    vocab_size = len(tokenizer) 
    
    # ==========================================
    # 2. CONFIGURE DEEPSEEK-STYLE ARCHITECTURE
    # ==========================================
    # DeepSeek-V2/V3 use Llama-based primitives (SwiGLU, RMSNorm, RoPE)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.model_hidden_size,                  # Model width
        intermediate_size=args.model_intermediate_size,      # SwiGLU hidden dimension (usually ~8/3 of hidden_size)
        num_hidden_layers=args.model_num_hidden_layers,      # Depth
        num_attention_heads=args.model_num_attention_heads,  # Attention heads
        num_key_value_heads=args.model_num_key_value_heads,  # Grouped-Query Attention (GQA) for speed
        hidden_act="silu",                                   # SiLU activation for SwiGLU
        max_position_embeddings=4096,                        # Context window length
        initializer_range=0.02,
        rms_norm_eps=1e-6,                                   # DeepSeek RMSNorm epsilon
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False                            # DeepSeek keeps input/output embeddings separate
    )
    
    model = LlamaForCausalLM(config)

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--- Dense DeepSeek-Style Model ---")
        print(f"Total Parameters:    {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    
    ## ==========================================
    ## 3. PREPARE YOUR DATASET (Line-by-Line)
    ## ==========================================
    ## Mock generator mimicking your line-by-line file reading pipeline
    #def data_generator():
    #    # Replace this block with your actual line-by-line reading logic
    #    with open("my_data.txt", "r", encoding="utf-8") as f:
    #        for line in f:
    #            clean_line = line.rstrip("\r\n")
    #            if clean_line:
    #                yield {"text": clean_line}
    #
    ## Convert your generator directly to a Hugging Face Dataset object
    #dataset = Dataset.from_generator(data_generator)
    #
    ## Tokenize the dataset using the configuration chosen above
    #def tokenize_function(examples):
    #    return tokenizer(
    #        examples["text"], 
    #        truncation=True, 
    #        max_length=config.max_position_embeddings
    #    )
    #
    #tokenized_dataset = dataset.map(
    #    tokenize_function, 
    #    batched=True, 
    #    remove_columns=["text"]
    #)
    #
    ## Data collator handles shifting labels internally for Causal LM training
    #data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    #
    ## ==========================================
    ## 4. TRAINING ARGUMENTS & EXECUTION
    ## ==========================================
    #training_args = TrainingArguments(
    #    output_dir="./deepseek_style_model",
    #    overwrite_output_dir=True,
    #    per_device_train_batch_size=4,   # Scale up based on your VRAM
    #    gradient_accumulation_steps=8,  # Simulates a larger batch size
    #    learning_rate=3e-4,             # Typical learning rate for small models
    #    weight_decay=0.1,
    #    adam_beta1=0.9,
    #    adam_beta2=0.95,                # DeepSeek beta2 standard
    #    logging_steps=10,
    #    save_steps=500,
    #    fp16=torch.cuda.is_available(), # Enable mixed precision if using GPU
    #    num_train_epochs=3,
    #    report_to="none"                # Switch to "wandb" if tracking metrics
    #)
    #
    #trainer = Trainer(
    #    model=model,
    #    args=training_args,
    #    train_dataset=tokenized_dataset,
    #    data_collator=data_collator,
    #)
    #
    ## Start pre-training from scratch
    #trainer.train()
    #
    ## Save final weights and configuration
    #model.save_pretrained("./deepseek_style_model")
    #tokenizer.save_pretrained("./deepseek_style_model")


if __name__ == "__main__":
    main()
