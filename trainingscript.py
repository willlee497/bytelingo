from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Configuration
max_seq_length = 2048
dtype = None
load_in_4bit = True
model_name = "unsloth/qwen2.5-coder-7b-instruct"

# Load and preprocess CodeAlpaca-20k Dataset
dataset = load_dataset("sahil2801/CodeAlpaca-20k")

# Format the dataset into a single "text" field
def format_codealpaca(examples):
    instruction = examples["instruction"]
    input_text = examples["input"]
    output_text = examples["output"]
    if input_text.strip():  # If input is non-empty
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    return {"text": text}

dataset = dataset.map(format_codealpaca, batched=False)

# Load Qwen2.5-Coder-7B-Instruct Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    max_seq_length=max_seq_length,
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Define Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],  # Use the preprocessed train split
    dataset_text_field="text",  # Specify the formatted "text" field
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

# Fine-Tune the Model
trainer.train()

# Save the Fine-Tuned Model
model.save_pretrained("fine_tuned_qwen2.5_instruct")
tokenizer.save_pretrained("fine_tuned_qwen2.5_instruct")
print("Fine-tuning complete. Model saved to 'fine_tuned_qwen2.5_instruct'")
