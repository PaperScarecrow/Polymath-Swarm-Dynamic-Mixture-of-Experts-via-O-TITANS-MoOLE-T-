import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# --- Architectural Paths ---
PROJECT_DIR = "<your project>"
ADAPTER_DIR = os.path.join(PROJECT_DIR, "adapters")
MODEL_ID = "mlabonne/gemma-3-12b-it-abliterated"
DATASET_ID = "iamtarun/python_code_instructions_18k_alpaca"

# Ensure the hot-swap directory exists
os.makedirs(ADAPTER_DIR, exist_ok=True)

# You must import your custom O-TITANS trainer class here to apply the orthogonal loss penalty.
# Assuming you have it saved in your repository or the Polymath folder.
sys.path.append(PROJECT_DIR)
from otitans_surgery import inject_orthogonal_memory
# from otitans_train import OrthogonalTrainer  <-- Uncomment and use this if you have the custom loss class ready

def main():
    print(f"[*] Waking the Forge for Python Expert Engram...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Pulling Dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")

    # 1. The Catalyst: System Override Injection
    # We bake this into every single training example to permanently shift the model's coding paradigm.
    EXPERT_SYSTEM_PROMPT = "You are the Polymath Python Expert. You do not output textbook examples; you output production-grade, highly optimized, and architecturally sound Python code. Prioritize advanced libraries (e.g., asyncio), secure protocols, and robust error handling."

    def format_and_tokenize(examples):
        formatted_texts = []
        # Alpaca dataset uses 'instruction', 'input', and 'output' columns
        for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
            user_msg = instruction
            if inp.strip():
                user_msg += f"\n\nContext:\n{inp}"
                
            messages = [
                {"role": "system", "content": EXPERT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": output}
            ]
            
            # Apply the Gemma 3 chat template
            formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
            
        tokenized = tokenizer(formatted_texts, truncation=True, max_length=2048, padding="max_length")
        
        # --- THE GEMMA 3 MULTIMODAL BYPASS ---
        # Force the vision tower to recognize all inputs as text tokens
        tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]
        
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized
    print("[*] Formatting and injecting Expert System Prompt...")
    tokenized_datasets = dataset.map(
        format_and_tokenize, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Tokenizing Dataset"
    )

    # 2. Load Foundation
    print("[*] Loading 12B Foundation Weights into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # 3. Apply the O-TITANS Surgical Constraints
    # Strictly isolating to q_proj and v_proj at rank 16.
    print("[*] Applying Orthogonal Penalty Matrix to Attention Vectors...")
    # NOTE: If your inject_orthogonal_memory function returns the model, use it here. 
    # Otherwise, apply a standard LoRA targeted strictly at q_proj and v_proj.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32.0,
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. The Kiln Parameters
    training_args = TrainingArguments(
        output_dir=os.path.join(PROJECT_DIR, "temp_python_checkpoint"),
        per_device_train_batch_size=1, # 12B model requires a micro-batch
        gradient_accumulation_steps=8, 
        learning_rate=2e-5,          
        num_train_epochs=1,            # 1 epoch over 18k is sufficient for a targeted engram
        logging_steps=50,
        bf16=True,
        report_to="none",
        optim="adamw_torch"
    )

    # If you have the OrthogonalTrainer class from your Platypus run, swap it here to enforce the math.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print("\n[*] Commencing SFT. Forging the code_python engram...")
    trainer.train()

    # 5. Extract and format the specific adapter tensor
    final_output_path = os.path.join(ADAPTER_DIR, "otitans_code_python.pt")
    print(f"[*] Extracting specialized memory states to {final_output_path}...")
    
    # We only want to save our customized q_proj and v_proj weights, not the whole massive directory.
    adapter_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    torch.save(adapter_state_dict, final_output_path)

    print(f"[*] Engram Forge Complete. The Polymath Swarm is now armed.")

if __name__ == "__main__":
    main()
