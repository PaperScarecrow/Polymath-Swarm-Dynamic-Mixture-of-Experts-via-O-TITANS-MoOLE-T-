import os
import sys
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- 1. Architectural Paths ---
PROJECT_DIR = "<your directory>"
ENGRAMS_PATH = os.path.join(PROJECT_DIR, "engrams.json")
ADAPTER_DIR = os.path.join(PROJECT_DIR, "adapters") # Folder holding our .pt files

# We use the 4B for the router and the 12B for the synthesizer
BRAINSTEM_ID = "mlabonne/gemma-3-4b-it-abliterated"
FRONTAL_LOBE_ID = "mlabonne/gemma-3-12b-it-abliterated"

sys.path.append(PROJECT_DIR)
from otitans_surgery import inject_orthogonal_memory

class PolymathOrchestrator:
    def __init__(self):
        print("[*] Initializing Polymath Swarm...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engrams = self.load_engrams()
        
        # Load Frontal Lobe (12B Synthesizer)
        print(f"[*] Booting Frontal Lobe Synthesis Core: {FRONTAL_LOBE_ID}")
        self.fl_tokenizer = AutoTokenizer.from_pretrained(FRONTAL_LOBE_ID, trust_remote_code=True)
        self.fl_model = AutoModelForCausalLM.from_pretrained(
            FRONTAL_LOBE_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare the Frontal Lobe's surgical architecture
        print("[*] Sculpting Frontal Lobe Orthogonal Vectors...")
        inject_orthogonal_memory(self.fl_model, target_modules=["q_proj", "v_proj"], rank=16, alpha=32.0)
        self.fl_model.to(torch.bfloat16).to(self.device)
        
        # Cache the sterile state of the Frontal Lobe so we can "wipe" it between prompts
        self.sterile_state_dict = {k: v.clone().cpu() for k, v in self.fl_model.state_dict().items() if "lora" in k}

        # Load Brainstem (4B Router)
        # Load Brainstem (4B Router Base + Lobotomized Adapter)
        print(f"[*] Booting Brainstem Routing Core: {BRAINSTEM_ID}")
        self.bs_tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_DIR, "brainstem_router_v1", "final_adapter"), trust_remote_code=True)
        
        bs_base_model = AutoModelForCausalLM.from_pretrained(
            BRAINSTEM_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        # Wrap the base model in our deterministic routing adapter
        self.bs_model = PeftModel.from_pretrained(
            bs_base_model, 
            os.path.join(PROJECT_DIR, "brainstem_router_v1", "final_adapter")
        )
        self.bs_model.eval()
        self.fl_model.eval()
        print("[*] Swarm Architecture Online.\n" + "-"*50)

    def load_engrams(self):
        with open(ENGRAMS_PATH, "r") as f:
            return json.load(f)

    def get_routing_decision(self, user_prompt):
        # 1. Inject the dynamic registry into the Brainstem
        system_text = "You are the Polymath Brainstem. You do not answer user queries. Your only function is task decomposition and routing. Analyze the user prompt, break down the required tasks inside <think> tags, and then output the exact engram keys required in a [ROUTE: key1, key2] format.\n\n[AVAILABLE ENGRAMS]\n"
        for key, desc in self.engrams.items():
            system_text += f"{key}: {desc}\n"
            
        messages = [
            {"role": "system", "content": system_text.strip()},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.bs_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.bs_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        print(f"\n[Brainstem] Analyzing intent...")
        with torch.no_grad():
            outputs = self.bs_model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
            
        response = self.bs_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[Brainstem] Output:\n{response.strip()}\n")
        
        # 2. Deterministic Parsing
        route_match = re.search(r"\[ROUTE:\s*(.*?)\]", response)
        if route_match:
            keys = [k.strip() for k in route_match.group(1).split(",")]
            # Filter against hallucinations
            valid_keys = [k for k in keys if k in self.engrams]
            return valid_keys
        return []

    def execute_hot_swap(self, active_engrams):
        # First, wipe the Frontal Lobe clean to prevent skill-bleed from the last prompt
        self.fl_model.load_state_dict(self.sterile_state_dict, strict=False)
        
        if not active_engrams:
            print("[Orchestrator] No specialized engrams required. Using baseline logic.")
            return

        print(f"[Orchestrator] Hot-swapping O-TITANS Adapters: {active_engrams}")
        for engram in active_engrams:
            adapter_file = os.path.join(ADAPTER_DIR, f"otitans_{engram}.pt")
            if os.path.exists(adapter_file):
                # We load the weights dynamically. Because they are orthogonal, they add linearly without destruction.
                adapter_weights = torch.load(adapter_file, map_location=self.device, weights_only=True)
                self.fl_model.load_state_dict(adapter_weights, strict=False)
            else:
                print(f"[!] Warning: Adapter {adapter_file} not found on disk.")

    def synthesize(self, user_prompt):
        messages = [{"role": "user", "content": user_prompt}]
        prompt = self.fl_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.fl_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        streamer = TextStreamer(self.fl_tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print(f"\n[Frontal Lobe Synthesis]: ", end="", flush=True)
        with torch.no_grad():
            self.fl_model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                streamer=streamer
            )
        print("\n" + "-"*50)

def main():
    swarm = PolymathOrchestrator()
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input.strip():
                continue
                
            active_engrams = swarm.get_routing_decision(user_input)
            swarm.execute_hot_swap(active_engrams)
            swarm.synthesize(user_input)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
