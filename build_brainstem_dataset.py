import os
import json
import random
import itertools

PROJECT_DIR = "<your directory>"
ENGRAMS_PATH = os.path.join(PROJECT_DIR, "engrams.json")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "brainstem_sft_dataset.jsonl")

def load_engrams():
    with open(ENGRAMS_PATH, "r") as f:
        return json.load(f)

def generate_system_prompt(engrams):
    system_text = "You are the Polymath Brainstem. You do not answer user queries. Your only function is task decomposition and routing. Analyze the user prompt, break down the required tasks inside <think> tags, and then output the exact engram keys required in a [ROUTE: key1, key2] format.\n\n[AVAILABLE ENGRAMS]\n"
    for key, desc in engrams.items():
        system_text += f"{key}: {desc}\n"
    return system_text.strip()

# --- Procedural Data Matrix ---
# We define atomic components to programmatically generate hundreds of variations.

DOMAINS = {
    "code_python": {
        "queries": [
            "Write a Python script to {task}.",
            "Can you debug this RecursionError in my {task} code?",
            "How do I optimize a Python loop that handles {task}?",
            "Show me the object-oriented architecture for {task} in Python."
        ],
        "tasks": ["data sorting", "API requests", "matrix multiplication", "asynchronous threading"],
        "think": "The user is requesting structural programming logic, Python syntax, or debugging. This strictly maps to the coding engram."
    },
    "philosophy": {
        "queries": [
            "Explain the philosophical implications of {concept}.",
            "How would Kant view {concept}?",
            "Debate the existential nature of {concept}.",
            "What is the logical fallacy inherent in {concept}?"
        ],
        "tasks": ["determinism", "artificial consciousness", "utilitarian ethics", "the ship of Theseus"],
        "think": "The user is querying abstract concepts, ethics, or philosophical logic structures. No other technical domains are required."
    },
    "creative": {
        "queries": [
            "Write a short story about {theme}.",
            "Compose a poem exploring {theme}.",
            "Give me a highly descriptive paragraph focusing on {theme}.",
            "Draft a creative narrative from the perspective of {theme}."
        ],
        "tasks": ["a lonely astronaut", "a forgotten AI in a server room", "the first rain on Mars", "a detective in a cyberpunk city"],
        "think": "The user is requesting narrative synthesis, emotive text, or creative writing. This maps directly to the creative engram."
    },
    "cybersec": {
        "queries": [
            "Analyze the attack vector for {threat}.",
            "What is the standard mitigation for {threat}?",
            "Explain how a threat actor might exploit {threat}.",
            "Detail the CVSS metrics associated with {threat}."
        ],
        "tasks": ["a zero-day buffer overflow", "cross-site scripting (XSS)", "SQL injection payloads", "privilege escalation in Linux"],
        "think": "The user is asking for vulnerability analysis, threat intelligence, or cybersecurity auditing. This requires the cybersec engram."
    }
}

def generate_single_domain_data(engrams, system_prompt, count_per_domain=50):
    dataset = []
    for domain, data in DOMAINS.items():
        for _ in range(count_per_domain):
            query_template = random.choice(data["queries"])
            task = random.choice(data["tasks"])
            user_prompt = query_template.format(task=task, concept=task, theme=task, threat=task)
            
            assistant_response = f"<think>\n{data['think']}\n</think>\n[ROUTE: {domain}]"
            
            dataset.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            })
    return dataset

def generate_multi_domain_data(engrams, system_prompt, count=100):
    dataset = []
    domain_keys = list(DOMAINS.keys())
    
    # Pre-defined logical intersections for the stress-tests
    intersections = [
        (["code_python", "cybersec"], "Audit this Python script that handles {code_task} for {sec_task} vulnerabilities.", "The user provides code and asks for a vulnerability audit. This requires both Python execution logic and cybersecurity threat analysis."),
        (["code_python", "philosophy"], "Write a Python simulation of {phil_task} and add comments explaining the ethical implications.", "The request requires functional Python code combined with philosophical ethical reasoning."),
        (["creative", "cybersec"], "Write a suspenseful cyberpunk story about a hacker deploying {sec_task}.", "The request asks for creative narrative writing focused on a specific cybersecurity exploit."),
        (["philosophy", "creative"], "Compose a melancholic poem about the existential dread of {phil_task}.", "The request merges emotive, creative synthesis with deep philosophical concepts.")
    ]
    
    for _ in range(count):
        combo, prompt_template, think_text = random.choice(intersections)
        
        # Pull random tasks for the specific domains
        format_dict = {}
        if "code_python" in combo: format_dict["code_task"] = random.choice(DOMAINS["code_python"]["tasks"])
        if "cybersec" in combo: format_dict["sec_task"] = random.choice(DOMAINS["cybersec"]["tasks"])
        if "philosophy" in combo: format_dict["phil_task"] = random.choice(DOMAINS["philosophy"]["tasks"])
        if "creative" in combo: format_dict["creative_task"] = random.choice(DOMAINS["creative"]["tasks"])
        
        # Clean up the format dictionary to only include what the template needs
        valid_formats = {k: v for k, v in format_dict.items() if f"{{{k}}}" in prompt_template}
        user_prompt = prompt_template.format(**valid_formats)
        
        route_str = ", ".join(combo)
        assistant_response = f"<think>\n{think_text}\n</think>\n[ROUTE: {route_str}]"
        
        dataset.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
        })
    return dataset

def build_dataset():
    engrams = load_engrams()
    system_prompt = generate_system_prompt(engrams)
    
    print("[*] Synthesizing Single-Domain Vectors...")
    single_data = generate_single_domain_data(engrams, system_prompt, count_per_domain=100)
    
    print("[*] Synthesizing Multi-Domain Intersections...")
    multi_data = generate_multi_domain_data(engrams, system_prompt, count=200)
    
    full_dataset = single_data + multi_data
    random.shuffle(full_dataset) # Shuffle to prevent chronological overfitting
    
    with open(OUTPUT_PATH, "w") as f:
        for data in full_dataset:
            f.write(json.dumps(data) + "\n")
            
    print(f"[*] Brainstem Lobotomy Dataset Complete. {len(full_dataset)} routing pairs generated at: {OUTPUT_PATH}")

if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    build_dataset()
