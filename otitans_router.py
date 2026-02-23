import torch
import torch.nn as nn

class OTitansTriArchRouter(nn.Module):
    """
    Phase 3: The Tri-Arch Router.
    Dynamically routes forward passes between the frozen base model, 
    the Memory OTITANS gate, and potential Skill OTITANS gates.
    """
    def __init__(self, base_model, memory_gate, skill_gate=None):
        super().__init__()
        self.base_model = base_model
        
        # The Isolated Parallel Highways
        self.memory_gate = memory_gate
        self.skill_gate = skill_gate
        
        # Manual Alpha Controls for the Router
        # In a fully autonomous deployment, a lightweight classifier would set these
        self.current_memory_alpha = 1.0
        self.current_skill_alpha = 0.0

    def set_routing_alphas(self, memory_alpha: float, skill_alpha: float):
        """Dynamically adjust the routing gates before a forward pass."""
        self.current_memory_alpha = memory_alpha
        self.current_skill_alpha = skill_alpha
        # Suppress printing during rapid token generation, but useful for debugging
        # print(f"[*] Tri-Arch Routing Updated -> Memory: {memory_alpha} | Skill: {skill_alpha}")

    def forward(self, input_ids, **kwargs):
        # 1. Base Model Feature Extraction
        # Extract the hidden states directly from the frozen Gemma engine
        base_outputs = self.base_model(
            input_ids, 
            output_hidden_states=True, 
            return_dict=True,
            **kwargs
        )
        
        # Grab the residual stream from the final layer
        hidden_states = base_outputs.hidden_states[-1] 
        
        # 2. Dynamic Memory Routing
        if self.current_memory_alpha > 0.0 and self.memory_gate is not None:
            # Pass the stream through the isolated memory highway
            memory_states = self.memory_gate(hidden_states)
            # Blend it back into the stream based on the alpha weight
            hidden_states = hidden_states + (memory_states * self.current_memory_alpha)
            
        # 3. Dynamic Skill Routing (Your G-Code/Python logic lane)
        if self.current_skill_alpha > 0.0 and self.skill_gate is not None:
            skill_states = self.skill_gate(hidden_states)
            hidden_states = hidden_states + (skill_states * self.current_skill_alpha)
            
        # 4. Final Projection
        # Project the successfully blended hidden states back into the vocabulary logits
        logits = self.base_model.lm_head(hidden_states)
        
        return logits
