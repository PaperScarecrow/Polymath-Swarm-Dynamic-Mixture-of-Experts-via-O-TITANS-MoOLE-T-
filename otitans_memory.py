import torch
import torch.nn as nn
import torch.nn.functional as F
from otitans_core import OLoRALinear

class OTitansMemoryGate(nn.Module):
    """
    Phase 2: The OTITANS Memory Core.
    A recurrent memory state shielded by orthogonal LoRA projections.
    """
    def __init__(self, hidden_size: int, rank: int = 8, memory_momentum: float = 0.9):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_momentum = memory_momentum
        
        # 1. The Orthogonal Projections
        # We use standard nn.Linear here as placeholders, but in the actual injection script,
        # we will map these directly to Gemma's layers wrapped in our OLoRALinear class.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 2. The Memory Gate
        # A learned parameter that decides how much to trust the recurrent memory vs the base attention.
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )
        
        # 3. The Persistent State
        # This is where Nyxxie's continuous memory lives. 
        self.register_buffer("memory_state", torch.zeros(hidden_size, hidden_size))

    def reset_memory(self):
        """Wipes the recurrent memory clean for a new session."""
        self.memory_state.zero_()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Generate Queries, Keys, and Values through the orthogonal pathways
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        memory_outputs = []
        
        # The Recurrent Engine (Autoregressive Delta Rule Update)
        # Note: In training, we will parallelize this. For inference, it processes step-by-step.
        current_memory = self.memory_state.clone()
        
        for t in range(seq_len):
            q_t = q[:, t, :]  # Current query
            k_t = k[:, t, :]  # Current key
            v_t = v[:, t, :]  # Current value
            
            # Read from the current memory state
            # Retrieval = Q * Memory
            retrieval = torch.matmul(q_t.unsqueeze(1), current_memory).squeeze(1)
            memory_outputs.append(retrieval)
            
            # Update the memory state using the Surprise / Delta mechanism
            # How much does the new Key/Value differ from what we already know?
            memory_prediction = torch.matmul(k_t.unsqueeze(1), current_memory).squeeze(1)
            surprise = v_t - memory_prediction
            
            # Update: M_t = momentum * M_{t-1} + (Surprise ⊗ Key)
            update = torch.bmm(surprise.unsqueeze(2), k_t.unsqueeze(1))
            current_memory = (self.memory_momentum * current_memory) + update
            
        # Stack the memory retrievals back into the sequence shape
        memory_out_tensor = torch.stack(memory_outputs, dim=1)
        
        # Save the updated memory state for the next generation step
        self.memory_state.copy_(current_memory.detach())
        
        # Calculate the Gating mechanism: How much should we blend memory with standard logic?
        # We concatenate the base hidden states with the memory retrieval to decide.
        gate_input = torch.cat([hidden_states, memory_out_tensor], dim=-1)
        gate_value = self.gate(gate_input)
        
        # Return the gated memory logic
        return hidden_states + (gate_value * memory_out_tensor)
