import torch
import torch.nn as nn
import torch.nn.functional as F

class OLoRALinear(nn.Module):
    """
    Phase 1: The Orthogonal LoRA Wrapper.
    Creates an isolated parallel highway for the memory gate.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        
        # 1. The Base Freeze
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.rank = rank
        self.scaling = alpha / rank
        
        # 2. The Isolated Memory Matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with strict mathematical orthogonality
        nn.init.orthogonal_(self.lora_A.weight)
        
        # Initialize B as zero so the module starts completely invisible to the network
        nn.init.zeros_(self.lora_B.weight)

    def get_orthogonal_penalty(self):
        """
        The OTITANS Shield. 
        Calculates how much the new memory weights overlap with the frozen base weights.
        We will add this to our loss later to force the memory into empty dimensions.
        """
        # Calculate the full Delta W matrix (B * A)
        delta_W = self.lora_B.weight @ self.lora_A.weight
        
        # Calculate cosine similarity between the base weights and the new memory weights
        # We flatten them to 1D to compare their overall directional vectors
        base_flat = self.base_layer.weight.view(-1)
        delta_flat = delta_W.view(-1)
        
        # The penalty is the absolute cosine similarity (0 = perfectly orthogonal, 1 = total overlap)
        penalty = torch.abs(F.cosine_similarity(base_flat, delta_flat, dim=0))
        return penalty

    def forward(self, x: torch.Tensor):
        # Pass 1: The frozen Gemma English syntax
        base_output = self.base_layer(x)
        
        # Pass 2: The parallel OTITANS memory logic
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        
        # Seamlessly merge the two isolated highways
        return base_output + lora_output
