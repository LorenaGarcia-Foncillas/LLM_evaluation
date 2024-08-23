import torch
import torch.nn as nn
import torch.nn.functional as F

# LoRA 
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):

        """
        LoRA (Low-Rank Adaptation) layer implementation.

        Args:
        - in_dim (int): Input dimension size.
        - out_dim (int): Output dimension size.
        - rank (int): Rank of the low-rank matrix approximation. Hyperparameter that controls the inner 
            dimension of the matrices A and B, i.e. controls the additional parameters introduced by LoRA
        - alpha (float): Scaling factor for the LoRA operation. Scaling hyperparameter applied to the 
            output of the low-rank adaptation. It controls the extent to which the adapted layer's output 
            is allowed to influence the original output of the layer being adapted (i.e. a way to regulate
            the impact of the low-rank adaptation on the layer's output)
        """
       
        super(LoRALayer, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim)) 
        self.alpha = alpha

    def forward(self, x):

        """
        Forward pass of the LoRA layer.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
        - torch.Tensor: Output tensor after applying LoRA operation.
        """

        x = self.alpha * torch.matmul(x, torch.matmul(self.A, self.B)) #(x @ self.A @ self.B) # @: matrix multiplication
        return x
    
class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):

        """
        Linear layer with added LoRA (Low-Rank Adaptation) operation.

        Args:
        - linear (nn.Linear): Base linear layer.
        - rank (int): Rank of the low-rank matrix approximation.
        - alpha (float): Scaling factor for the LoRA operation.
        """

        # super().__init__()
        super(LinearWithLoRA, self).__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):

        """
        Forward pass of the LinearWithLoRA module. Equivalent to: xW + xAB

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, linear.in_features).

        Returns:
        - torch.Tensor: Output tensor after applying the linear layer followed by LoRA.
        """
        
        # Apply the base linear layer
        linear_output = self.linear(x)
        # Apply the LoRA layer
        lora_output = self.lora(x)
        
        # Return the sum of the outputs from the linear and LoRA layers
        return linear_output + lora_output
    
class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):

        """
        Linear layer with merged LoRA (Low-Rank Adaptation) and original weights.
        Equivalent to: x(W+AB)

        Args:
        - linear (nn.Linear): Base linear layer.
        - rank (int): Rank of the low-rank matrix approximation.
        - alpha (float): Scaling factor for the LoRA operation.
        """

        # super().__init__()
        super(LinearWithLoRAMerged, self).__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        """
        Forward pass of the LinearWithLoRAMerged module.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, linear.in_features).

        Returns:
        - torch.Tensor: Output tensor after applying the linear layer with merged weights.
        """
        
        # Combine LoRA matrices
        lora = self.lora.A @ self.lora.B 
        # Then combine LoRA with orig. weights
        combined_weight = self.linear.weight + self.lora.alpha*lora.T 

        return F.linear(x, combined_weight, self.linear.bias)
 

def freeze_linear_layers(model):
    """
    Freezes all linear layers (nn.Linear) in the provided model by setting requires_grad to False.

    Args:
    - model (nn.Module): The model whose linear layers need to be frozen.
    """
    for child in model.children():
        if isinstance(child, nn.Linear):
            # Freeze parameters of the linear layer
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)


# DoRA
class LinearWithDoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        """
        Initialize the LinearWithDoRA module.

        Args:
        - linear (nn.Linear): The original linear layer.
        - rank (int): Rank parameter for LoRALayer.
        - alpha (float): Alpha parameter for LoRALayer.
        """
        super(LinearWithDoRA, self).__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        # Initialize the parameter 'm' as the L2 norm of the original linear layer weights
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        """
        Forward pass of the LinearWithDoRA module.

        Args:
        - x (torch.Tensor): Input tensor to be passed through the linear layer.

        Returns:
        - torch.Tensor: Output tensor after passing through the modified linear layer.
        """
        # Apply the base linear layer
        linear_output = self.linear(x)

        # Calculate LoRA transformation
        lora = self.lora.A @ self.lora.B

        # Compute numerator of the modified weight
        numerator = self.linear.weight + self.lora.alpha * lora.T

        # Compute denominator for normalization
        denominator = numerator.norm(p=2, dim=0, keepdim=True)

        # Compute directional component (normalized numerator)
        directional_component = numerator / denominator

        # Multiply by parameter 'm' to get the new weight
        new_weight = self.m * directional_component

        # Apply the linear transformation with the new weight and bias
        dora_output = F.linear(x, new_weight, self.linear.bias)

        # Return the sum of the outputs from the linear and DoRA layers
        return linear_output + dora_output


class LinearWithDoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        """
        Initialize the LinearWithDoRAMerged module.

        Args:
        - linear (nn.Linear): The original linear layer whose weights will be modified.
        - rank (int): Rank parameter for LoRALayer.
        - alpha (float): Alpha parameter for LoRALayer.
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
        # Initialize the parameter 'm' as the L2 norm of the original linear layer weights
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x):
        """
        Forward pass of the LinearWithDoRAMerged module.

        Args:
        - x (torch.Tensor): Input tensor to be passed through the linear layer.

        Returns:
        - torch.Tensor: Output tensor after passing through the modified linear layer.
        """
        # Calculate LoRA transformation
        lora = self.lora.A @ self.lora.B

        # Compute numerator of the modified weight
        numerator = self.linear.weight + self.lora.alpha * lora.T

        # Compute denominator for normalization
        denominator = numerator.norm(p=2, dim=0, keepdim=True)

        # Compute directional component (normalized numerator)
        # Ensures that each column of the combined weight matrix has a unit norm, 
        # which can help stabilize the learning process by maintaining the scale of weight updates
        directional_component = numerator / denominator

        # Multiply by parameter 'm' to get the new weight
        new_weight = self.m * directional_component

        # Apply the linear transformation with the new weight and bias
        return F.linear(x, new_weight, self.linear.bias)
    
"""
HOW TO USE NOTES:

# Load the MODEL
model = DistilBertForTokenClassification.from_pretrained(distilbert_directory, config=config)


"""

