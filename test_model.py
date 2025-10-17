import torch
from src.model import CNN
from src import config

if __name__ == '__main__':
    model = CNN().to(config.DEVICE)
    dummy_input = torch.randn(4, 3, 32, 32).to(config.DEVICE)
    
    # Forward pass
    output = model(dummy_input)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model is on device: {next(model.parameters()).device}")