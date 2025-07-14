import torch
from torchvision import datasets, transforms

def load_data(data_dir="./data"):
    """Load MNIST data with shared download location"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, transform=transform)

    return train_data, test_data

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters): # Added 'model' as first argument
    """Handle both numpy arrays and tensors"""
    params_dict = zip(model.state_dict().keys(), parameters) # Use passed 'model'
    state_dict = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                for k, v in params_dict}
    model.load_state_dict(state_dict)