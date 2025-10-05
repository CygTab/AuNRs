import torch
import wandb
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from data import DataLoader as DL
from Regressor import MultiOutputRegressor3l as MR3


# Initialize wandb project
wandb.init(project="CVV_Final", entity="gg1224")
config = wandb.config

config.data_path = './dataall.CSV'
config.train_samples = 1306  # Number of training samples
config.total_samples = 1633  # Total number of samples in dataset

# Model parameters
config.input_dim = 6
config.output_dim = 12
config.hid1 = 25
config.hid2 = 50
config.hid3 = 25

# Training parameters
config.batch_size = 256
config.test_batch_size = 500
config.epochs = 15000  # Total training epochs
config.lr = 5e-4
config.weight_decay = 1e-5
config.eps = 1e-8  # Epsilon to avoid division by zero in normalization

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE.upper()}")

dataset_loader = DL(config.data_path)
data = dataset_loader.load_data()

x = data.iloc[:, :config.input_dim]
y = data.iloc[:, config.input_dim:]
x_tensor = torch.tensor(x.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

train_idx = list(range(config.train_samples))
test_idx = list(range(config.train_samples, config.total_samples))

x_train = x_tensor[train_idx]
x_mean = x_train.mean(dim=0)  
x_std = x_train.std(dim=0) + config.eps  

x_train_norm = (x_train - x_mean) / x_std
x_test_norm = (x_tensor[test_idx] - x_mean) / x_std

y_train = y_tensor[train_idx]
y_test = y_tensor[test_idx]

train_dataset = TensorDataset(x_train_norm, y_train)
test_dataset = TensorDataset(x_test_norm, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.test_batch_size,
    shuffle=False,
    num_workers=0
)


def init_weights(model):
    """Initialize weights for linear layers using Kaiming normalization"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)

model = MR3(
    input_dim=config.input_dim,
    output_dim=config.output_dim,
    hid1=config.hid1,
    hid2=config.hid2,
    hid3=config.hid3
)
model.apply(init_weights)
model.to(DEVICE)  


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=config.weight_decay,
    amsgrad=False
)


def train(model, device, train_loader, optimizer, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets) 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {avg_loss:.6f}')
    wandb.log({"Train Loss": avg_loss})


def test(model, device, test_loader, epoch):
    """Testing loop for one epoch"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += nn.MSELoss()(outputs, targets).item()
    avg_loss = total_loss / len(test_loader)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{config.epochs}], Test Loss: {avg_loss:.6f}')
        save_path = f"./weights/model_{epoch}.pth"
        torch.save(model.state_dict(), save_path)
    wandb.log({"Test Loss": avg_loss})


for epoch in range(config.epochs):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)

final_model_path = 'final_model.pth'
torch.save(model.state_dict(), final_model_path)
wandb.save(final_model_path)