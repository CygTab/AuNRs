import torch
import wandb
import math
import torch.nn as nn
from scipy.stats import zscore
from torch.utils.data import DataLoader, TensorDataset, Subset
from data import DataLoader as DL
from Regressor import MultiOutputRegressor3l as MR3

wandb.init(project="CVV_Final", entity="gg1224")
config = wandb.config
config.batch_size = 256
config.test_batch_size = 500
config.epoch = 15000
config.lr = 5e-4
config.weight_decay = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())


dataset = DL('./dataall.CSV')
data = dataset.load_data()
x = data.iloc[:, :6]
y = data.iloc[:, 6:]

x_data = torch.tensor(x.values, dtype=torch.float32)
y_data = torch.tensor(y.values, dtype=torch.float32)

indice = list(range(1633))
train_idx = indice[:1306]
test_idx = indice[1306:]
dataset = TensorDataset(x_data, y_data)

train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=config.test_batch_size, shuffle=False, num_workers=0)


def init_weights(model):
    for m in model.modules():
        if isinstance(m,nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)


model = MR3(input_dim=6, output_dim=12, hid1=25, hid2=50, hid3=25)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.weight_decay, amsgrad=False)


def train(model, DEVICE, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        criterion = nn.MSELoss()
        loss = criterion(outputs, targets)
        loss_sum = loss_sum + loss.item()
        loss.backward()
        optimizer.step()
    loss_sum = loss_sum/math.ceil(1389/config.batch_size)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}], Train Loss: {loss_sum:.6f}')
    wandb.log({
        "Train Loss": loss_sum
    })


def test(model, DEVICE, test_loader,epoch):
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            criterion = nn.MSELoss()
            loss = criterion(outputs, targets)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}], Ttest Loss: {loss.item():.6f}')
            str = f"./weights/model{epoch}.pth"
            torch.save(model.state_dict(), str)
    wandb.log({
        "Test Loss": loss
    })


for epoch in range(1, config.epoch):
    train(model, DEVICE, train_loader,optimizer, epoch)
    test(model, DEVICE, test_loader, epoch)
torch.save(model.state_dict(),'model.pth')
wandb.save('model.pth')
