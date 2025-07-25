import torch
import pandas as pd
from scipy.stats import zscore
from torch.utils.data import DataLoader, TensorDataset, Subset
from data import DataLoader as DL
from Regressor import MultiOutputRegressor3l as MR3
import time
import timeit
DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())


def z_score_normalize(data):
    normalized_data = zscore(data)
    return normalized_data

dataset = DL('./val3.CSV')
data = dataset.load_data()
x = data.iloc[:, :6]
y = data.iloc[:, 6:]

x = z_score_normalize(x)
y = z_score_normalize(y)

x_data = torch.tensor(x.values, dtype=torch.float32)
y_data = torch.tensor(y.values, dtype=torch.float32)

dataset = TensorDataset(x_data, y_data)
test_loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=0)

model = MR3(input_dim=6, output_dim=12, hid1=25, hid2=50, hid3=25)
weight = torch.load('model.pth')
model.load_state_dict(weight)
model.to(DEVICE)
input_tensor1 = torch.tensor([3.928571429, 407.16, 18.76333333, 13, 55, 14], dtype=torch.float32)
input_tensor2 = torch.tensor([3, 339.3, 17.6625, 9, 66, 22], dtype=torch.float32)
input_tensor3 = torch.tensor([4.166666667, 271.44, 11.58545455, 12, 50, 12], dtype=torch.float32)
model.eval()

with torch.no_grad():
    for _ in range(10):
        model(input_tensor1)

num_runs = 10000  
start_time = time.perf_counter()

with torch.no_grad(): 
    for _ in range(num_runs):
        model(input_tensor1)

end_time = time.perf_counter()
total_time = end_time - start_time
avg_time = total_time / num_runs

print(f"average time: {avg_time * 1000:.6f} ms")
