import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler

# 定义两层全连接网络
class TwoLayerNet(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=128, output_size=10):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)   # 展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TwoLayerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    batch_size = 10

    def trace_handler(prof: torch.profiler.profile):
        file_name = "prof_result"
        # prof.export_chrome_trace(f"{file_name}.json")
        prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")
        import os
        file_path = f"{file_name}.html"
        try:
            os.remove(file_path)
            print(f"{file_path} has been delete.")
        except FileNotFoundError:
            print(f"{file_path} not exist.")
        except PermissionError:
            print(f"No permission to delete {file_path}.")

    # Profiler 配置
    prof = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        schedule=profiler.schedule(wait=0, warmup=2, active=1, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    # 训练并采集性能数据
    with prof:
        for epoch in range(3):
            images = torch.rand(batch_size, 1, 28, 28).to(device)
            labels = torch.randint(0, 10, (batch_size,)).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            prof.step()