import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
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

    # 数据加载（MNIST）
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = TwoLayerNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    def trace_handler(prof: torch.profiler.profile):
        file_name = "prof_result"
        prof.export_memory_timeline(f"{file_name}.html", device="cuda:0")

    # Profiler 配置
    prof = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA  # 如果是 CPU 训练可以去掉
        ],
        schedule=profiler.schedule(wait=0, warmup=2, active=1, repeat=1),  # 等待1步，热身1步，采集3步
        on_trace_ready=trace_handler,
        # on_trace_ready=profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    # 训练并采集性能数据
    with prof:
        for epoch in range(1):  # 演示只跑 1 个 epoch
            total_loss = 0
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize()  # 确保 GPU 任务完成再计时
                total_loss += loss.item()

                prof.step()  # 通知 profiler 进入下一步（必须）
            print(f"Epoch [{epoch+1}], Loss: {total_loss/len(train_loader):.4f}")

# 查看 TensorBoard：
# tensorboard --logdir=./log
# 然后打开 http://localhost:6006