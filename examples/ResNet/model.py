import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler as profiler
from torchvision.models import resnet18

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 64
    # 定义ResNet模型（调整为CIFAR-10的32x32输入）
    def get_resnet():
        model = resnet18(pretrained=False)
        # 修改第一层卷积（原始ResNet是为224x224设计的）
        model.conv1 = nn.Conv2d(3, batch_size, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改最后的全连接层
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model

    model = get_resnet().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

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
            images = torch.rand(batch_size, 3, 32, 32).to(device)
            labels = torch.randint(0, 10, (batch_size,)).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()

            prof.step()