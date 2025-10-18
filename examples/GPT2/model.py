# import包
import torch
import torch.profiler as profiler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader


# 创建数据集
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def create_dataset(model_path, data_path):
    # 加载GPT-2的分词器
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # 假设你有一个文本文件，每行是一个样本
    with open(data_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    tokenizer.pad_token = tokenizer.eos_token   # 将 eos_token 设置为 pad_token
    encodings = tokenizer(                      # 将文本转换为token IDs
        texts,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )
    dataset = TextDataset(encodings)            # 制作训练数据
    return dataset


def train():
    model_path = "./gpt2_source/gpt2"
    data_path = "./gpt2_source/data/sample.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained(model_path)                             # 加载预训练的GPT-2模型
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)                       # 定义优化器

    dataset = create_dataset(model_path, data_path)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)                 # 创建DataLoader

    model.train()

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

    with prof:
        for epoch in range(3):
            for batch in train_loader:
                optimizer.zero_grad()                                                       # 梯度重置为0
                input_ids = batch['input_ids'].to(device)                                   # 模型输入
                attention_mask = batch['attention_mask'].to(device)                         # 设置mask
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids) # 前向计算
                loss = outputs.loss
                loss.backward()                                                             # 反向计算
                optimizer.step()                                                            # 梯度更新
            prof.step()