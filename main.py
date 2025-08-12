# 跑训练过程，获取原始的json格式数据
from sample.train import train
train()

# 将原始数据转为dot文件
from util.json2dot import transfer_json_to_dot
transfer_json_to_dot()

# 将dot文件可视化
import subprocess
dot_file = "./sample/sample_result/graph.dot"
png_file = "./sample/sample_result/graph.png"
subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)
print("Generated ./sample/sample_result/graph.png")