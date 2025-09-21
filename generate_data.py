import json
from core.json_to_complex_json import json_to_complex_json
import argparse

def main(model = 'DNN'):
    # 劫持profiler函数
    from hijack_function.hijack_profiler import hijack_profiler
    hijack_profiler(model)

    # 跑训练过程，获取原始的json格式数据
    if model == 'DNN':
        from examples.DNN.model import train
        train()
    elif model == 'ResNet':
        from examples.ResNet.model import train
        train()
        
    with open(f'./data/{model}/graph.json', 'r') as graph_json_file, open(f'./data/{model}/tree.json', 'r') as tree_json_file:
        graph_data = json.load(graph_json_file)
        tree_data = json.load(tree_json_file)
        json_content = json_to_complex_json(graph_data, tree_data)

        with open(f'./data/{model}/complex_graph.json', "w") as json_file:
            json.dump(json_content, json_file, indent=4)
            print(f"Generated ./data/{model}/complex_graph.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model_name")

    parser.add_argument("--model", type=str, help="model name", required=True)

    args = parser.parse_args()

    main(args.model)