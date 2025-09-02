import json
from core.json2dot import graph_json_to_dot, tree_json_to_dot
from core.json_to_complex_dot import json_to_complex_dot
from core.dot2png import dot_to_png
import argparse

def main(model = 'DNN', is_train = 1, is_generate_tree = 1, is_graph = 1):
    if is_train:
        # 劫持profiler函数
        from hijack_function.hijack_profiler import hijack_profiler
        hijack_profiler(model, is_generate_tree)

        # 跑训练过程，获取原始的json格式数据
        if model == 'DNN':
            from examples.DNN.model import train
            train()
        elif model == 'ResNet':
            from examples.ResNet.model import train
            train()

    # 将原始数据转为dot文件
    with open(f'./data/{model}/graph.json', 'r') as json_file:
        data = json.load(json_file)
        dot_content = graph_json_to_dot(data)

        with open(f'./data/{model}/graph.dot', "w") as dot_file:
            dot_file.write(dot_content)
            print(f"Transfered ./data/{model}/graph.json to ./data/{model}/graph.dot")

    if is_generate_tree:
        with open(f'./data/{model}/tree.json', 'r') as json_file:
            data = json.load(json_file)
            dot_content = tree_json_to_dot(data)

            with open(f'./data/{model}/tree.dot', "w") as dot_file:
                dot_file.write(dot_content)
                print(f"Transfered ./data/{model}/tree.json to ./data/{model}/tree.dot")
        
        with open(f'./data/{model}/graph.json', 'r') as graph_json_file, open(f'./data/{model}/tree.json', 'r') as tree_json_file:
            graph_data = json.load(graph_json_file)
            tree_data = json.load(tree_json_file)
            dot_content = json_to_complex_dot(graph_data, tree_data)

            with open(f'./data/{model}/complex_graph.dot', "w") as dot_file:
                dot_file.write(dot_content)
                print(f"Generated ./data/{model}/complex_graph.dot")
    
    # 将dot文件可视化
    if is_graph:
        dot_to_png(f'./data/{model}', is_generate_tree)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model_name, train or not, generate tree")

    parser.add_argument("--model", type=str, help="model name", required=True)
    parser.add_argument("--train", type=int, help="1 train, 0 not train")
    parser.add_argument("--tree", type=int, help="1 generate tree, 0 not")
    parser.add_argument("--graph", type=int, help="1 generate png, 0 not")

    args = parser.parse_args()

    main(args.model, args.train, args.tree, args.graph)