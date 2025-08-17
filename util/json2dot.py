import json

def json_to_dot(json_data):
    dot_lines = []
    dot_lines.append("digraph G {")
    dot_lines.append('  rankdir=LR;')  # 从左到右绘制
    dot_lines.append('  node [fontname="Arial"];')

    tensor_nodes = set()  # 用来去重 (id, version)

    for node in json_data:
        op_id = f"op_{node['id']}"
        op_label = node['name']
        dot_lines.append(f'  "{op_id}" [label="{op_label}", shape=box];')

        # 输入边
        for t in node.get("in_edges", []):
            tensor_key = (t["id"], t["version"])
            tensor_id = f"tensor_{t['id']}_v{t['version']}"
            if tensor_key not in tensor_nodes:
                tensor_nodes.add(tensor_key)
                dot_lines.append(f'  "{tensor_id}" [label="{t["shape"]}", shape=ellipse];')
            # tensor -> op
            dot_lines.append(f'  "{tensor_id}" -> "{op_id}";')

        # 输出边
        for t in node.get("out_edges", []):
            tensor_key = (t["id"], t["version"])
            tensor_id = f"tensor_{t['id']}_v{t['version']}"
            if tensor_key not in tensor_nodes:
                tensor_nodes.add(tensor_key)
                dot_lines.append(f'  "{tensor_id}" [label="{t["shape"]}", shape=ellipse];')
            # op -> tensor
            dot_lines.append(f'  "{op_id}" -> "{tensor_id}";')

    dot_lines.append("}")
    return "\n".join(dot_lines)

def transfer_json_to_dot():
    with open('./sample/sample_result/graph_data.json', 'r') as file:
        data = json.load(file)
        dot_content = json_to_dot(data)

    with open("./sample/sample_result/graph.dot", "w") as file:
        file.write(dot_content)

    print("Generated ./sample/sample_result/graph.dot")
