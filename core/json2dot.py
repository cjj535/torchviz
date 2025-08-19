import json
from typing import Dict, List

def graph_json_to_dot(json_data: List[Dict]) -> str:
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

def tree_json_to_dot(json_data: List[Dict]) -> str:
    dot_lines = [
        'digraph G {',
        '    rankdir=TB;',  # 图形方向：TB (Top-Bottom), LR (Left-Right)
        '    node [style=filled, fillcolor=lightgrey];'
    ]
    
    # 去除无子无父节点
    json_data = [node for node in json_data if node['parent'] or node['children']]

    # 添加所有节点
    node_dict = {node['id']: node for node in json_data}
    for node_id, node in node_dict.items():
        color = 'lightblue' if node['is_leaf'] else 'lightgrey'
        shape = 'ellipse' if node['is_leaf'] else 'box'
        
        label = (f"{node['name']}")
        
        dot_lines.append(
            f'    {node_id} [label="{label}", shape={shape}, fillcolor={color}];'
        )
    
    # 添加所有边
    for node in json_data:
        for child_id in node['children']:
            dot_lines.append(f'    {node["id"]} -> {child_id};')
    
    # 处理多根节点的情况
    roots = [node for node in json_data if node.get('parent') is None]
    if len(roots) > 1:
        dot_lines.append('    {rank=same; ' + '; '.join(str(r['id']) for r in roots) + ';}')
    
    dot_lines.append('}')
    
    return "\n".join(dot_lines)