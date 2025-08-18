import json
from typing import Dict, List

def generate_dot_file(json_file: str, output_dot: str = 'tree.dot'):
    """
    从JSON文件生成Graphviz DOT文件
    
    :param json_file: 包含节点数据的JSON文件路径
    :param output_dot: 输出的DOT文件路径
    """
    with open(json_file) as f:
        nodes = json.load(f)
    
    dot_lines = [
        'digraph G {',
        '    rankdir=TB;',  # 图形方向：TB (Top-Bottom), LR (Left-Right)
        '    node [style=filled, fillcolor=lightgrey];'
    ]
    
    # 添加所有节点
    node_dict = {node['id']: node for node in nodes}
    for node_id, node in node_dict.items():
        color = 'lightblue' if node['is_leaf'] else 'lightgrey'
        shape = 'ellipse' if node['is_leaf'] else 'box'
        
        label = (f"{node['name']}")
        
        dot_lines.append(
            f'    {node_id} [label="{label}", shape={shape}, fillcolor={color}];'
        )
    
    # 添加所有边
    for node in nodes:
        for child_id in node['children']:
            dot_lines.append(f'    {node["id"]} -> {child_id};')
    
    # 处理多根节点的情况
    roots = [node for node in nodes if node.get('parent') is None]
    if len(roots) > 1:
        dot_lines.append('    {rank=same; ' + '; '.join(str(r['id']) for r in roots) + ';}')
    
    dot_lines.append('}')
    
    # 写入DOT文件
    with open(output_dot, 'w') as f:
        f.write('\n'.join(dot_lines))
    
    print(f"DOT文件已生成: {output_dot}")

# 使用示例
generate_dot_file('./sample/sample_result/tree_data.json', './sample/sample_result/tree.dot')

# 将dot文件可视化
import subprocess
dot_file = "./sample/sample_result/tree.dot"
png_file = "./sample/sample_result/tree.png"
subprocess.run(['dot', '-Tpng', '-Gdpi=300', "-Nfontname=Helvetica", "-Nfontsize=10", dot_file, '-o', png_file], check=True)
print("Generated ./sample/sample_result/tree.png")