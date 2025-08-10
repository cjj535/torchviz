import json
import networkx as nx
import matplotlib.pyplot as plt

# 从 JSON 文件中读取数据
with open('./sample/sample_result/graph_data.json', 'r') as f:
    data_list = json.load(f)

# 用来存储所有节点的ID，方便后续匹配边的目标节点
node_data = {}

# 存储所有的边信息，字典的 key 为 (edge_id, version)，value 为 边的 label 和两个端点 id
edges_dict = {}

# 解析每个节点的数据并构建字典
def process_edges(data_list):
    for data in data_list:
        node_id = data['id']
        
        # 处理入边
        for edge in data['in_edges']:
            edge_id = edge['id']
            version = edge['version']
            device = edge['device']
            size = edge['size']
            category = edge['category']
            
            # 边的标签
            # edge_label = f"device: {device}, size: {size}, category: {category}"
            edge_label = f"{size}"
            
            # 在字典中添加该边信息，key 为 (edge_id, version)，value 为 边的 label 和两个端点 id
            key = (edge_id, version)

            if key not in edges_dict:
                edges_dict[key] = {
                    'edge_label': edge_label,
                    'left_id': [],  # 入边的右端点是该节点的ID
                    'right_id': []     # 暂时设置为 None，待后续出边处理
                }
            
            edges_dict[key]['right_id'].append(node_id)  # 将该节点作为边的右端点

        # 处理出边
        for edge in data['out_edges']:
            edge_id = edge['id']
            version = edge['version']
            device = edge['device']
            size = edge['size']
            category = edge['category']
            
            # 边的标签
            # edge_label = f"device: {device}, size: {size}, category: {category}"
            edge_label = f"{size}"
            
            # 在字典中添加该边信息，key 为 (edge_id, version)，value 为 边的 label 和两个端点 id
            key = (edge_id, version)
            if key not in edges_dict:
                edges_dict[key] = {
                    'edge_label': edge_label,
                    'left_id': [],  # 出边的左端点是该节点的ID
                    'right_id': []     # 暂时设置为 None，待后续入边处理
                }
            
            edges_dict[key]['left_id'].append(node_id)  # 将该节点作为边的左端点

# 解析每个节点的数据并构建图
def create_dot(data_list):
    dot_lines = ["digraph G {"]

    # 将节点数据存入字典，方便后续查找目标节点
    for data in data_list:
        node_id = data['id']
        node_name = data['name']
        in_edges = data['in_edges']
        out_edges = data['out_edges']

        # 添加节点，节点属性包含名称和是否反向
        shape = 'box'
        dot_lines.append(f'    {node_id} [shape={shape}, label="{node_name}"]')

    # 为每个节点添加边，确定目标节点
    for data in data_list:
        node_id = data['id']
        in_edges = data['in_edges']
        out_edges = data['out_edges']

        # 添加入边
        for edge in in_edges:
            edge_id = edge['id']
            version = edge['version']
            
            # 搜索是否有其他节点包含这条边的相同id和version
            key = (edge_id, version)
            edge_info = edges_dict[key]
            # 没有源点和终点的边暂时不展示
            for target_node_id in edge_info['left_id']:
                label = edge_info['edge_label']
                dot_lines.append(f'    {target_node_id} -> {node_id} [label="{label}"]')

        # # 添加出边
        # for edge in out_edges:
        #     edge_id = edge['id']
        #     version = edge['version']
            
        #     key = (edge_id, version)
        #     edge_info = edges_dict[key]
        #     # 没有源点和终点的边暂时不展示
        #     for target_node_id in edge_info['right_id']:
        #         label = edge_info['edge_label']
        #         dot_lines.append(f'    {node_id} -> {target_node_id} [label="{label}"]')
    
    dot_lines.append("}")
    return "\n".join(dot_lines)

process_edges(data_list)
# 调用函数来创建图
dot_output = create_dot(data_list)

def save_dot_file(dot_lines, filename="output.dot"):
    with open(filename, "w") as file:
        file.write(dot_lines)

# 保存为 .dot 文件
save_dot_file(dot_output, "./sample/sample_result/graph.dot")