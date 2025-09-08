import json
from typing import Dict, List

forward_node_name = "[forward]"
backward_node_name = "[backward]"
postprocess_node_name = "[postprocess]"

class TensorInfo:
    producer: int
    comsumers: List[int]
    label: str

    def __init__(self, tensor: Dict):
        self.producer = -1
        self.comsumers = []
        self.label = f'{tensor["shape"]}'

def is_leaf(node: Dict) -> bool:
    return node["is_leaf"]

def get_tensor_key(tensor: Dict) -> str:
    return f'{tensor["id"]}_{tensor["version"]}_{tensor["device"]}'


# 增加三个虚拟的根节点，分别为forward、backward、postprocess，便于可视化时区分前向、反向、权重更新三个阶段
def preprocess_tree(tree_dict: Dict[int, Dict]) -> None:
    max_id = -1
    forward_root_node: Dict = {
        "id": -1,
        "name": forward_node_name,
        "start_time": -1,
        "end_time": -1,
        "is_leaf": False,
        "scope": "forward",
        "parent": None,
        "children": []
    }
    backward_root_node: Dict = {
        "id": -1,
        "name": backward_node_name,
        "start_time": -1,
        "end_time": -1,
        "is_leaf": False,
        "scope": "backward",
        "parent": None,
        "children": []
    }
    postprocess_root_node: Dict = {
        "id": -1,
        "name": postprocess_node_name,
        "start_time": -1,
        "end_time": -1,
        "is_leaf": False,
        "scope": "postprocess",
        "parent": None,
        "children": []
    }

    for id, node in tree_dict.items():
        max_id = max(max_id, id)
        if node["parent"] is not None:
            continue
        if node["scope"] == "backward":
            backward_root_node["start_time"] = min(backward_root_node["start_time"], node["start_time"]) if backward_root_node["start_time"] != -1 else node["start_time"]
            backward_root_node["end_time"] = max(backward_root_node["end_time"], node["end_time"]) if backward_root_node["end_time"] != -1 else node["end_time"]
            backward_root_node["children"].append(id)
        elif node["scope"] == "forward":
            forward_root_node["start_time"] = min(forward_root_node["start_time"], node["start_time"]) if forward_root_node["start_time"] != -1 else node["start_time"]
            forward_root_node["end_time"] = max(forward_root_node["end_time"], node["end_time"]) if forward_root_node["end_time"] != -1 else node["end_time"]
            forward_root_node["children"].append(id)
        elif node["scope"] == "postprocess":
            postprocess_root_node["start_time"] = min(postprocess_root_node["start_time"], node["start_time"]) if postprocess_root_node["start_time"] != -1 else node["start_time"]
            postprocess_root_node["end_time"] = max(postprocess_root_node["end_time"], node["end_time"]) if postprocess_root_node["end_time"] != -1 else node["end_time"]
            postprocess_root_node["children"].append(id)

    # 增加三个节点的id
    forward_root_node["id"] = max_id + 1
    backward_root_node["id"] = max_id + 2
    postprocess_root_node["id"] = max_id + 3
    for id, node in tree_dict.items():
        max_id = max(max_id, id)
        if node["parent"] is not None:
            continue
        if node["scope"] == "backward":
            node["parent"] = backward_root_node["id"]
        elif node["scope"] == "forward":
            node["parent"] = forward_root_node["id"]
        elif node["scope"] == "postprocess":
            node["parent"] = postprocess_root_node["id"]

    # 添加三个根节点
    tree_dict[forward_root_node["id"]] = forward_root_node
    tree_dict[backward_root_node["id"]] = backward_root_node
    tree_dict[postprocess_root_node["id"]] = postprocess_root_node

def delete_postprocess_node(tree_dict: Dict[int, Dict], leaf_node_dict: Dict[int, Dict]) -> None:
    postprocess_node_id: List[int] = []
    for id, node in tree_dict.items():
        if node["scope"] == "postprocess":
            postprocess_node_id.append(id)

    for key in postprocess_node_id:
        if key in tree_dict:
            del tree_dict[key]
    for key in postprocess_node_id:
        if key in leaf_node_dict:
            del leaf_node_dict[key]

def delete_backward_node(tree_dict: Dict[int, Dict], leaf_node_dict: Dict[int, Dict]) -> None:
    backward_node_id: List[int] = []
    for id, node in tree_dict.items():
        if node["scope"] == "backward":
            backward_node_id.append(id)

    for key in backward_node_id:
        if key in tree_dict:
            del tree_dict[key]
    for key in backward_node_id:
        if key in leaf_node_dict:
            del leaf_node_dict[key]

def json_to_complex_dot(graph_data: List[Dict], tree_data: List[Dict]) -> str:
    # 1.每条json数据以id为key
    leaf_node_map: Dict[int, Dict] = {n["id"]: n for n in graph_data}
    node_map: Dict[int, Dict] = {n["id"]: n for n in tree_data}

    # 添加三个虚拟根节点：forward、backward、postprocess
    preprocess_tree(node_map)

    # 删除图中的postprocess节点
    delete_postprocess_node(node_map, leaf_node_map)

    # 删除图中的backward节点
    delete_backward_node(node_map, leaf_node_map)

    # 对森林设置一个总的虚拟的根节点，id为-1，方便后续算法计算
    virtual_root_id = -1

    # 2.获取tensor的生产者和消费者节点id

    # 数据保证tensor最多只有一个生产者，如果没有生产者则设为-1，消费者可能有多个，如果没有消费者则为空list
    # tensor不会生产者、消费者均没有
    tensor_map: Dict[str, TensorInfo] = {}
    for node_id, node in leaf_node_map.items():
        for tensor in node["in_edges"]:
            tensor_key = get_tensor_key(tensor)
            if tensor_key not in tensor_map:
                tensor_map[tensor_key] = TensorInfo(tensor)
            tensor_map[tensor_key].comsumers.append(node_id)
        for tensor in node["out_edges"]:
            tensor_key = get_tensor_key(tensor)
            if tensor_key not in tensor_map:
                tensor_map[tensor_key] = TensorInfo(tensor)
            tensor_map[tensor_key].producer = node_id

    # 3.推导op属于哪个子图中
    def get_subgraph_id(op_id: int) -> int:
        # 无生产者
        if op_id == virtual_root_id:
            return virtual_root_id

        # 获取生产者的父节点id
        assert node_map[op_id]["parent"] is not None
        return node_map[op_id]["parent"]

    tensors_in_subgraph: Dict[int, List[str]] = {}
    for tensor_key, tensor_info in tensor_map.items():
        subgraph_id = get_subgraph_id(tensor_info.producer)         # 生产者决定tensor属于哪个子图
        if subgraph_id in tensors_in_subgraph:
            tensors_in_subgraph[subgraph_id].append(tensor_key)
        else:
            tensors_in_subgraph[subgraph_id] = [tensor_key]

    # 4.dfs遍历树并分析从属关系
    dot_lines: List[str] = []

    # dfs遍历函数，每次调用分析一个子图中应有的节点，包括tensor节点、op节点、子图
    def dfs(root_id: int, children: List[int], depth: int, ) -> List[str]:
        sub_dot_lines: List[str] = []
        for node_id in children:
            if is_leaf(node_map[node_id]):
                # 添加op
                sub_dot_lines.append(f'{"    "*depth}"node_{node_id}" [label="{node_map[node_id]["name"]}", shape=box];')
            else:
                # 添加子图
                sub_dot_lines.append(f'{"    "*depth}subgraph cluster_{node_id} {{')
                # if node_map[node_id]["scope"] == "forward":
                #     sub_dot_lines.append(f'{"    "*(depth+1)}rankdir=LR;')
                # elif node_map[node_id]["scope"] == "backward":
                #     sub_dot_lines.append(f'{"    "*(depth+1)}rankdir=LR;')
                sub_dot_lines.append(f'{"    "*(depth+1)}label="{node_map[node_id]["name"]}";')
                sub_dot_lines.append(f'{"    "*(depth+1)}style=rounded;')
                sub_dot_lines.append(f'{"    "*(depth+1)}color=blue;')
                sub_dot_lines += dfs(node_id, node_map[node_id]["children"], depth+1)
                sub_dot_lines.append(f'{"    "*depth}}}')

        # 添加tensor节点
        for tensor_key in tensors_in_subgraph.get(root_id, []):
            sub_dot_lines.append(f'{"    "*depth}"tensor_{tensor_key}" [label="{tensor_map[tensor_key].label}", shape=ellipse];')

        return sub_dot_lines

    root_nodes = [k for k, v in node_map.items() if v["parent"] is None]
    dot_lines = dfs(virtual_root_id, root_nodes, depth=1)

    # 5.添加边
    for subgraph_id, tensor_key_list in tensors_in_subgraph.items():
        for tensor_key in tensor_key_list:
            producer_id = tensor_map[tensor_key].producer
            if producer_id != virtual_root_id:
                dot_lines.append(f'{"    "}"node_{producer_id}" -> "tensor_{tensor_key}";')
            for comsumer_id in tensor_map[tensor_key].comsumers:
                dot_lines.append(f'{"    "}"tensor_{tensor_key}" -> "node_{comsumer_id}";')

    # 6.定义dot文件头尾，完成组装
    root_dot_lines: List[str] = []
    root_dot_lines.append("digraph G {")
    root_dot_lines.append('    rankdir=LR;')              # 从左到右绘制
    root_dot_lines.append('    node [fontname="Arial"];')
    root_dot_lines.append("}")
    result = "\n".join(root_dot_lines[:-1] + dot_lines + root_dot_lines[-1:])

    return result
