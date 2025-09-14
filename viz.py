from typing import Optional, Dict, List, Tuple
import json
import copy
import subprocess


origin_nodes: Dict[int, "Node"] = {}


class Node:
    id: int
    isTensor: bool
    isLeaf: bool
    label: str
    parent: Optional[int]
    children: List[int]
    nextNodes: List[int]
    isCollapse: bool

    # layout
    x: int
    y: int
    rx: int
    ry: int

    def __init__(self, node_json: Optional[Dict] = None):
        if node_json:
            self.id = node_json["id"]
            self.isTensor = node_json["isTensor"]
            self.isLeaf = node_json["isLeaf"]
            self.label = node_json["label"]
            self.parent = node_json["parent"]
            self.children = node_json["children"]
            self.nextNodes = node_json["nextNodes"]
        else:
            self.id = 0
            self.isTensor = False
            self.isLeaf = False
            self.label = ""
            self.parent = None
            self.children = []
            self.nextNodes = []

        self.isCollapse = True          # 默认均折叠

        self.x = 0
        self.y = 0
        self.rx = 0
        self.ry = 0


def ReadFile(path: str):
    with open(path, 'r') as graph_json_file:
        nodes_json = json.load(graph_json_file)

    for node in nodes_json:
        origin_nodes[node["id"]] = Node(node)


# 每次更新状态后，重新计算坐标点
def ComputeLayout(nodes: Dict[int, Node]):
    # compute x

    # compute y

    return


def GetOutTensorsInCollapseNode(root_node_id: int) -> List[int]:
    # 记录需要调整从属关系的tensor
    tensor_id_list: List[int] = []

    # 判断边的目的节点是否属于当前子树
    def IsInRoot(node_id: Optional[int]) -> bool:
        if node_id is None:
            return False
        elif node_id == root_node_id:
            return True
        else:
            return IsInRoot(origin_nodes[node_id].parent)
    
    # 遍历子树中有无tensor作为其他子树的输入
    def dfs(node_id: int) -> None:
        if origin_nodes[node_id].isLeaf:
            if origin_nodes[node_id].isTensor:
                if any([not IsInRoot(id) for id in origin_nodes[node_id].nextNodes]):
                    tensor_id_list.append(node_id)
        else:
            for id in origin_nodes[node_id].children:
                dfs(id)
        return

    dfs(root_node_id)

    return tensor_id_list


# 生成新的图
def GenerateNewGraph(nodes: Dict[int, Node]) -> Dict[int, Node]:
    new_nodes: Dict[int, Node] = {}

    # 1.获取根节点id
    roots_id: List[int] = [id for id, node in nodes.items() if node.parent is None]

    # 2.更新节点在树中和图中位置
    def DfsUpdateNode(node_id: int) -> List[int]:
        if nodes[node_id].isLeaf:
            new_nodes[node_id] = copy.deepcopy(nodes[node_id])          # 还有nextNodes未更新（指向折叠的子图中）
        else:
            if nodes[node_id].isCollapse:
                new_nodes[node_id] = copy.deepcopy(nodes[node_id])
                new_nodes[node_id].isLeaf = True
                new_nodes[node_id].children = []
                new_nodes[node_id].nextNodes = GetOutTensorsInCollapseNode(node_id) # 更新折叠节点的nextNodes
                return copy.deepcopy(new_nodes[node_id].nextNodes)
            else:
                new_nodes[node_id] = copy.deepcopy(nodes[node_id])      # 未更新children
                new_children: List[int] = []
                for id in nodes[node_id].children:
                    new_children += DfsUpdateNode(id)
                
                # 更改tensor节点在树中的位置
                for id in new_children:
                    new_nodes[id] = copy.deepcopy(nodes[id])
                    new_nodes[id].parent = node_id
                
                new_nodes[node_id].children += new_children             # 更新children
        return []
    
    new_children: List[int] = []
    for id in roots_id:
        new_children += DfsUpdateNode(id)
    for id in new_children:
        new_nodes[id] = copy.deepcopy(nodes[id])
        new_nodes[id].parent = None
        roots_id.append(id)

    # 3.更新边的指向
    def GetAncestorId(id: Optional[int]) -> Optional[int]:
        if id is None:
            return None
        if id in new_nodes:
            return id
        else:
            return GetAncestorId(nodes[id].parent)

    def DfsUpdateEdge(node_id: int) -> None:
        for i in range(len(new_nodes[node_id].nextNodes)):
            if new_nodes[node_id].nextNodes[i] not in new_nodes:
                new_next_id = GetAncestorId(new_nodes[node_id].nextNodes[i])
                assert new_next_id is not None
                if new_next_id is not None:
                    new_nodes[node_id].nextNodes[i] = new_next_id
        for id in new_nodes[node_id].children:
            DfsUpdateEdge(id)

    for id in roots_id:
        DfsUpdateEdge(id)
    
    # 4.返回新的图
    return new_nodes


def GenerateDot(nodes: Dict[int, Node]) -> str:
    node_dot_lines: List[str] = []
    edges_dot_lines: List[str] = []

    # dfs遍历函数，每次调用分析一个子图中应有的节点，包括tensor节点、op节点、子图
    def dfsGenerateDot(children: List[int], depth: int) -> List[str]:
        sub_dot_lines: List[str] = []
        for node_id in children:
            if nodes[node_id].isLeaf:
                # 添加op
                shape = "ellipse" if nodes[node_id].isTensor else "box"
                sub_dot_lines.append(f'{"    "*depth}"{node_id}" [label="{nodes[node_id].label}", shape={shape}];')
                for id in nodes[node_id].nextNodes:
                    edges_dot_lines.append(f'{"    "}"{node_id}" -> "{id}";')
            else:
                # 添加子图
                sub_dot_lines.append(f'{"    "*depth}subgraph cluster_{node_id} {{')
                sub_dot_lines.append(f'{"    "*(depth+1)}label="{nodes[node_id].label}";')
                sub_dot_lines.append(f'{"    "*(depth+1)}style=rounded;')
                sub_dot_lines.append(f'{"    "*(depth+1)}color=blue;')
                sub_dot_lines += dfsGenerateDot(nodes[node_id].children, depth+1)
                sub_dot_lines.append(f'{"    "*depth}}}')

        return sub_dot_lines

    root_nodes_id = [k for k, v in nodes.items() if v.parent is None]
    node_dot_lines = dfsGenerateDot(root_nodes_id, depth=1)

    # 6.定义dot文件头尾，完成组装
    root_dot_lines: List[str] = []
    root_dot_lines.append("digraph G {")
    root_dot_lines.append('    rankdir=LR;')              # 从左到右绘制
    root_dot_lines.append('    node [fontname="Arial"];')
    root_dot_lines.append("}")
    result = "\n".join(root_dot_lines[:-1] + node_dot_lines + edges_dot_lines + root_dot_lines[-1:])
    return result


def Draw(nodes: Dict[int, Node], id) -> None:
    dot_content = GenerateDot(nodes)
    with open(f'./click{id}.dot', "w") as dot_file:
        dot_file.write(dot_content)
        print(f"Generated ./click{id}.dot")
    
    complex_graph_dot_file = f"./click{id}.dot"
    complex_graph_png_file = f"./click{id}.png"
    subprocess.run(['dot', '-Tpng', '-Gdpi=300', complex_graph_dot_file, '-o', complex_graph_png_file], check=True)
    print(f"Transfered {complex_graph_dot_file} to {complex_graph_png_file}.")

# 更新节点状态，折叠或者展开
def Click(id: int) -> None:
    origin_nodes[id].isCollapse = not origin_nodes[id].isCollapse
    new_nodes = GenerateNewGraph(origin_nodes)
    Draw(new_nodes, id)

ReadFile("./data/DNN/complex_graph.json")

new_nodes = GenerateNewGraph(origin_nodes)
Draw(new_nodes, -1)

Click(1)

Click(4)

Click(15)

Click(9)