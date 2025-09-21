from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict, deque
import json
import copy
from graphviz import Source

class Node:
    id: int
    isTensor: bool
    isLeaf: bool
    label: str
    parent: Optional[int]
    children: List[int]
    nextNodes: List[int]
    isCollapse: bool

    def __init__(self, node_json: Dict):
        self.id = node_json["id"]
        self.isTensor = node_json["isTensor"]
        self.isLeaf = node_json["isLeaf"]
        self.label = node_json["label"]
        self.parent = node_json["parent"]
        self.children = node_json["children"]
        self.nextNodes = node_json["nextNodes"]

        self.isCollapse = True          # 默认均折叠


class Graph:
    nodes: Dict[int, Node]

    def __init__(self):
        self.nodes = {}

    def generate_dot(self) -> str:
        """
        生成dot文件用于可视化
        """
        node_dot_lines: List[str] = []
        edges_dot_lines: List[str] = []

        # dfs遍历函数，每次调用分析一个子图中应有的节点，包括tensor节点、op节点、子图
        def dfs_generate_dot(children: List[int], depth: int) -> List[str]:
            sub_dot_lines: List[str] = []
            for node_id in children:
                if self.nodes[node_id].isLeaf:
                    # 添加op
                    shape = "ellipse" if self.nodes[node_id].isTensor else "box"
                    sub_dot_lines.append(f'{"    "*depth}"{node_id}" [label="{self.nodes[node_id].label}", shape={shape}];')
                    for id in self.nodes[node_id].nextNodes:
                        edges_dot_lines.append(f'{"    "}"{node_id}" -> "{id}";')
                else:
                    # 添加子图
                    sub_dot_lines.append(f'{"    "*depth}subgraph cluster_{node_id} {{')
                    sub_dot_lines.append(f'{"    "*(depth+1)}label="{self.nodes[node_id].label}";')
                    sub_dot_lines.append(f'{"    "*(depth+1)}style=rounded;')
                    sub_dot_lines.append(f'{"    "*(depth+1)}color=blue;')
                    sub_dot_lines += dfs_generate_dot(self.nodes[node_id].children, depth+1)
                    sub_dot_lines.append(f'{"    "*depth}}}')
            return sub_dot_lines

        root_nodes_id = [k for k, v in self.nodes.items() if v.parent is None]
        node_dot_lines = dfs_generate_dot(root_nodes_id, depth=1)

        # 定义dot文件头尾，完成组装
        root_dot_lines: List[str] = []
        root_dot_lines.append("digraph G {")
        root_dot_lines.append('    rankdir=LR;')              # 从左到右绘制
        root_dot_lines.append('    node [fontname="Arial"];')
        root_dot_lines.append("}")
        result = "\n".join(root_dot_lines[:-1] + node_dot_lines + edges_dot_lines + root_dot_lines[-1:])
        return result

    def generate_svg(self) -> str:
        """
        渲染 SVG。
        返回生成的 SVG 内容。
        """
        dot_str = self.generate_dot()

        # 用 Source 包装 DOT 字符串
        src = Source(dot_str, format="svg")
        
        # 渲染为 SVG 字符串
        svg_bytes = src.pipe(format="svg")
        svg_str = svg_bytes.decode("utf-8")
        
        return svg_str

    def _get_out_tensors_of_collapse_node(self, root_id: int) -> List[int]:
        """找到折叠节点子树中作为外部输入的 tensor"""
        result = []

        def in_root(node_id: Optional[int]) -> bool:
            while node_id is not None:
                if node_id == root_id:
                    return True
                node_id = self.nodes[node_id].parent
            return False

        def dfs(node_id: int):
            node = self.nodes[node_id]
            if node.isLeaf:
                if node.isTensor and any(not in_root(n) for n in node.nextNodes):
                    result.append(node_id)
            else:
                for child in node.children:
                    dfs(child)

        dfs(root_id)
        return result

    def generate_new_graph(self) -> "Graph":
        new_graph = Graph()

        # 获取根节点
        roots = [i for i, n in self.nodes.items() if n.parent is None]

        # ------------------------------
        # 调整部分tensor在拓扑图和树中的位置，并更改相关属性的值
        # ------------------------------
        def dfs_build(node_id: int) -> List[int]:
            """
            重新生成拓扑图节点，并更新tensor在拓扑图中的拓扑关系和在树中的从属关系
            """
            node = self.nodes[node_id]

            # 1.叶节点，停止dfs
            if node.isLeaf:
                new_graph.nodes[node_id] = copy.deepcopy(node)
                return []

            # 2.折叠节点，停止dfs
            if node.isCollapse:
                collapsed = copy.deepcopy(node)
                collapsed.isLeaf = True
                collapsed.children = []

                # 更新拓扑关系
                collapsed.nextNodes = self._get_out_tensors_of_collapse_node(node_id)
                new_graph.nodes[node_id] = collapsed
                return collapsed.nextNodes

            # 3.1.非叶子节点或者非折叠节点继续dfs
            new_graph.nodes[node_id] = copy.deepcopy(node)
            extra_children = []
            for child in node.children:
                extra_children.extend(dfs_build(child))

            # 3.2.更新新增子节点（tensor类型节点）的 parent
            for cid in extra_children:
                new_graph.nodes[cid] = copy.deepcopy(self.nodes[cid])
                new_graph.nodes[cid].parent = node_id

            # 3.3.新增子节点（tensor类型节点）
            new_graph.nodes[node_id].children.extend(extra_children)
            return []

        extra = []
        for r in roots:
            extra.extend(dfs_build(r))

        for cid in extra:
            new_graph.nodes[cid] = copy.deepcopy(self.nodes[cid])
            new_graph.nodes[cid].parent = None
            roots.append(cid)

        # ------------------------------
        # 基于新的拓扑关系重新连接边
        # ------------------------------
        def find_ancestor(nid: Optional[int]) -> Optional[int]:
            """
            从节点开始，在树中向上寻找首个出现在生成的graph的祖先节点，即被折叠的子图
            """
            while nid is not None and nid not in new_graph.nodes:
                nid = self.nodes[nid].parent
            return nid

        def dfs_edges(nid: int):
            """
            在生成的graph中，部分节点因为折叠被去除，则边的源点和目的点需要改成其被折叠的祖先节点
            """
            node = new_graph.nodes[nid]
            updated = {find_ancestor(n) for n in node.nextNodes}
            node.nextNodes = [n for n in updated if n is not None]
            for c in node.children:
                dfs_edges(c)

        for r in roots:
            dfs_edges(r)

        return new_graph

    # 更新节点状态，折叠或者展开
    def click(self, id: int) -> "Graph":
        if id in self.nodes:
            self.nodes[id].isCollapse = not self.nodes[id].isCollapse
        return self.generate_new_graph()


def get_graph_from_file(path: str):
    graph = Graph()
    with open(path, 'r') as graph_json_file:
        nodes_json = json.load(graph_json_file)

    for node in nodes_json:
        graph.nodes[node["id"]] = Node(node)
    return graph


def draw(graph: Graph, id) -> None:
    svg_content = graph.generate_svg()
    with open(f'./sample_{id}.svg', "w", encoding="utf-8") as svg_file:
        svg_file.write(svg_content)
        print(f"Generated ./sample_{id}.svg")


origin_graph = get_graph_from_file("../data/DNN/complex_graph.json")

draw(origin_graph.click(-1), 0)
draw(origin_graph.click(1), 1)
draw(origin_graph.click(4), 2)
draw(origin_graph.click(15), 3)
draw(origin_graph.click(9), 4)