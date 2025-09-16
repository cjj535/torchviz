from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict, deque
import json
import copy
import subprocess


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

    def __init__(self, node_json: Dict):
        self.id = node_json["id"]
        self.isTensor = node_json["isTensor"]
        self.isLeaf = node_json["isLeaf"]
        self.label = node_json["label"]
        self.parent = node_json["parent"]
        self.children = node_json["children"]
        self.nextNodes = node_json["nextNodes"]

        self.isCollapse = True          # 默认均折叠

        self.x = 0
        self.y = 0
        self.rx = 0
        self.ry = 0


class Graph:
    nodes: Dict[int, Node]
    CHAR_WIDTH = 1
    CHAR_HEIGHT = 2
    STR_X_PADDING = 1
    STR_Y_PADDING = 1
    FIGURE_X_PADDING = 4
    FIGURE_Y_PADDING = 2
    MARGIN = 2

    def __init__(self):
        self.nodes = {}

    def get_graph_from_file(self, path: str):
        with open(path, 'r') as graph_json_file:
            nodes_json = json.load(graph_json_file)

        for node in nodes_json:
            self.nodes[node["id"]] = Node(node)

    def _compute_path_len_between_nodes(self, pre_node_id: Optional[int], post_node_id: Optional[int]) -> int:
        def get_ancestors(node_id: Optional[int]) -> List[int]:
            ancestors: List[int] = []
            while node_id is not None:
                ancestors.append(node_id)
                node_id = self.nodes[node_id].parent
            return ancestors

        pre_node_ancestors = get_ancestors(pre_node_id)
        post_node_ancestors = get_ancestors(post_node_id)

        i, j = len(pre_node_ancestors) - 1, len(post_node_ancestors) - 1
        # 从尾部开始找公共后缀
        while i >= 0 and j >= 0 and pre_node_ancestors[i] == post_node_ancestors[j]:
            i -= 1
            j -= 1
        # 剩余部分长度之和
        path_length = (i + 1) + (j + 1)
        return path_length

    def compute_x_layout(self):
        """
        计算有向图节点的横向布局（只计算 x 坐标和 rx）
        """

        # ==========================================================
        # 1. 计算每个节点的宽度 (rx)
        # ==========================================================
        for node in self.nodes.values():
            node.rx = len(node.label) * self.CHAR_WIDTH + 2 * self.STR_X_PADDING

        # ==========================================================
        # 2. 构建前驱映射 + 出度统计
        # ==========================================================
        predecessors = defaultdict(list)
        out_degree = defaultdict(int)
        for u, node in self.nodes.items():
            for v in node.nextNodes:
                predecessors[v].append(u)
                out_degree[u] += 1

        # ==========================================================
        # 3. longest-path 分层 (从叶子往回推)
        # ==========================================================
        layer = {}
        queue = deque([u for u in self.nodes if out_degree[u] == 0])
        for u in queue:
            layer[u] = 0

        while queue:
            u = queue.popleft()
            for pred in predecessors[u]:
                layer[pred] = max(layer.get(pred, 0), layer[u] + 1)
                out_degree[pred] -= 1
                if out_degree[pred] == 0:
                    queue.append(pred)

        if not layer:
            return

        max_layer = max(layer.values())
        for u in layer:
            layer[u] = max_layer - layer[u]

        # ==========================================================
        # 4. 按层收集节点 & 层最大宽度
        # ==========================================================
        layer_nodes = defaultdict(list)
        for u, l in layer.items():
            layer_nodes[l].append(u)

        layer_width = {
            l: max(self.nodes[u].rx for u in nodes)
            for l, nodes in layer_nodes.items()
        }

        # ==========================================================
        # 5. 预计算每层的 margin（避免重复调用函数）
        # ==========================================================
        def max_length_between_layers(l: Optional[int]) -> int:
            if l is None:
                return max(
                    (self._get_out_tensors_of_collapse_node(None, nid) for nid in layer_nodes[0]),
                    default=0,
                )
            if l == max_layer:
                return max(
                    (self._get_out_tensors_of_collapse_node(nid, None) for nid in layer_nodes[l]),
                    default=0,
                )
            return max(
                (
                    self._get_out_tensors_of_collapse_node(nid, next_id)
                    for nid in layer_nodes[l]
                    for next_id in self.nodes[nid].nextNodes
                    if next_id in layer_nodes[l + 1]
                ),
                default=0,
            )

        layer_margin = {l: max_length_between_layers(l) for l in range(max_layer + 1)}
        margin_start = max_length_between_layers(None)

        # ==========================================================
        # 6. 分配每层的横坐标中心
        # ==========================================================
        layer_x = {}
        pos = margin_start * self.MARGIN * 2
        for l in range(max_layer + 1):
            rx = layer_width[l]
            x_center = pos + self.FIGURE_X_PADDING + rx / 2
            layer_x[l] = x_center
            pos += rx + self.FIGURE_X_PADDING * 2 + layer_margin[l] * self.MARGIN * 2

        # ==========================================================
        # 7. 叶子节点赋值坐标
        # ==========================================================
        for u, node in self.nodes.items():
            if node.isLeaf:
                node.x = layer_x[layer[u]]

        # ==========================================================
        # 8. 递归计算子图 (中心点 x, 宽度 rx)
        # ==========================================================
        def dfs_subgraph(node_id: int) -> Tuple[float, float]:
            node = self.nodes[node_id]
            left, right = None, None
            for cid in node.children:
                c_node = self.nodes[cid]
                if c_node.isLeaf:
                    c_left = c_node.x - c_node.rx / 2 - self.FIGURE_X_PADDING
                    c_right = c_node.x + c_node.rx / 2 + self.FIGURE_X_PADDING
                else:
                    c_left, c_right = dfs_subgraph(cid)

                left = c_left if left is None else min(left, c_left)
                right = c_right if right is None else max(right, c_right)

            assert left is not None and right is not None and left <= right

            left -= self.MARGIN * 2
            right += self.MARGIN * 2

            node.x = (left + right) / 2
            node.rx = right - left - self.MARGIN * 2
            return left, right

        for root_id, root_node in self.nodes.items():
            if root_node.parent is None:
                dfs_subgraph(root_id)

    def compute_y_layout(self):

        return

    def compute_layout(self):
        """
        计算布局，包括图形中心坐标点以及图形大小
        从左向右布局，暂不支持其他方式
        """
        self.compute_x_layout()
        self.compute_y_layout()

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


def draw(graph: Graph, id) -> None:
    dot_content = graph.generate_dot()
    with open(f'./sample_{id}.dot', "w") as dot_file:
        dot_file.write(dot_content)
        print(f"Generated ./sample_{id}.dot")

    complex_graph_dot_file = f"./sample_{id}.dot"
    complex_graph_png_file = f"./sample_{id}.png"
    subprocess.run(['dot', '-Tpng', '-Gdpi=300', complex_graph_dot_file, '-o', complex_graph_png_file], check=True)
    print(f"Transfered {complex_graph_dot_file} to {complex_graph_png_file}.")

    import os
    if os.path.exists(complex_graph_dot_file):  # 先检查文件是否存在
        os.remove(complex_graph_dot_file)


origin_graph = Graph()
origin_graph.get_graph_from_file("./graph.json")

draw(origin_graph.click(-1), 0)
draw(origin_graph.click(1), 1)
draw(origin_graph.click(4), 2)
draw(origin_graph.click(15), 3)
draw(origin_graph.click(9), 4)
