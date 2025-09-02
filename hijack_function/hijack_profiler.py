from torch.profiler._memory_profiler import (
    MemoryProfile,
    DataFlowGraph,
    CategoryDict,
    SizeMap,
    TensorKey,
    Category,
    OpTree,
    TensorAndID,
    SchemaMatcher,
    get_scopes,
)
from torch._C._profiler import (
    _EventType,
    _ExtraFields_Allocation,
    _ExtraFields_TorchOp,
    _ProfilerEvent,
    _TensorMetadata,
    RecordScope,
)
from typing import (
    Any,
    cast,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict, deque
import torch
from torch.profiler import _utils
import json
from pathlib import Path
import weakref


node_id_map = weakref.WeakKeyDictionary()


def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3


class TimeMap:
    def __init__(self, op_tree: OpTree) -> None:
        self._values: Dict[TensorKey, List[int]] = {}
        for node in op_tree.sorted_nodes:
            if node.typed[0] == _EventType.Allocation:
                alloc_fields = node.typed[1]
                key = TensorKey.from_allocation(alloc_fields)
                if key:
                    if key not in self._values:
                        self._values[key] = [-1, -1]
                    if alloc_fields.alloc_size > 0:
                        self._values[key][0] = node.start_time_ns
                    else:
                        self._values[key][1] = node.start_time_ns
    
    def GetStartTime(self, key: TensorKey) -> int:
        if key in self._values:
            return self._values[key][0]
        else:
            return -1
    
    def GetEndTime(self, key: TensorKey) -> int:
        if key in self._values:
            return self._values[key][1]
        else:
            return -1


class TensorInfoMap:
    def __init__(self, data_flow_graph: DataFlowGraph) -> None:
        self._shapeMap: Dict[TensorAndID, str] = {}
        self._dtypeMap: Dict[TensorAndID, str] = {}
        
        def FindTensorAndID(key: Optional[TensorKey], set: Set[TensorAndID]) -> Optional[TensorAndID]:
            if key is None:
                return None
            for tensorAndId in set:
                if key == tensorAndId[0]:
                    return tensorAndId
            return None
            
        for node in data_flow_graph._flow_nodes:
            input_tensor_versions: Set[TensorAndID] = set()
            input_tensor_versions.update(((k, v) for k, (_, v) in node.inputs.items()))
            
            output_tensor_versions: Set[TensorAndID] = set()
            output_tensor_versions.update(((k, v) for (k, v) in node.outputs.items() if v != 0))
            
            subtree = tuple(_utils.traverse_dfs([node._event]))
            op_list = [i.typed[1] for i in subtree if i.typed[0] == _EventType.TorchOp]
            for op in op_list:
                for op_input in op.inputs:
                    # Tensor
                    if isinstance(op_input, _TensorMetadata):
                        key = TensorKey.from_tensor(op_input)
                        key_and_version = FindTensorAndID(key, input_tensor_versions)
                        if key_and_version is not None:
                            self._shapeMap.setdefault(key_and_version, ",".join(map(str, op_input.sizes)))
                            self._dtypeMap.setdefault(key_and_version, str(op_input.dtype))

                    # TensorList
                    elif isinstance(op_input, list):
                        for op_input_i in op_input:
                            key = TensorKey.from_tensor(op_input_i)
                            key_and_version = FindTensorAndID(key, input_tensor_versions)
                            if key_and_version is not None:
                                self._shapeMap.setdefault(key_and_version, ",".join(map(str, op_input_i.sizes)))
                                self._dtypeMap.setdefault(key_and_version, str(op_input_i.dtype))

            for op in reversed(op_list):
                for op_input in op.inputs:
                    # Tensor
                    if isinstance(op_input, _TensorMetadata):
                        key = TensorKey.from_tensor(op_input)
                        key_and_version = FindTensorAndID(key, output_tensor_versions)
                        if key_and_version is not None:
                            self._shapeMap.setdefault(key_and_version, ",".join(map(str, op_input.sizes)))
                            self._dtypeMap.setdefault(key_and_version, str(op_input.dtype))

                    # TensorList
                    elif isinstance(op_input, list):
                        for op_input_i in op_input:
                            key = TensorKey.from_tensor(op_input_i)
                            key_and_version = FindTensorAndID(key, output_tensor_versions)
                            if key_and_version is not None:
                                self._shapeMap.setdefault(key_and_version, ",".join(map(str, op_input_i.sizes)))
                                self._dtypeMap.setdefault(key_and_version, str(op_input_i.dtype))

    def getShape(self, key: TensorAndID) -> str:
        if key in self._shapeMap:
            return "[" + self._shapeMap[key] + "]"
        else:
            return "[]"
    
    def getDtype(self, key: TensorAndID) -> str:
        if key in self._dtypeMap:
            return self._dtypeMap[key]
        else:
            return "unknown"

def graph_to_json(
    graph: DataFlowGraph,
    category: CategoryDict,
    sizeMap: SizeMap,
    timeMap: TimeMap,
    tensorInfoMap: TensorInfoMap,
) -> Tuple[List[Dict], List[int]]:
    graph_id_list: List[int] = []
    json_list: List[Dict] = []
    _CATEGORY_TO_STRING = {
        Category.PARAMETER: "parameter",
        Category.OPTIMIZER_STATE: "optimizer_state",
        Category.INPUT: "input",
        Category.TEMPORARY: "temporary",
        Category.ACTIVATION: "activation",
        Category.GRADIENT: "gradient",
        Category.AUTOGRAD_DETAIL: "autograd_detail",
    }
    for node in graph.flow_nodes:
        # 过滤掉Allocation节点（这些节点基本是free事件）
        if node._event.typed[0] != _EventType.TorchOp:
            continue

        node_dict = {}
        
        # 没有展示绝对时间，而是使用一个递增的id，由于遍历是按时间顺序遍历，因此id顺序即为时间顺序
        node_dict['id'] = node_id_map[node._event]
        node_dict['name'] = node._event.name
        node_dict['start_time'] = node._event.start_time_ns
        node_dict['end_time'] = node._event.end_time_ns
        
        # 暂时不考虑展示intermediate中间tensor，只展示输入输出tensor
        def edge_to_dict(key: TensorKey, version: int):
            # 不展示cpu上的tensor
            if key.device.type != "cpu":
                return {
                    "id": key.id,
                    "version": version,
                    "device": f"{key.device.type}:{key.device.index}",
                    "shape": f"{tensorInfoMap.getShape((key, version))}",
                    "dtype": f"{tensorInfoMap.getDtype((key, version))}",
                    "size": sizeMap[key],
                    "start_time": timeMap.GetStartTime(key),
                    "end_time": timeMap.GetEndTime(key),
                    "category": _CATEGORY_TO_STRING[c] if (c := category.get(key, version)) is not None else "unknown"
                    # 还需要补充哪些信息，如生命周期
                }

        node_dict['in_edges'] = [res for k, (_, v) in node.inputs.items()
                                 if (res := edge_to_dict(k, v)) is not None]
        node_dict['out_edges'] = [res for k, v in node.outputs.items()
                                  if (res := edge_to_dict(k, v)) is not None]

        # 只保留有在device上计算的算子
        if node_dict['in_edges'] or node_dict['out_edges']:
            json_list.append(node_dict)
            graph_id_list.append(node_id_map[node._event])

    
    return json_list, graph_id_list


class Node:
    def __init__(self, id: int, name: str, start_time: int, end_time: int, is_leaf: bool, is_backward: bool, parent: Optional[int] = None):
        self.id = id
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.is_leaf = is_leaf
        self.is_backward = is_backward
        self.parent = parent
        self.children = []


def filter_tree(nodes: List[Dict], leaf_id_list: List[int], id_list: List[int]) -> List[Dict]:
    """
    过滤树节点：
    1. 去除叶子节点不在id_list中的节点
    2. 去除子树中无叶子节点的非叶节点

    :param nodes: 节点列表，每个节点包含id, children, parent
    :param leaf_id_list: 所有叶子节点的ID列表
    :param id_list: 需要保留的叶子节点ID列表
    :return: 过滤后的节点列表
    """
    # 转换为字典格式便于查找
    node_dict = {node['id']: node for node in nodes}
    valid_leaf_ids = set(leaf_id_list) & set(id_list)  # 需要保留的叶子节点

    # 第一步：标记所有有效叶子节点的祖先路径
    valid_nodes = set()

    # 从有效叶子节点向上追溯父节点
    for leaf_id in valid_leaf_ids:
        current_id = leaf_id
        while current_id in node_dict:
            if current_id in valid_nodes:
                break  # 已经处理过这个分支
            valid_nodes.add(current_id)
            current_id = node_dict[current_id].get('parent')

    # 第二步：过滤节点
    result = []
    for node in nodes:
        node_id = node['id']
        
        # 如果是叶子节点且不在有效列表中，跳过
        if node_id in leaf_id_list and node_id not in valid_leaf_ids:
            continue
        
        # 如果是非叶子节点且不在有效路径中，跳过
        if node_id not in leaf_id_list and node_id not in valid_nodes:
            continue

        # 复制节点并过滤子节点
        filtered_node = {
            "id": node_id,
            "name": node["name"],
            "start_time": node["start_time"],
            "end_time": node["end_time"],
            "is_leaf": node["is_leaf"],
            "is_backward": node["is_backward"],
            "parent": node.get("parent"),
            "children": [child_id for child_id in node.get("children", []) 
                    if child_id in valid_nodes],
        }
        result.append(filtered_node)

    return result

def is_leaf(e: _ProfilerEvent) -> bool:
    return (e.typed[0] == _EventType.TorchOp and (
        e.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        or bool(SchemaMatcher.match_schemas(e.typed[1]))
    )) or e.typed[0] == _EventType.Allocation

def is_tree_node(e: _ProfilerEvent) -> bool:
    return (e.typed[0] == _EventType.TorchOp and (
        e.typed[1].scope == RecordScope.BACKWARD_FUNCTION
        or bool(SchemaMatcher.match_schemas(e.typed[1]))
        # bool(SchemaMatcher.match_schemas(e.typed[1]))
    )) or (e.typed[0] == _EventType.PyCall and "nn.Module:" in e.name)

def is_backward(e: _ProfilerEvent) -> bool:
    if e.typed[0] == _EventType.TorchOp:
        return RecordScope.BACKWARD_FUNCTION in get_scopes(e):
    elif e.typed[0] == _EventType.PyCall:
        return False

def tree_to_json(op_tree: OpTree, graph_id_list: List[int]) -> List[Dict]:
    # 第一步：构建树，省去非module节点
    nodes: Dict[int, Node] = {}
    leaf_node_id_list: List[int] = []

    def dfs(events: List[_ProfilerEvent], parent_id: Optional[int]):
        """递归处理事件树，构建节点关系"""
        for event in events:
            if not is_tree_node(event):
                if not is_leaf(event):
                    dfs(event.children, parent_id)  # 非树节点继续递归但保持父节点
                continue

            # 处理树节点
            node_id = node_id_map[event]
            nodes[node_id] = Node(
                id=node_id,
                name=event.name,
                start_time=event.start_time_ns,
                end_time=event.end_time_ns,
                is_leaf=is_leaf(event),
                is_backward=is_backward(event),
                parent=parent_id
            )
            
            # 更新父子关系
            if parent_id is not None:
                nodes[parent_id].children.append(node_id)
            
            # 递归处理非叶节点
            if not is_leaf(event):
                dfs(event.children, node_id)
            else:
                leaf_node_id_list.append(node_id)
    
    dfs(op_tree._root_nodes, None)

    nodes_list: List[Dict] = []
    for _, node in nodes.items():
        nodes_list.append({
            "id": node.id,
            "name": node.name,
            "start_time": node.start_time,
            "end_time": node.end_time,
            "is_leaf": node.is_leaf,
            "is_backward": node.is_backward,
            "parent": node.parent,
            "children": node.children,
        })
    filter_nodes = filter_tree(nodes_list, leaf_node_id_list, graph_id_list)
    return filter_nodes

def set_id(op_tree: OpTree):
    id = 0

    def dfs(events: List[_ProfilerEvent]):
        nonlocal id
        for event in events:
            if is_tree_node(event):
                node_id_map[event] = id
                id += 1
            
            # 递归处理非叶节点
            if not is_leaf(event):
                dfs(event.children)
    
    dfs(op_tree._root_nodes)

# 先保存原始 __init__ 方法
_original_init = MemoryProfile.__init__

model = None
generate_tree = False

# 定义新的 __init__
def my_init(self, *args, **kwargs):
    print("Enter MemoryProfiler.__init__")

    # 最后调用原始 __init__
    _original_init(self, *args, **kwargs)
    
    set_id(self._op_tree)
    
    timeMap = TimeMap(self._op_tree)
    tensorInfoMap = TensorInfoMap(self._data_flow_graph)
    graph_json, graph_id_list = graph_to_json(self._data_flow_graph, self._categories, self._size_map, timeMap, tensorInfoMap)

    tree_json = tree_to_json(self._op_tree, graph_id_list)

    # validate
    # 1、校验反向节点的祖先都是反向
    # 2、校验is_leaf是否正确
    # 3、校验必为有向无环图

    # 最后导出为json文件
    global model
    global generate_tree
    folder_path = Path(f'./data/{model}')
    folder_path.mkdir(parents=True, exist_ok=True)
    if model != '':
        with open(f'./data/{model}/graph.json', 'w') as f:
            json.dump(graph_json, f, indent=4)
    if generate_tree:
        with open(f'./data/{model}/tree.json', 'w') as f:
            json.dump(tree_json, f, indent=4)


def hijack_profiler(model_name: str, is_generate_tree: int):
    global model
    global generate_tree
    model = model_name if model_name != '' else None
    generate_tree = True if is_generate_tree == 1 else False

    # 替换 __init__

    MemoryProfile.__init__ = my_init

