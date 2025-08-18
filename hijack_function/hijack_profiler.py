from torch.profiler._memory_profiler import (
    MemoryProfile,
    DataFlowGraph,
    CategoryDict,
    SizeMap,
    _EventType,
    TensorKey,
    Category,
    OpTree,
    _TensorMetadata,
    _ProfilerEvent,
    TensorAndID,
    RecordScope,
    SchemaMatcher,
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
) -> List[int]:
    id_list: List[int] = []
    json_list = []
    id = 0
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
        # 过滤掉Allocation节点，这些节点基本是free事件
        if node._event.typed[0] != _EventType.TorchOp:
            continue

        node_dict = {}
        
        # 没有展示绝对时间，而是使用一个递增的id，由于遍历是按时间顺序遍历，因此id顺序即为时间顺序
        node_dict['id'] = id
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
            id_list.append(node_dict["id"])
        
        # id递增
        id += 1

    # 最后导出为json文件
    import json
    with open('./sample/sample_result/graph_data.json', 'w') as f:
        json.dump(json_list, f, indent=4)
    
    return id_list

def tree_to_json(op_tree: OpTree, id_list: List[int]) -> None:
    
    class Node:
        def __init__(self, id: int, name: str, start_time: int, end_time: int, is_leaf: bool, parent: Optional[int] = None):
            self.id = id
            self.name = name
            self.start_time = start_time,
            self.end_time = end_time,
            self.is_leaf = is_leaf
            self.children = []
            self.parent = parent

    def process_forest(root_nodes: List[_ProfilerEvent]) -> List[Dict]:
        """处理原始森林结构，过滤并重新编号节点"""
        # 第一步：构建树，省去非module节点
        id_count = 0
        nodes: Dict[int, Node] = {}  # id -> Node 对象
        leaf_nodes: List[Node] = []
        non_leaf_nodes: List[Node] = []

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
        
        def dfs(event: _ProfilerEvent, parent_id: Optional[int]):
            """递归处理事件树，构建节点关系"""
            nonlocal id_count
            for child in event.children:
                if not is_tree_node(child):
                    if not is_leaf(child):
                        dfs(child, parent_id)  # 非树节点继续递归但保持父节点
                    continue

                # 处理树节点
                node_id = id_count
                nodes[node_id] = Node(
                    id=node_id,
                    name=child.name,  # 移除前11个字符
                    start_time=child.start_time_ns,
                    end_time=child.end_time_ns,
                    is_leaf=is_leaf(child),
                    parent=parent_id
                )
                
                # 更新父子关系
                if parent_id is not None:
                    nodes[parent_id].children.append(node_id)
                
                id_count += 1
                
                # 分类存储节点
                target_list = leaf_nodes if is_leaf(child) else non_leaf_nodes
                target_list.append(nodes[node_id])
                
                # 递归处理非叶节点
                if not is_leaf(child):
                    dfs(child, node_id)
            
        for root in root_nodes:
            if not is_tree_node(root):
                if not is_leaf(root):
                    dfs(root, None)  # 非树节点继续递归但保持父节点
                continue
            node_id = id_count
            nodes[node_id] = Node(
                id=node_id,
                name=root.name,
                start_time=root.start_time_ns,
                end_time=root.end_time_ns,
                is_leaf=is_leaf(root))
            id_count += 1
            target_list = leaf_nodes if is_leaf(root) else non_leaf_nodes
            target_list.append(nodes[node_id])
            if not is_leaf(root):
                dfs(root, node_id)
        leaf_nodes.sort(key=lambda x: x.start_time)
        non_leaf_nodes.sort(key=lambda x: x.start_time)

        # 第二步：重新编号节点（叶节点优先）
        new_id_map = {}  # 旧ID -> 新ID
        new_id_count = 0
        
        # 先编号叶节点（按原始顺序）
        for node in leaf_nodes:
            if node.id in nodes and node.id not in new_id_map:
                new_id_map[node.id] = new_id_count
                new_id_count += 1
            else:
                print("warn: invalid node id")
        
        # 再编号非叶节点（按原始顺序）
        for node in non_leaf_nodes:
            if node.id in nodes and node.id not in new_id_map:
                new_id_map[node.id] = new_id_count
                new_id_count += 1
            else:
                print("warn: invalid node id")
        
        # 第三步：构建新的森林结构
        node_details: Dict[int, Dict] = {}  # 新ID -> 节点信息
        
        # 填充节点信息
        for old_id, new_id in new_id_map.items():
            node = nodes[old_id]
            node_details[new_id] = {
                "id": new_id,
                "name": node.name,
                "start_time": node.start_time,
                "end_time": node.end_time,
                "is_leaf": node.is_leaf,
                "parent": None,
                "children": [],
            }
        
        # 重建父子关系
        for old_id, new_id in new_id_map.items():
            node = nodes[old_id]
            for child_id in node.children:
                if child_id in new_id_map:
                    node_details[new_id]["children"].append(new_id_map[child_id])
                    node_details[new_id_map[child_id]]["parent"] = new_id
        
        new_leaf_node_id_list = []
        for node in leaf_nodes:
            if node.id in new_id_map:
                new_leaf_node_id_list.append(new_id_map[node.id])
        
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
                    "parent": node.get("parent"),
                    "children": [child_id for child_id in node.get("children", []) 
                            if child_id in valid_nodes],
                }
                result.append(filtered_node)
            
            return result

        new_nodes: List[Dict] = []
        for _, node in node_details.items():
            new_nodes.append(node)
        filter_nodes = filter_tree(new_nodes, new_leaf_node_id_list, id_list)
        return filter_nodes

    json_list = process_forest(op_tree._root_nodes)

    # 最后导出为json文件
    import json
    with open('./sample/sample_result/tree_data.json', 'w') as f:
        json.dump(json_list, f, indent=4)

# 先保存原始 __init__ 方法
_original_init = MemoryProfile.__init__

# 定义新的 __init__
def my_init(self, *args, **kwargs):
    print("Enter MemoryProfiler.__init__")

    # 最后调用原始 __init__
    _original_init(self, *args, **kwargs)
    
    timeMap = TimeMap(self._op_tree)
    tensorInfoMap = TensorInfoMap(self._data_flow_graph)
    id_list = graph_to_json(self._data_flow_graph, self._categories, self._size_map, timeMap, tensorInfoMap)

    tree_to_json(self._op_tree, id_list)

def hijack_profiler():
    # 替换 __init__
    MemoryProfile.__init__ = my_init