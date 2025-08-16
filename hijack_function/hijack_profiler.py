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
    TensorAndID,
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
) -> None:
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
        
        # id递增
        id += 1

    # 最后导出为json文件
    import json
    with open('./sample/sample_result/graph_data.json', 'w') as f:
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
    graph_to_json(self._data_flow_graph, self._categories, self._size_map, timeMap, tensorInfoMap)

def hijack_profiler():
    # 替换 __init__
    MemoryProfile.__init__ = my_init