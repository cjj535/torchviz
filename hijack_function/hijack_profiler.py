from torch.profiler._memory_profiler import (
    MemoryProfile,
    DataFlowGraph,
    CategoryDict,
    SizeMap,
    _EventType,
    TensorKey,
    Category,
)

# 先保存原始 __init__ 方法
_original_init = MemoryProfile.__init__

# cjj add
def graph_to_json(graph: DataFlowGraph, category: CategoryDict, sizeMap: SizeMap) -> None:
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
        
        # 暂时不考虑展示intermediate中间tensor，只展示输入输出tensor
        def edge_to_dict(key: TensorKey, version: int):
            # 不展示cpu上的tensor
            if key.device.type != "cpu":
                return {
                    "id": key.id,
                    "version": version,
                    "device": f"{key.device.type}:{key.device.index}",
                    "size": sizeMap[key],
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

# 定义新的 __init__
def my_init(self, *args, **kwargs):
    print("💡 MemoryProfiler.__init__ 被调用了！")
    # 最后调用原始 __init__
    _original_init(self, *args, **kwargs)
    
    graph_to_json(self._data_flow_graph, self._categories, self._size_map)

# 替换 __init__
MemoryProfile.__init__ = my_init