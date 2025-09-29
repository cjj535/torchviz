// 全局变量
let maxTime = -1;
let minTime = -1;
let relativeMaxTime = -1;
let relativeMinTime = -1;
let nsToS = 10000000;
// *******************************************************************************************
// node
// *******************************************************************************************
class Node {
  constructor(node_json) {
    this.id = node_json.id;
    this.start_time = node_json.start_time;
    this.end_time = node_json.end_time
    this.isTensor = !!node_json.isTensor;
    this.isLeaf = !!node_json.isLeaf;
    this.label = node_json.label ?? String(node_json.id);
    this.parent = node_json.parent ?? null;
    this.children = Array.isArray(node_json.children) ? [...node_json.children] : [];
    this.nextNodes = Array.isArray(node_json.nextNodes) ? [...node_json.nextNodes] : [];
    this.isCollapse = true;
    if (this.isTensor && node_json?.info) {
      const device = node_json.info.device ?? 'unknown';
      const dtype = node_json.info.dtype ?? 'unknown';
      const size = node_json.info.size ?? 'unknown';
      const shape = node_json.info.shape ?? 'unknown';
      this.info = `${device}\n${dtype}\n${size}\n${shape}`;
    } else {
      this.info = "";
    }

    // 相对时间（会在 Graph 初始化时计算）
    this.relative_start_time = 0;
    this.relative_end_time = 0;
  }
}

// *******************************************************************************************
// graph
// *******************************************************************************************
class Graph {
  constructor() {
    this.nodes = new Map();
  }

  isLegalGraph() {
    const inDegree = new Map();

    // 初始化入度表
    for (const node of this.nodes.values()) {
      inDegree.set(node.id, 0);
    }

    // 第一遍遍历：检查各种规则并计算入度
    for (const [u, node] of this.nodes) {
      // 规则1: 非叶子节点不应该有 nextNodes
      if (!node.isLeaf && node.nextNodes.length > 0) {
        console.log("subgraph should not have next nodes.");
        return false;
      }

      // 规则2: 叶子节点不应该有 children
      if (node.isLeaf && node.children.length > 0) {
        console.log("leaf node should not have children.");
        return false;
      }

      // 规则3: 非叶子节点应该有 children
      if (!node.isLeaf && node.children.length === 0) {
        console.log("non leaf node should have children.");
        return false;
      }

      // 计算入度并检查规则4
      for (const v of node.nextNodes) {
        const nextNode = this.nodes.get(v);
        if (!nextNode) {
          console.log(`Next node ${v} not found in graph.`);
          return false;
        }

        // 规则4: op节点的输出应该是tensor，tensor应该是op节点的输入
        if (node.isTensor === nextNode.isTensor) {
          console.log("op node's output should be tensor, tensor should be input of op node.");
          return false;
        }

        // 更新入度
        inDegree.set(v, (inDegree.get(v) || 0) + 1);
      }
    }

    // 检查规则5: tensor节点入度不能超过1
    for (const [nodeId, degree] of inDegree) {
      const node = this.nodes.get(nodeId);
      if (node.isTensor && degree > 1) {
        console.log("tensor only has one producer.");
        return false;
      }
    }

    // 拓扑排序检查环
    const queue = [];
    for (const [nodeId, degree] of inDegree) {
      if (degree === 0) {
        queue.push(nodeId);
      }
    }

    let count = 0;
    while (queue.length > 0) {
      const u = queue.shift();
      count++;
      const currentNode = this.nodes.get(u);
      for (const v of currentNode.nextNodes) {
        const currentDegree = inDegree.get(v) - 1;
        inDegree.set(v, currentDegree);
        if (currentDegree === 0) {
          queue.push(v);
        }
      }
    }
    return count === this.nodes.size;
  }

  setRelativeTime() {
    for (const node of this.nodes.values()) {
      if (node.start_time !== -1) {
        if (minTime == -1) {
          minTime = node.start_time;
        } else {
          minTime = Math.min(minTime, node.start_time);
        }
        maxTime = Math.max(maxTime, node.start_time);
      }
      if (node.end_time !== -1) {
        if (minTime == -1) {
          minTime = node.end_time;
        } else {
          minTime = Math.min(minTime, node.end_time);
        }
        maxTime = Math.max(maxTime, node.end_time);
      }
    }
    if (minTime !== -1 && maxTime !== -1) {
      minTime = minTime - 10000000;
      maxTime = maxTime + 10000000;
    } else {
      minTime = 0;
      maxTime = 10000000;
    }
    relativeMinTime = 0;
    // relativeMaxTime = ((maxTime - minTime) / nsToS).toFixed(3);
    relativeMaxTime = maxTime - minTime;
    for (const node of this.nodes.values()) {
      if (node.start_time !== -1) {
        // node.relative_start_time = ((node.start_time - minTime) / nsToS).toFixed(3);
        node.relative_start_time = node.start_time - minTime;
      } else {
        node.relative_start_time = relativeMinTime;
      }
      if (node.end_time !== -1) {
        // node.relative_end_time = ((node.end_time - minTime) / nsToS).toFixed(3);
        node.relative_end_time = node.end_time - minTime;
      } else {
        // node.relative_end_time = (relativeMaxTime / nsToS).toFixed(3);
        node.relative_end_time = relativeMaxTime;
      }
      node.info += `\n${node.relative_start_time},${node.relative_end_time}`;
    }
  }

  // 获取在指定时间活跃的节点ID
  getActiveNodesAtTime(currentTime) {
    const activeNodes = new Set();
    for (const [nodeId, node] of this.nodes) {
      if (node.isLeaf && currentTime >= node.relative_start_time && currentTime <= node.relative_end_time) {
        activeNodes.add(nodeId);
      }
    }
    return activeNodes;
  }

  generate_dot(rootNodes = null, highlightNodes = []) {
    const node_dot_lines = [], edges_dot_lines = [];

    const dfs_generate_dot = (children, depth) => {
      const sub = [];
      children.forEach(node_id => {
        const node = this.nodes.get(node_id);
        if (!node) return;

        // 构建包含时间信息的标签
        let timeInfo = '';
        if (node.relativeStart !== undefined && node.relativeEnd !== undefined) {
          timeInfo = `\\n[${node.relativeStart.toFixed(1)}-${node.relativeEnd.toFixed(1)}ms]`;
        }

        const isHighlighted = highlightNodes.includes(node_id);
        const colorAttr = isHighlighted ? 'color=green, style=filled, fillcolor=lightgreen' : '';

        if (node.isLeaf) {
          const shape = node.isTensor ? "ellipse" : "box";
          const tooltip = `tooltip="${escapeDotLabel(node.info || node.label)}"`;
          sub.push(`${"    ".repeat(depth)}"${node_id}" [label="${escapeDotLabel(node.label + timeInfo)}", shape=${shape}, ${tooltip}, ${colorAttr}];`);
          node.nextNodes.forEach(id => { 
              edges_dot_lines.push(`${"    "}"${node_id}" -> "${id}";`) 
          });
        } else {
          const clusterLabel = node.label + timeInfo;
          sub.push(`${"    ".repeat(depth)}subgraph cluster_${node_id} {`);
          sub.push(`${"    ".repeat(depth+1)}label="${escapeDotLabel(clusterLabel)}";`);
          sub.push(`${"    ".repeat(depth+1)}style=rounded;`);
          if (isHighlighted) {
            sub.push(`${"    ".repeat(depth+1)}color=green;`);
            sub.push(`${"    ".repeat(depth+1)}style="rounded,filled";`);
            sub.push(`${"    ".repeat(depth+1)}fillcolor=lightgreen;`);
          } else {
            sub.push(`${"    ".repeat(depth+1)}color=blue;`);
          }
          const clusterTooltip = `tooltip="${escapeDotLabel(node.info || `Cluster: ${node.label}`)}"`;
          sub.push(`${"    ".repeat(depth+1)}${clusterTooltip};`);
          sub.push(...dfs_generate_dot(node.children, depth+1));
          sub.push(`${"    ".repeat(depth)}}`);
        }
      });
      return sub;
    };

    if (!rootNodes) rootNodes = [...this.nodes.values()].filter(n => n.parent===null).map(n=>n.id);
    node_dot_lines.push(...dfs_generate_dot(rootNodes, 1));
    const root_dot_lines=[
        "digraph G {",
        '    rankdir=LR;',
        '    node [fontname="Arial"];',
        '    tooltip = "";',
        " }"
    ];
    return [...root_dot_lines.slice(0,-1),...node_dot_lines,...edges_dot_lines,...root_dot_lines.slice(-1)].join("\n");
  }
  _get_out_tensors_of_collapse_node(root_id) {
    const result=[];
    const in_root=(node_id)=>{while(node_id!=null){if(node_id===root_id)return true;node_id=this.nodes.get(node_id)?.parent??null;}return false;}
    const dfs=(nid)=>{const node=this.nodes.get(nid);if(!node)return;if(node.isLeaf){if(node.isTensor && node.nextNodes.some(n=>!in_root(n)))result.push(nid);}else{node.children.forEach(c=>dfs(c));}};
    dfs(root_id);return result;
  }
  generate_new_graph() {
    const new_graph = new Graph();
    const roots = [...this.nodes.values()].filter(n => n.parent===null).map(n=>n.id);
    const dfs_build = (node_id) => {
      const node=this.nodes.get(node_id);if(!node)return[];
      if(node.isLeaf){new_graph.nodes.set(node_id, deepCopyNode(node)); return [];}
      if(node.isCollapse){const c=deepCopyNode(node);c.isLeaf=true;c.children=[];c.nextNodes=this._get_out_tensors_of_collapse_node(node_id);new_graph.nodes.set(node_id,c);return c.nextNodes||[];}
      new_graph.nodes.set(node_id, deepCopyNode(node)); let extra=[]; node.children.forEach(child=>{extra=extra.concat(dfs_build(child))});
      extra.forEach(cid=>{const o=this.nodes.get(cid);if(o){const copy=deepCopyNode(o);copy.parent=node_id;new_graph.nodes.set(cid,copy)}})
      const ngNode=new_graph.nodes.get(node_id); if(ngNode){ngNode.children=Array.from(new Set([...(ngNode.children||[]),...extra]))}
      return [];
    };
    let extra=[]; roots.forEach(r=>{extra=extra.concat(dfs_build(r))});
    extra.forEach(cid=>{const o=this.nodes.get(cid);if(o){const copy=deepCopyNode(o);copy.parent=null;new_graph.nodes.set(cid,copy)}})
    const find_ancestor=(nid)=>{while(nid!=null&&!new_graph.nodes.has(nid)){nid=this.nodes.get(nid)?.parent??null;}return nid;}
    const dfs_edges=(nid)=>{const node=new_graph.nodes.get(nid);if(!node)return;const updated=new Set([...node.nextNodes].map(n=>find_ancestor(n)).filter(x=>x!=null));node.nextNodes=Array.from(updated);(node.children||[]).forEach(c=>dfs_edges(c));}
    roots.forEach(r=>{if(new_graph.nodes.has(r))dfs_edges(r);});
    return new_graph;
  }
}

// *******************************************************************************************
// timeline manager
// *******************************************************************************************
class TimelineManager {
  constructor() {
    this.slider = document.getElementById('time-slider');
    this.currentTimeDisplay = document.getElementById('current-time');
    this.minTimeDisplay = document.getElementById('min-time');
    this.maxTimeDisplay = document.getElementById('max-time');
    this.currentTime = 0;
    this.isDragging = false;
    
    this.init();
  }

  init() {
    this.slider.addEventListener('input', (e) => {
      this.currentTime = parseInt(e.target.value);
      this.updateDisplay();
      this.onTimeChange(this.currentTime);
    });
    
    this.slider.addEventListener('mousedown', () => {
      this.isDragging = true;
    });
    
    this.slider.addEventListener('mouseup', () => {
      this.isDragging = false;
    });
  }

  updateTimeRange(min_time, max_time) {
    this.slider.min = min_time;
    this.slider.max = max_time;
    this.slider.value = 0;
    this.currentTime = 0;

    this.minTimeDisplay.textContent = '0';
    this.maxTimeDisplay.textContent = max_time;

    this.updateDisplay();
  }

  updateDisplay() {
    this.currentTimeDisplay.textContent = `当前时间: ${this.currentTime}`;
    this.slider.value = this.currentTime;
  }

  onTimeChange(time) {
    if (window.highlightNodesAtTime) {
      window.highlightNodesAtTime(time);
    }
  }

  setTime(time) {
    this.currentTime = Math.max(this.slider.min, Math.min(this.slider.max, time));
    this.updateDisplay();
    this.onTimeChange(this.currentTime);
  }
}

function addHoverEffects(svgEl) {
  // 创建 tooltip 元素
  const tooltip = document.createElement('div');
  tooltip.className = 'graph-tooltip';
  tooltip.style.display = 'none';
  document.body.appendChild(tooltip);

  // 提取节点 ID 的辅助函数
  function extractNodeId(elementId) {
    // 处理 Graphviz 生成的各种 ID 格式
    if (elementId.startsWith('cluster_')) {
      return elementId.replace('cluster_', '');
    }
    if (elementId.startsWith('node')) {
      return elementId.replace('node', '');
    }
    return elementId;
  }

  // 为节点添加事件监听
  const nodes = svgEl.querySelectorAll('g.node, g.cluster');

  nodes.forEach(node => {
    // 获取节点 ID
    let nodeId = node.id;
    if (!nodeId && node.querySelector('title')) {
      nodeId = node.querySelector('title').textContent.replace(/^"|"$/g, '');
    }

    const originalNodeId = extractNodeId(nodeId);
    const originalNode = currentRenderGraph.nodes.get(originalNodeId) || originGraph.nodes.get(originalNodeId);
    
    if (originalNode) {
      // 添加悬停类名
      if (node.classList.contains('cluster')) {
        node.classList.add('graph-cluster');
      } else {
        node.classList.add('graph-node');
      }

      // 鼠标进入事件
      node.addEventListener('mouseenter', (event) => {
        const info = originalNode.info || originalNode.label;
        if (info) {
          tooltip.textContent = info;
          tooltip.style.display = 'block';
          updateTooltipPosition(tooltip, event);
        }
      });

      // 鼠标移动事件
      node.addEventListener('mousemove', (event) => {
        updateTooltipPosition(tooltip, event);
      });

      // 鼠标离开事件
      node.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
      });
    }
  });

  // SVG 容器的鼠标离开事件
  svgEl.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
  });

  // 更新 tooltip 位置
  function updateTooltipPosition(tooltipElement, event) {
    const x = event.clientX + 15;
    const y = event.clientY + 15;

    // 确保 tooltip 不会超出窗口边界
    const maxX = window.innerWidth - tooltipElement.offsetWidth - 20;
    const maxY = window.innerHeight - tooltipElement.offsetHeight - 20;

    tooltipElement.style.left = Math.min(x, maxX) + 'px';
    tooltipElement.style.top = Math.min(y, maxY) + 'px';
  }

  return tooltip;
}

// 带时间高亮的渲染函数
async function renderWithTimeHighlight(highlightNodes = []) {
  if (!currentRenderGraph) return;
  const dot = currentRenderGraph.generate_dot(null, highlightNodes);
  try {
    const svgEl = await viz.renderSVGElement(dot);
    svgContainer.innerHTML = '';
    svgContainer.appendChild(svgEl);

    // 重新附加事件
    attachClickHandlersToRenderedSVG(svgEl);
    addHoverEffects(svgEl);
  } catch (err) {
    console.error(err);
  }
}

function escapeDotLabel(s){
  return String(s)
    .replace(/\\/g,"\\\\")
    .replace(/"/g,'\\"')
    .replace(/\n/g,'\\n')
    .replace(/\r/g,'\\r')
    .replace(/\t/g,'\\t');
}
function deepCopyNode(node){return JSON.parse(JSON.stringify(node));}

const viz = new Viz();
const svgContainer=document.getElementById('svgContainer');
const status=document.getElementById('status');
let originGraph=new Graph();
let currentRenderGraph=null;
let timelineManager = null;

// 高亮节点的函数
function highlightNodesAtTime(currentTime) {
  if (!currentRenderGraph) return;

  // 移除之前的高亮
  const svgEl = svgContainer.querySelector('svg');
  if (!svgEl) return;

  // 重置所有节点边框样式
  const nodeShapes = svgEl.querySelectorAll('.highlighted-node > g > a > ellipse, .highlighted-node > g > a > polygon');
  nodeShapes.forEach(shape => {
    shape.removeAttribute('style');
  });

  // 移除所有高亮类名
  const highlightedElements = svgEl.querySelectorAll('.highlighted-node');
  highlightedElements.forEach(el => {
    el.classList.remove('highlighted-node');
  });

  // 获取当前时间活跃的节点
  const activeNodes = currentRenderGraph.getActiveNodesAtTime(currentTime);

  // 高亮活跃节点
  activeNodes.forEach(nodeId => {
    // 查找对应的 SVG 元素
    const nodeElements = findSvgElementsByTitle(svgEl, nodeId.toString());
    nodeElements.forEach(element => {
      if (element.classList.contains('node')) {
        element.classList.add('highlighted-node');
        
        const shapes = element.querySelectorAll('ellipse, polygon, path');
        shapes.forEach(shape => {
          // 只高亮边框
          shape.style.stroke = '#4CAF50';
          shape.style.strokeWidth = '3px';
          shape.style.filter = 'drop-shadow(0 0 8px rgba(76, 175, 80, 0.6))';
        });
      }
    });
  });
}

// 通过 title 内容查找 SVG 元素的辅助函数
function findSvgElementsByTitle(svgElement, titleText) {
  const elements = [];
  // 查找所有 title 元素
  const titles = svgElement.querySelectorAll('title');
  titles.forEach(title => {
    // 获取 title 的文本内容并清理
    const titleContent = title.textContent.trim().replace(/^"|"$/g, '');

    // 检查 title 内容是否匹配节点 ID
    if (titleContent === titleText) {
      // 找到对应的父元素（node 或 cluster）
      let parent = title.parentElement;
      while (parent && parent !== svgElement) {
        if (parent.classList.contains('node') || parent.classList.contains('cluster')) {
          elements.push(parent);
          break;
        }
        parent = parent.parentElement;
      }
    }
  });
  return elements;
}

// 将高亮函数暴露给全局
window.highlightNodesAtTime = highlightNodesAtTime;

async function renderFromOriginGraph() {
  currentRenderGraph=originGraph.generate_new_graph();
  const dot=currentRenderGraph.generate_dot();
  try{
    const svgEl=await viz.renderSVGElement(dot);
    svgContainer.innerHTML='';
    svgContainer.appendChild(svgEl);
    // 添加点击事件
    attachClickHandlersToRenderedSVG(svgEl);
    // 添加悬停效果
    addHoverEffects(svgEl);
    // 初始高亮
    highlightNodesAtTime(timelineManager ? timelineManager.currentTime : 0);
  }catch(err){
    console.error(err);
  }
}

function attachClickHandlersToRenderedSVG(svgEl){
  const nodeGroupList=svgEl.querySelectorAll('g.node');
  nodeGroupList.forEach(g=>{
    const title=g.querySelector('title');
    if(!title)return;
    const nodeIdText=title.textContent.trim().replace(/^"|"$/g,'');
    const nid=isNaN(Number(nodeIdText))?nodeIdText:Number(nodeIdText);
    g.style.cursor='pointer';
    g.addEventListener('mouseenter',()=>g.style.opacity='0.7');
    g.addEventListener('mouseleave',()=>g.style.opacity='1');
    g.addEventListener('click',async e=>{
      e.stopPropagation();
      if(originGraph.nodes.has(nid)){
        originGraph.nodes.get(nid).isCollapse=!originGraph.nodes.get(nid).isCollapse;
        await renderFromOriginGraph();
      }
    });
  });
  const clusterGroupList = svgEl.querySelectorAll('g.cluster');
  clusterGroupList.forEach(g => {
    const title = g.querySelector('title');
    if(!title) return;
    let clusterIdText = title.textContent.trim().replace(/^"|"$/g,'');
    // 我们在 DOT 中生成的 cluster 名称是 cluster_{id}
    if(clusterIdText.startsWith('cluster_')){
      const nid = Number(clusterIdText.replace('cluster_',''));
      g.style.cursor = 'pointer';
      g.addEventListener('mouseenter', ()=> g.style.opacity='0.7');
      g.addEventListener('mouseleave', ()=> g.style.opacity='1');
      g.addEventListener('click', async e => {
        e.stopPropagation();
        if(originGraph.nodes.has(nid)){
          originGraph.nodes.get(nid).isCollapse = !originGraph.nodes.get(nid).isCollapse;
          await renderFromOriginGraph();
        }
      });
    }
  });
}

document.getElementById('jsonFileInput').addEventListener('change', async (event)=>{
  const file=event.target.files[0];
  if(!file){return;}
  const reader=new FileReader();
  reader.onload = async function(e){
    try{
      // 创建原始图
      const nodes_json=JSON.parse(e.target.result);
      originGraph=new Graph();
      nodes_json.forEach(nj=>{originGraph.nodes.set(nj.id,new Node(nj))});

      // 更新时间为相对时间
      originGraph.setRelativeTime();

      // 判断图是否合法
      const isValid = originGraph.isLegalGraph();
      if (!isValid) {
        console.log("illegal graph, exit!");
        return;
      }

      // 初始化时间条管理器
      if (!timelineManager) {
        timelineManager = new TimelineManager();
        timelineManager.updateTimeRange(relativeMinTime, relativeMaxTime)
      }
      await renderFromOriginGraph();
    }catch(err){
      console.error(err);
    }
  }
  reader.readAsText(file,'utf-8');
});
