class Node {
  constructor(node_json) {
    this.id = node_json.id;
    this.isTensor = !!node_json.isTensor;
    this.isLeaf = !!node_json.isLeaf;
    this.label = node_json.label ?? String(node_json.id);
    this.parent = node_json.parent ?? null;
    this.children = Array.isArray(node_json.children) ? [...node_json.children] : [];
    this.nextNodes = Array.isArray(node_json.nextNodes) ? [...node_json.nextNodes] : [];
    this.isCollapse = true;
  }
}

class Graph {
  constructor() { this.nodes = new Map(); }
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
  generate_dot(rootNodes = null) {
    const node_dot_lines = [], edges_dot_lines = [];
    const dfs_generate_dot = (children, depth) => {
      const sub = [];
      children.forEach(node_id => {
        const node = this.nodes.get(node_id);
        if (!node) return;
        if (node.isLeaf) {
          const shape = node.isTensor ? "ellipse" : "box";
          sub.push(`${"    ".repeat(depth)}"${node_id}" [label="${escapeDotLabel(node.label)}", shape=${shape}];`);
          node.nextNodes.forEach(id => { edges_dot_lines.push(`${"    "}"${node_id}" -> "${id}";`) });
        } else {
          sub.push(`${"    ".repeat(depth)}subgraph cluster_${node_id} {`);
          sub.push(`${"    ".repeat(depth+1)}label="${escapeDotLabel(node.label)}";`);
          sub.push(`${"    ".repeat(depth+1)}style=rounded;`);
          sub.push(`${"    ".repeat(depth+1)}color=blue;`);
          sub.push(...dfs_generate_dot(node.children, depth+1));
          sub.push(`${"    ".repeat(depth)}}`);
        }
      });
      return sub;
    };
    if (!rootNodes) rootNodes = [...this.nodes.values()].filter(n => n.parent===null).map(n=>n.id);
    node_dot_lines.push(...dfs_generate_dot(rootNodes, 1));
    const root_dot_lines=["digraph G {",'    rankdir=LR;','    node [fontname="Arial"];'," }"];
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
  click(id){if(this.nodes.has(id)){this.nodes.get(id).isCollapse=!this.nodes.get(id).isCollapse;}return this.generate_new_graph();}
}

function escapeDotLabel(s){return String(s).replace(/\\/g,"\\\\").replace(/"/g,'\\"').replace(/\n/g,'\\n');}
function deepCopyNode(node){return JSON.parse(JSON.stringify(node));}

const viz = new Viz();
const svgContainer=document.getElementById('svgContainer');
const status=document.getElementById('status');
let originGraph=new Graph();
let currentRenderGraph=null;

async function renderFromOriginGraph() {
  status.textContent='生成渲染图...';
  currentRenderGraph=originGraph.generate_new_graph();
  const dot=currentRenderGraph.generate_dot();
  status.textContent='渲染 SVG...';
  try{
    const svgEl=await viz.renderSVGElement(dot);
    svgContainer.innerHTML=''; svgContainer.appendChild(svgEl);
    attachClickHandlersToRenderedSVG(svgEl);
    status.textContent='渲染完成，点击节点切换折叠状态';
  }catch(err){console.error(err);status.textContent='渲染失败: '+err;}
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
      const nodes_json=JSON.parse(e.target.result);
      originGraph=new Graph();
      nodes_json.forEach(nj=>{originGraph.nodes.set(nj.id,new Node(nj))});
      const isValid = originGraph.isLegalGraph();
      if (!isValid) {console.log("illegal graph, exit!");return;}
      await renderFromOriginGraph();
    }catch(err){console.error(err); status.textContent='解析 JSON 错误: '+err;}
  }
  reader.readAsText(file,'utf-8');
});
