"""Export PyTorch experiment diagrams. Invoke only inside the experiment sandbox."""
from __future__ import annotations

import copy
import html
import json
import math
from pathlib import Path

import torch
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp


class VisualizationError(RuntimeError):
    """The example cannot be represented faithfully by a static FX graph."""


def _shape(value):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _shape(item) for key, item in value.items()}
    return None


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        value = value.values()
    if isinstance(value, (list, tuple)) or type(value).__name__ == "dict_values":
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    return None


def _activation(value):
    tensor = _first_tensor(value)
    if tensor is None or tensor.layout != torch.strided or tensor.is_complex():
        return None
    flat = tensor.detach().reshape(-1)
    count = min(64, flat.numel())
    if not count:
        return {"values": [], "sample_size": 0, "total_elements": 0}
    indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
    values = flat[indices].float().cpu().tolist()
    values = [v if math.isfinite(v) else None for v in values]
    finite = [v for v in values if v is not None]
    return {
        "values": values,
        "sample_size": count,
        "total_elements": flat.numel(),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "mean": sum(finite) / len(finite) if finite else None,
    }


class _Inspector(ShapeProp):
    def __init__(self, module):
        super().__init__(module)
        self.activations = {}

    def run_node(self, node):
        result = super().run_node(node)
        self.activations[node.name] = _activation(result)
        return result


def describe_model(model, example_inputs, *, title="Neural network", max_nodes=500):
    """Trace an eval-mode copy and execute one example forward pass.

    example_inputs is a tuple of positional arguments. Parameters and buffers are
    copied; the original model's modes, gradients, and running statistics survive.
    Custom forward methods still execute Python: this is not a host-side parser.
    """
    if not isinstance(example_inputs, tuple):
        raise TypeError("example_inputs must be a tuple of positional arguments")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    try:
        # A CPU example with a CPU model is recommended to avoid duplicating VRAM.
        # Preserve RNG state for randomized custom forward methods.
        cuda_devices = sorted({
            t.device.index
            for t in (*model.parameters(), *model.buffers(), *example_inputs)
            if isinstance(t, torch.Tensor) and t.is_cuda
        })
        with torch.random.fork_rng(devices=cuda_devices):
            traced = symbolic_trace(copy.deepcopy(model).eval())
            if len(list(traced.graph.nodes)) > max_nodes:
                raise VisualizationError(f"Graph exceeds the {max_nodes}-node limit")
            inspector = _Inspector(traced)
            with torch.no_grad():
                inspector.propagate(*copy.deepcopy(example_inputs))
    except VisualizationError:
        raise
    except Exception as exc:
        raise VisualizationError(
            "FX tracing or example execution failed. Data-dependent control flow "
            "may be unsupported; supply a traceable wrapper or another exporter. "
            "No sequential fallback diagram was generated."
        ) from exc

    nodes, edges = [], []
    for node in traced.graph.nodes:
        module = traced.get_submodule(str(node.target)) if node.op == "call_module" else None
        target = str(node.target)
        label = type(module).__name__ if module is not None else getattr(node.target, "__name__", target)
        nodes.append({
            "id": node.name,
            "op": node.op,
            "target": target,
            "label": label,
            "parameters": sum(p.numel() for p in module.parameters()) if module is not None else 0,
            "output": _shape(node.meta.get("tensor_meta")),
            "activation_sample": inspector.activations.get(node.name),
        })
        edges.extend({"source": parent.name, "target": node.name} for parent in node.all_input_nodes)
    return {
        "schema_version": 1,
        "title": str(title),
        "mode": "eval",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "notes": [
            "Shapes and activations describe one eval-mode example, not every possible input.",
            "Activation values and statistics use at most 64 sampled elements of the first tensor output.",
            "Per-node parameter counts can repeat for shared modules; the total counts unique parameters.",
        ],
        "nodes": nodes,
        "edges": edges,
    }


def _shape_label(output):
    if isinstance(output, dict) and "shape" in output:
        return str(output["shape"])
    if output is None:
        return "no tensor"
    return "nested output"


def render_svg(graph):
    """Render a dependency graph without Graphviz or browser dependencies."""
    depths, groups, positions = {}, {}, {}
    parents = {node["id"]: [] for node in graph["nodes"]}
    for edge in graph["edges"]:
        parents[edge["target"]].append(edge["source"])
    for node in graph["nodes"]:
        name = node["id"]
        depth = max((depths[parent] + 1 for parent in parents[name]), default=0)
        depths[name] = depth
        group = groups.setdefault(depth, [])
        positions[name] = (32 + depth * 256, 70 + len(group) * 150)
        group.append(name)
    width = max(640, len(groups) * 256 + 48)
    height = max(260, max((len(group) for group in groups.values()), default=1) * 150 + 100)
    esc = html.escape
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(graph["title"], quote=True)}">',
        '<style>.node{cursor:pointer}.node:focus rect{stroke:#f59e0b;stroke-width:4}.edge{fill:none;stroke:#94a3b8;stroke-width:2}</style>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#94a3b8"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="32" y="32" font-family="sans-serif" font-size="19" fill="#0f172a">{esc(str(graph["title"])[:100])}</text>',
    ]
    for edge in graph["edges"]:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        mid = (x1 + 212 + x2) / 2
        parts.append(f'<path class="edge" marker-end="url(#arrow)" d="M{x1+212},{y1+55} C{mid},{y1+55} {mid},{y2+55} {x2},{y2+55}"/>')
    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        color = "#ccfbf1" if node["op"] == "call_module" else "#e0e7ff"
        parts.append(f'<g class="node" data-node="{esc(node["id"], quote=True)}" tabindex="0" role="button" aria-label="{esc(node["id"], quote=True)}">')
        parts.append(f'<title>{esc(json.dumps(node, ensure_ascii=True))}</title>')
        parts.append(f'<rect x="{x}" y="{y}" width="212" height="110" rx="12" fill="{color}" stroke="#64748b"/>')
        labels = [node["id"], node["label"], _shape_label(node["output"]), f'{node["parameters"]:,} parameters']
        for index, label in enumerate(labels):
            text = str(label)
            if len(text) > 29:
                text = text[:26] + "..."
            parts.append(f'<text x="{x+12}" y="{y+23+index*23}" font-family="sans-serif" font-size="13" fill="#0f172a">{esc(text)}</text>')
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def render_dot(graph):
    lines = ["digraph neural_network {", '  rankdir="LR";', '  node [shape="box"];']
    for node in graph["nodes"]:
        label = f'{node["id"]}\n{node["label"]}\n{_shape_label(node["output"])}'
        lines.append(f'  {json.dumps(node["id"])} [label={json.dumps(label)}];')
    for edge in graph["edges"]:
        lines.append(f'  {json.dumps(edge["source"])} -> {json.dumps(edge["target"])};')
    return "\n".join(lines + ["}"]) + "\n"


def render_html(graph):
    # Escape script delimiters before embedding experiment-controlled text.
    payload = json.dumps(graph, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = html.escape(graph["title"])
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>""" + title + """</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font:15px system-ui,sans-serif}
header{padding:24px;background:#fff;border-bottom:1px solid #cbd5e1}h1{margin:0 0 8px;font-size:26px}
.controls{display:flex;flex-wrap:wrap;gap:20px;margin-top:16px}input[type=search]{padding:8px;border:1px solid #94a3b8;border-radius:6px}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;padding:16px}
#viewport{overflow:auto;max-height:76vh;background:#fff;border-radius:12px}#diagram{min-width:640px;transform-origin:top left}svg{display:block;width:100%}
aside{background:#fff;padding:20px;border-radius:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:12px}
#heatmap{display:grid;grid-template-columns:repeat(8,1fr);gap:3px}.cell{height:24px;border:1px solid #cbd5e1;border-radius:3px}
.dim{opacity:.18}.selected rect{stroke:#e11d48;stroke-width:4}small{color:#475569}
@media(max-width:900px){main{grid-template-columns:1fr}aside{min-height:240px}}
</style></head><body>
<header><h1>""" + title + """</h1><div id="summary"></div>
<small>One eval-mode example. Click or focus a node to inspect its layer and sampled activations.</small>
<div class="controls"><label>Find a layer <input id="search" type="search" placeholder="Name or layer type"></label>
<label>Zoom <input id="zoom" type="range" min="50" max="200" value="100"> <output id="zoom-value">100%</output></label></div></header>
<main><section id="viewport" aria-label="Architecture"><div id="diagram">""" + render_svg(graph) + """</div></section>
<aside><h2 id="selection">Select a node</h2><pre id="details"></pre><h3>Activation sample</h3>
<p><small>Up to 64 values, blue negative / red positive. Gray means non-finite or absent. Statistics describe the sample.</small></p>
<div id="heatmap"></div><pre id="stats"></pre></aside></main>
<script id="graph-data" type="application/json">""" + payload + """</script>
<script>
"use strict";
const graph=JSON.parse(document.getElementById("graph-data").textContent);
const nodes=new Map(graph.nodes.map(n=>[n.id,n]));
document.getElementById("summary").textContent=
  graph.nodes.length+" graph nodes / "+graph.parameters.toLocaleString()+" parameters / "+graph.trainable_parameters.toLocaleString()+" trainable";
function select(id){
  const node=nodes.get(id);if(!node)return;
  document.querySelectorAll(".node").forEach(el=>el.classList.toggle("selected",el.dataset.node===id));
  document.getElementById("selection").textContent=node.id;
  const {activation_sample,...details}=node;
  document.getElementById("details").textContent=JSON.stringify(details,null,2);
  const heatmap=document.getElementById("heatmap");heatmap.replaceChildren();
  const sample=activation_sample;
  const values=sample?sample.values:[];
  const scale=Math.max(1e-12,...values.filter(v=>v!==null).map(v=>Math.abs(v)));
  values.forEach(value=>{
    const cell=document.createElement("div");cell.className="cell";cell.title=String(value);
    if(value===null){cell.style.background="#cbd5e1";}
    else{const intensity=Math.round(95-55*Math.abs(value)/scale);cell.style.background="hsl("+(value<0?220:0)+" 75% "+intensity+"%)";}
    heatmap.append(cell);
  });
  const {values:ignored,...stats}=sample||{};
  document.getElementById("stats").textContent=sample?JSON.stringify(stats,null,2):"No dense real-valued tensor sample.";
}
document.querySelectorAll(".node").forEach(el=>{
  el.addEventListener("click",()=>select(el.dataset.node));
  el.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();select(el.dataset.node);}});
});
document.getElementById("search").addEventListener("input",event=>{
  const query=event.target.value.toLowerCase();
  document.querySelectorAll(".node").forEach(el=>{
    const node=nodes.get(el.dataset.node);
    el.classList.toggle("dim",!(node.id+" "+node.label+" "+node.target).toLowerCase().includes(query));
  });
});
document.getElementById("zoom").addEventListener("input",event=>{
  document.getElementById("diagram").style.width=event.target.value+"%";
  document.getElementById("zoom-value").textContent=event.target.value+"%";
});
</script></body></html>
"""


def export_visualizations(model, example_inputs, output_dir, *, title="Neural network", max_nodes=500):
    """Create offline HTML, SVG, DOT, and JSON artifacts from the same FX graph."""
    graph = describe_model(model, example_inputs, title=title, max_nodes=max_nodes)
    output = Path(output_dir)
    artifacts = {
        "architecture.json": json.dumps(graph, indent=2, allow_nan=False) + "\n",
        "architecture.svg": render_svg(graph),
        "architecture.dot": render_dot(graph),
        "index.html": render_html(graph),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output / name).write_text(content, encoding="utf-8")
    return {name: output / name for name in artifacts}
