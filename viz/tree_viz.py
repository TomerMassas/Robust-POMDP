"""
Static tree visualizer for A1 fixed-policy trees.

Renders an A1 tree (HistoryNode + ActionNode + node_data) to a standalone HTML
file with cytoscape.js embedded. Open the HTML in any browser for native
pan/zoom; the "Save PNG" button in the header exports the current view.

Sibling to `viz/app.py` (which is an event-replay UI for the online solver).
A1 trees are static, so this is a much simpler module: no Dash server, no
state machine, just a one-shot HTML render. Cytoscape conventions (history /
action node shapes, dagre layout) follow the patterns in app.py.
"""

from __future__ import annotations

import json
from pathlib import Path


def render_tree_to_html(root,
                        node_data,
                        action_names,
                        output_path,
                        *,
                        title: str = "A1 fixed-policy tree",
                        ) -> None:
    """Render an A1 fixed-policy tree to a standalone HTML file.

    Args:
        root: HistoryNode at the root of the A1 tree.
        node_data: dict[node.id -> NodeData] from build_fixed_policy_tree.
        action_names: sequence mapping action index -> display name
                      (e.g. ACTION_NAMES from synthetic_pomdp).
        output_path: where to write the HTML file.
        title: shown in the page header.
    """
    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    _walk(root, node_data, action_names, nodes_out, edges_out)
    elements = nodes_out + edges_out

    n_history = sum(1 for n in nodes_out if "history" in n["classes"])
    n_action = sum(1 for n in nodes_out if "action" in n["classes"])
    summary = (f"{n_history} history nodes, {n_action} action nodes, "
               f"{len(edges_out)} edges")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = (_HTML_TEMPLATE
            .replace("__TITLE__", _escape_html(title))
            .replace("__SUMMARY__", _escape_html(summary))
            .replace("__ELEMENTS_JSON__", json.dumps(elements))
            .replace("__DOWNLOAD_NAME__", output_path.stem + ".png"))
    output_path.write_text(html, encoding="utf-8")


def _walk(node,
          node_data,
          action_names,
          nodes_out: list[dict],
          edges_out: list[dict]
          ) -> None:
    data = node_data[node.id]
    hist_id = f"h{node.id}"
    history_label = "\n".join([
        hist_id,
        f"d={data.depth}",
        f"V={node.V_robust:+.3f}",
    ] + (["[TERM]"] if data.is_terminal else []))
    nodes_out.append({
        "data": {
            "id": hist_id,
            "label": history_label,
            "depth": data.depth,
            "V_robust": float(node.V_robust),
            "belief": [float(b) for b in data.belief],
            "policy_action": int(data.policy_action),
            "is_terminal": bool(data.is_terminal),
        },
        "classes": "history-node terminal" if data.is_terminal else "history-node",
    })

    if data.is_terminal:
        return

    action_obj = node.children[data.policy_action]
    action_id = f"a{action_obj.id}"
    aname = action_names[data.policy_action] if action_names else str(data.policy_action)
    action_label = f"{action_id}\n{aname}"
    nodes_out.append({
        "data": {
            "id": action_id,
            "label": action_label,
            "action_index": int(data.policy_action),
            "action_name": aname,
        },
        "classes": f"action-node action-{aname}",
    })
    edges_out.append({"data": {"source": hist_id, "target": action_id}})

    for z in sorted(action_obj.children.keys()):
        child = action_obj.children[z]
        edges_out.append({
            "data": {
                "source": action_id,
                "target": f"h{child.id}",
                "label": f"z={z}",
            }
        })
        _walk(child, node_data, action_names, nodes_out, edges_out)


def _escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  html, body {
    margin: 0;
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    color: #222;
  }
  #header {
    padding: 8px 16px;
    background: #f8f8f8;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  #header strong { font-size: 14px; }
  #header .summary, #header .legend { color: #666; font-size: 12px; }
  #header button {
    padding: 4px 12px;
    font-size: 12px;
    cursor: pointer;
    border: 1px solid #bbb;
    background: #fff;
    border-radius: 3px;
  }
  #header button:hover { background: #eee; }
  .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
    border: 1px solid #888;
  }
  #cy {
    width: 100vw;
    height: calc(100vh - 50px);
  }
</style>
</head>
<body>
<div id="header">
  <strong>__TITLE__</strong>
  <button onclick="savePng()">Save PNG</button>
  <button onclick="cy.fit()">Fit</button>
  <span class="summary">__SUMMARY__</span>
  <span class="legend">
    <span class="swatch" style="background:#cfe2f3;border-color:#3d85c6;"></span>history
    <span class="swatch" style="background:#f4cccc;border-color:#990000;border-style:dashed;"></span>terminal
    <span class="swatch" style="background:#eeeeee;border-color:#888888;"></span>inspect
    <span class="swatch" style="background:#d9ead3;border-color:#38761d;"></span>proceed
    <span class="swatch" style="background:#fff2cc;border-color:#bf9000;"></span>mitigate
    <span class="swatch" style="background:#f4cccc;border-color:#cc0000;"></span>abort
  </span>
</div>
<div id="cy"></div>
<script>
const elements = __ELEMENTS_JSON__;

if (typeof cytoscapeDagre !== 'undefined') {
  cytoscape.use(cytoscapeDagre);
}

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: elements,
  style: [
    {
      selector: 'node.history-node',
      style: {
        'shape': 'round-rectangle',
        'background-color': '#cfe2f3',
        'border-color': '#3d85c6',
        'border-width': 2,
        'label': 'data(label)',
        'text-wrap': 'wrap',
        'text-max-width': 80,
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 11,
        'width': 90,
        'height': 60,
      }
    },
    {
      selector: 'node.history-node.terminal',
      style: {
        'background-color': '#f4cccc',
        'border-color': '#990000',
        'border-style': 'dashed',
      }
    },
    {
      selector: 'node.action-node',
      style: {
        'shape': 'ellipse',
        'background-color': '#eeeeee',
        'border-color': '#888888',
        'border-width': 2,
        'label': 'data(label)',
        'text-wrap': 'wrap',
        'text-max-width': 70,
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': 10,
        'width': 70,
        'height': 50,
      }
    },
    {
      selector: 'node.action-inspect',
      style: { 'background-color': '#eeeeee', 'border-color': '#888888' }
    },
    {
      selector: 'node.action-proceed',
      style: { 'background-color': '#d9ead3', 'border-color': '#38761d' }
    },
    {
      selector: 'node.action-mitigate',
      style: { 'background-color': '#fff2cc', 'border-color': '#bf9000' }
    },
    {
      selector: 'node.action-abort',
      style: { 'background-color': '#f4cccc', 'border-color': '#cc0000' }
    },
    {
      selector: 'edge',
      style: {
        'width': 1.5,
        'line-color': '#999',
        'target-arrow-color': '#999',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.8,
        'curve-style': 'bezier',
        'label': 'data(label)',
        'font-size': 10,
        'color': '#444',
        'text-background-color': '#fff',
        'text-background-opacity': 0.9,
        'text-background-padding': 2,
      }
    },
  ],
  layout: {
    name: 'dagre',
    rankDir: 'TB',
    nodeSep: 40,
    rankSep: 60,
    edgeSep: 20,
  },
});

function savePng() {
  const png = cy.png({ full: true, scale: 2, bg: '#ffffff' });
  const a = document.createElement('a');
  a.href = png;
  a.download = '__DOWNLOAD_NAME__';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
</script>
</body>
</html>
"""