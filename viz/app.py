"""
Robust POMCP Explorer — Event Replay UI.

Loads a chronological event log produced by the solver
(see experiments/tiger/tiger_test.py) and replays the algorithm step by step.

Usage:
    cd viz
    pip install -r requirements.txt
    python app.py

Open http://localhost:8050 in a browser.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import dash_cytoscape as cyto

# Enables non-default layouts (e.g. dagre) so each subtree gets its own region
# and sibling branches don't overlap visually.
cyto.load_extra_layouts()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DEFAULT_JSON_PATH = Path(__file__).parent.parent / "experiments" / "tiger" / "tiger_run.json"


def load_run(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# State replay
# ---------------------------------------------------------------------------
#
# For any event index k in [0, N-1], we can replay events[0..k] and derive:
#   - `nodes`: dict  node_id -> dict with the node's current fields
#   - `edges`: list of (src_id, tgt_id, label) triples
#   - `active_node`: node to highlight at this moment
#   - `recently_created`: ids of nodes that were created by event k
#
# Node fields stored per history node:
#   {type: "history", id, depth, N, Q_nominal?, Q_robust?, V_robust,
#    V_nominal, V_rollout, has_rollout, particles, S_in}
# Per action node:
#   {type: "action", id, action, N, Q_nominal, Q_robust}


def rebuild_state(events, up_to_idx):
    """Replay events 0..up_to_idx (inclusive) and return a derived state dict."""
    nodes = {}   # id -> dict
    edges = []   # list of (src, tgt, label)
    active_node = None
    recently_created = set()

    # Bootstrap: the root must be seeded by the first sim_start event.
    # It doesn't appear via a "create" event — we create it when we see it referenced.
    def ensure_history_node(nid, depth=None):
        if nid not in nodes:
            nodes[nid] = {
                "type": "history", "id": nid, "depth": depth,
                "N": 0, "S_in": [], "particles_count": 0,
                "V_robust": 0.0, "V_nominal": 0.0,
                "V_rollout": None, "has_rollout": False,
            }
        return nodes[nid]

    def ensure_action_node(nid, action=None):
        if nid not in nodes:
            nodes[nid] = {
                "type": "action", "id": nid, "action": action,
                "N": 0, "Q_nominal": 0.0, "Q_robust": 0.0,
            }
        return nodes[nid]

    for idx, ev in enumerate(events[:up_to_idx + 1]):
        is_last = idx == up_to_idx
        if is_last:
            recently_created = set()

        etype = ev["type"]
        if etype == "sim_start":
            root_id = ev["active_node"]
            ensure_history_node(root_id, depth=0)
            active_node = root_id

        elif etype == "sim_step":
            h_id = ev["active_node"]
            a_id = ev["action_node_id"]
            child_id = ev["next_history_id"]
            depth = ev["depth"]
            ensure_history_node(h_id, depth=depth)
            a_node = ensure_action_node(a_id, action=ev["action_chosen"])
            # edge from history to action (if not already present)
            if not any(e for e in edges if e[0] == h_id and e[1] == a_id):
                edges.append((h_id, a_id, f"a={ev['action_chosen']}"))
            # The child history
            ensure_history_node(child_id, depth=depth + 1)
            if not any(e for e in edges if e[0] == a_id and e[1] == child_id):
                edges.append((a_id, child_id, f"z={ev['obs']}"))
            # Mark newly-created nodes if this event created them
            if is_last:
                for nid in ev["created_action_nodes"]:
                    recently_created.add(nid)
                for nid in ev["created_history_nodes"]:
                    recently_created.add(nid)
            # Visit count + particles bump on the active history
            nodes[h_id]["N"] += 1
            nodes[h_id]["particles_count"] += 1
            # Track S_in
            if ev["state"] not in nodes[h_id]["S_in"]:
                nodes[h_id]["S_in"].append(ev["state"])
            active_node = h_id

        elif etype == "expand":
            h_id = ev["node_id"]
            ensure_history_node(h_id, depth=ev["depth"])
            # An expand event marks the leaf-visit where add_particle! fired in the
            # algorithm but no sim_step was emitted (since it's a leaf). Bump N and
            # particles_count to mirror that visit.
            nodes[h_id]["N"] += 1
            nodes[h_id]["particles_count"] += 1
            # expand provides action indices that map to the created action ids
            action_indices = ev.get("created_action_indices", [])
            for i, a_id in enumerate(ev["created_action_nodes"]):
                action_idx = action_indices[i] if i < len(action_indices) else None
                ensure_action_node(a_id, action=action_idx)
                # Update action index if we already had a placeholder
                if action_idx is not None:
                    nodes[a_id]["action"] = action_idx
                if not any(e for e in edges if e[0] == h_id and e[1] == a_id):
                    edges.append((h_id, a_id, f"a={action_idx}" if action_idx else ""))
                if is_last:
                    recently_created.add(a_id)
            active_node = h_id

        elif etype == "rollout":
            leaf_id = ev["leaf_node_id"]
            ensure_history_node(leaf_id)
            nodes[leaf_id]["V_rollout"] = ev["total_return"]
            nodes[leaf_id]["has_rollout"] = True
            # For leaf V_robust/V_nominal the solver uses the rollout fallback
            nodes[leaf_id]["V_robust"] = ev["total_return"]
            nodes[leaf_id]["V_nominal"] = ev["total_return"]
            active_node = leaf_id

        elif etype == "backup_nominal_step":
            a_id = ev["action_node_id"]
            ensure_action_node(a_id)
            nodes[a_id]["Q_nominal"] = ev["new_Q_nominal"]
            nodes[a_id]["N"] = ev["new_N"]
            # Propagate to parent history node's V_nominal
            _recompute_v_nominal_of_parent(nodes, edges, a_id)
            active_node = a_id

        elif etype == "sim_end":
            pass  # no tree change

        elif etype == "robust_backup":
            h_id = ev["history_node_id"]
            a_id = ev["action_node_id"]
            ensure_action_node(a_id, action=ev["action"])
            nodes[a_id]["Q_robust"] = ev["new_Q_robust"]
            # Propagate to parent history node's V_robust
            _recompute_v_robust_of_parent(nodes, edges, a_id)
            active_node = a_id

    return nodes, edges, active_node, recently_created


def _find_parent_history(edges, action_id):
    """Return the id of the history node that is the parent of this action node."""
    for src, tgt, _ in edges:
        if tgt == action_id and src.startswith("h"):
            return src
    return None


def _recompute_v_nominal_of_parent(nodes, edges, action_id):
    """Recompute V_nominal at the parent history node of the given action node."""
    parent_h = _find_parent_history(edges, action_id)
    if parent_h is None or parent_h not in nodes:
        return
    # Find all action children of parent_h
    action_children = [tgt for src, tgt, _ in edges
                       if src == parent_h and tgt.startswith("a")]
    # max over VISITED actions' Q_nominal; fallback to V_rollout
    visited_qs = [nodes[a]["Q_nominal"] for a in action_children
                  if a in nodes and nodes[a].get("N", 0) > 0]
    if visited_qs:
        nodes[parent_h]["V_nominal"] = max(visited_qs)
    elif nodes[parent_h].get("has_rollout"):
        nodes[parent_h]["V_nominal"] = nodes[parent_h]["V_rollout"]


def _recompute_v_robust_of_parent(nodes, edges, action_id):
    """Recompute V_robust at the parent history node of the given action node."""
    parent_h = _find_parent_history(edges, action_id)
    if parent_h is None or parent_h not in nodes:
        return
    action_children = [tgt for src, tgt, _ in edges
                       if src == parent_h and tgt.startswith("a")]
    visited_qs = [nodes[a]["Q_robust"] for a in action_children
                  if a in nodes and nodes[a].get("N", 0) > 0]
    if visited_qs:
        nodes[parent_h]["V_robust"] = max(visited_qs)
    elif nodes[parent_h].get("has_rollout"):
        nodes[parent_h]["V_robust"] = nodes[parent_h]["V_rollout"]


def to_cytoscape_elements(nodes, edges, active_id, recently_created, meta=None):
    """Convert our derived state dict to cytoscape elements (nodes + edges).

    Labels include the node's own id, so you can see h1, a2, h5, ... directly on the tree.
    Action nodes also include the action's readable name when metadata provides it.
    """
    meta = meta or {}
    action_names = meta.get("action_names", [])

    def aname(a):
        if a is None:
            return "?"
        if action_names and 0 <= a < len(action_names):
            return action_names[a]
        return str(a)

    elements = []
    for nid, n in nodes.items():
        if n["type"] == "history":
            # e.g. "h1\nHistory d=0\nN=500\nVr=-12.5"
            label = (f"{nid}\nHistory d={n.get('depth', '?')}\n"
                     f"N={n['N']}  Vr={n['V_robust']:.1f}")
            classes = "history-node"
        else:
            a = n.get("action")
            # e.g. "a2\nAction: listen\nN=456 Qr=-2.3"
            label = (f"{nid}\nAction: {aname(a)}\n"
                     f"N={n['N']}  Qn={n['Q_nominal']:.1f}  Qr={n['Q_robust']:.1f}")
            classes = "action-node"
        if nid == active_id:
            classes += " active-node"
        if nid in recently_created:
            classes += " just-created"
        elements.append({
            "data": {"id": nid, "label": label, "raw": n},
            "classes": classes,
        })
    for (src, tgt, lbl) in edges:
        elements.append({
            "data": {"source": src, "target": tgt, "label": lbl}
        })
    return elements


# ---------------------------------------------------------------------------
# Event renderers (the side panel)
# ---------------------------------------------------------------------------

TABLE_STYLE = {
    "borderCollapse": "collapse",
    "border": "1px solid #999",
    "marginTop": "4px",
    "marginBottom": "4px",
    "fontSize": "12px",
}
TH_STYLE = {
    "border": "1px solid #999",
    "padding": "4px 8px",
    "backgroundColor": "#eee",
    "textAlign": "left",
}
TD_STYLE = {
    "border": "1px solid #999",
    "padding": "3px 8px",
}


def _th(content):
    return html.Th(content, style=TH_STYLE)


def _td(content):
    return html.Td(content, style=TD_STYLE)


def render_event_description(ev, meta):
    state_names = meta.get("state_names", [])
    action_names = meta.get("action_names", [])
    obs_names = meta.get("obs_names", [])

    def sname(s): return state_names[s] if state_names else str(s)
    def aname(a): return action_names[a] if action_names else str(a)
    def oname(z): return obs_names[z] if obs_names else str(z)

    etype = ev["type"]
    if etype == "sim_start":
        return [
            html.H4(f"Simulation {ev['sim_index']} — START"),
            html.P(f"Initial state sampled from belief: {sname(ev['initial_state'])}"),
        ]

    if etype == "sim_step":
        # UCB table
        ucb_rows = [html.Tr([_th("action"), _th("Q"), _th("N")])]
        for i, (q, n) in enumerate(zip(ev["ucb_qs"], ev["ucb_ns"]), start=1):
            ucb_rows.append(html.Tr([
                _td(aname(i)), _td(f"{q:.2f}"), _td(str(n))
            ]))
        created = ev.get("created_history_nodes", []) + ev.get("created_action_nodes", [])
        return [
            html.H4(f"Simulation {ev['sim_index']} — Step {ev['step_in_sim']}"),
            html.P(f"At history node {ev['active_node']} (depth={ev['depth']})"),
            html.P(f"Current state: {sname(ev['state'])}"),
            html.P("UCB table:"),
            html.Table(ucb_rows, style=TABLE_STYLE),
            html.P([
                html.B("Action chosen: "), f"{aname(ev['action_chosen'])}"
            ]),
            html.P(f"Sampled: s_next = {sname(ev['s_next'])}, obs = {oname(ev['obs'])}, "
                   f"reward = {ev['reward']:.2f}"),
            html.P(f"New nodes created: {created}" if created else "No new nodes"),
        ]

    if etype == "expand":
        return [
            html.H4(f"Expand — sim {ev['sim_index']}"),
            html.P(f"History node {ev['node_id']} (depth={ev['depth']}) expanded."),
            html.P(f"Created action nodes: {ev['created_action_nodes']}"),
        ]

    if etype == "rollout":
        rows = [html.Tr([_th("state"), _th("action"), _th("reward"), _th("s_next")])]
        for st in ev["rollout_steps"]:
            rows.append(html.Tr([
                _td(sname(st["state"])),
                _td(aname(st["action"])),
                _td(f"{st['reward']:.2f}"),
                _td(sname(st["s_next"])),
            ]))
        return [
            html.H4(f"Rollout — sim {ev['sim_index']}"),
            html.P(f"Leaf: {ev['leaf_node_id']} (depth={ev['start_depth']}, "
                   f"start state {sname(ev['start_state'])})"),
            html.P(f"Total return: {ev['total_return']:.2f}"),
            html.Table(rows, style=TABLE_STYLE),
        ]

    if etype == "backup_nominal_step":
        return [
            html.H4(f"Backup Q_nominal — sim {ev['sim_index']}"),
            html.P(f"Action node: {ev['action_node_id']}"),
            html.P(f"Q_nominal: {ev['previous_Q_nominal']:.3f}  →  "
                   f"{ev['new_Q_nominal']:.3f}"),
            html.P(f"N: {ev['previous_N']}  →  {ev['new_N']}"),
            html.P(f"Return that drove update: {ev['total_return_from_here']:.2f}"),
        ]

    if etype == "sim_end":
        return [
            html.H4(f"Simulation {ev['sim_index']} — END"),
            html.P(f"Total return: {ev['total_return']:.2f}"),
        ]

    if etype == "robust_backup":
        rows = [
            html.H4(f"Robust Backup #{ev['backup_index']}"),
            html.P(f"Node: {ev['history_node_id']} (depth={ev['node_depth']})  |  "
                   f"Action node: {ev['action_node_id']} ({aname(ev['action'])})"),
            html.P(f"S_in: {[sname(s) for s in ev['S_in']]}"),
            html.P(f"Z_in: {[oname(z) for z in ev['Z_in']]}"),
            html.P(f"belief: {['%.3f' % x for x in ev['belief']]}"),
            html.P(f"V_children: {['%.3f' % x for x in ev['V_children']]}"),
            html.Hr(),
            html.H5("Step 1 — Observation LPs"),
        ]
        for lp in ev["obs_lp_results"]:
            sn = lp["s_next"]
            tbl = [html.Tr([_th("observation"), _th("nominal P_Z^o"), _th("optimal P_Z^in")])]
            for i, z in enumerate(ev["Z_in"]):
                tbl.append(html.Tr([
                    _td(oname(z)),
                    _td(f"{lp['nominal_p'][i]:.4f}"),
                    _td(f"{lp['optimal_p'][i]:.4f}"),
                ]))
            rows.append(html.Div([
                html.B(f"For s' = {sname(sn)} (ρ_Z = {lp['radius']}):"),
                html.Table(tbl, style=TABLE_STYLE),
                html.P(f"w(s') = {lp['objective_value']:.4f}"),
            ], style={"marginBottom": "10px"}))
        rows.append(html.H5("Step 2 — Transition LPs"))
        for lp in ev["trans_lp_results"]:
            s = lp["state"]
            tbl = [html.Tr([_th("s_next"), _th("nominal P_T^o"), _th("optimal P_T^in")])]
            for i, sn in enumerate(ev["S_in"]):
                tbl.append(html.Tr([
                    _td(sname(sn)),
                    _td(f"{lp['nominal_p'][i]:.4f}"),
                    _td(f"{lp['optimal_p'][i]:.4f}"),
                ]))
            rows.append(html.Div([
                html.B(f"For s = {sname(s)} (ρ_T = {lp['radius']}):"),
                html.Table(tbl, style=TABLE_STYLE),
                html.P(f"σ(s) = {lp['objective_value']:.4f}"),
            ], style={"marginBottom": "10px"}))
        rows.append(html.Hr())
        rows.append(html.P(f"Q_robust: {ev['previous_Q_robust']:.3f}  →  "
                           f"{ev['new_Q_robust']:.3f}",
                           style={"fontWeight": "bold"}))
        return rows

    return [html.P(f"Unknown event type: {etype}")]


# ---------------------------------------------------------------------------
# Node detail panel
# ---------------------------------------------------------------------------

def render_node_detail(raw, meta):
    if raw is None:
        return [html.P("(click any node to inspect)", style={"fontStyle": "italic"})]
    state_names = meta.get("state_names", [])
    action_names = meta.get("action_names", [])

    rows = [html.H4(f"{'History' if raw['type']=='history' else 'Action'} Node  —  {raw['id']}")]
    if raw["type"] == "history":
        rows.append(html.P(f"depth: {raw.get('depth')}"))
        rows.append(html.P(f"N (visits): {raw['N']}"))
        rows.append(html.P(f"V_robust:  {raw['V_robust']:.4f}"))
        rows.append(html.P(f"V_nominal: {raw['V_nominal']:.4f}"))
        if raw.get("has_rollout"):
            rows.append(html.P(f"V_rollout: {raw['V_rollout']:.4f}"))
        s_in = raw.get("S_in", [])
        if state_names:
            rows.append(html.P(f"S_in: {[state_names[s] for s in s_in]}"))
        else:
            rows.append(html.P(f"S_in: {s_in}"))
        rows.append(html.P(f"particles (count): {raw.get('particles_count', 0)}"))
    else:
        a = raw.get("action")
        aname = action_names[a] if (a is not None and action_names) else str(a)
        rows.append(html.P(f"action: {a}  ({aname})"))
        rows.append(html.P(f"N: {raw['N']}"))
        rows.append(html.P(f"Q_nominal: {raw['Q_nominal']:.4f}"))
        rows.append(html.P(f"Q_robust:  {raw['Q_robust']:.4f}"))
    return rows


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def jump_to_next_of_type(events, current_idx, target_type):
    """Find the next event of `target_type` strictly after current_idx."""
    for i in range(current_idx + 1, len(events)):
        if events[i]["type"] == target_type:
            return i
    return current_idx  # stay put if no more


def jump_to_prev_of_type(events, current_idx, target_type):
    for i in range(current_idx - 1, -1, -1):
        if events[i]["type"] == target_type:
            return i
    return current_idx


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, title="Robust POMCP Explorer")

def _resolve_json_path() -> Path:
    parser = argparse.ArgumentParser(description="Robust POMCP Explorer")
    parser.add_argument("--json",
                        type=Path,
                        default=DEFAULT_JSON_PATH,
                        help="Path to the run JSON file (default: %(default)s)"
                        )
    args, _ = parser.parse_known_args()
    return args.json


JSON_PATH = _resolve_json_path()
DATA = load_run(JSON_PATH) if JSON_PATH.exists() else None

cyto_stylesheet = [
    {"selector": "node", "style": {
        "label": "data(label)", "text-valign": "center", "text-halign": "center",
        "font-size": "10px", "text-wrap": "wrap", "background-color": "#eee",
        "border-width": 1, "border-color": "#888", "width": 80, "height": 55,
        "shape": "round-rectangle",
    }},
    {"selector": ".history-node", "style": {"background-color": "#aed6f1"}},
    {"selector": ".action-node",  "style": {"background-color": "#f9e79f"}},
    {"selector": ".active-node", "style": {
        "border-color": "red", "border-width": 4,
    }},
    {"selector": ".just-created", "style": {
        "border-color": "green", "border-width": 3,
    }},
    {"selector": "edge", "style": {
        "label": "data(label)", "font-size": "9px", "curve-style": "bezier",
        "target-arrow-shape": "triangle", "width": 1,
    }},
]


def build_layout():
    if DATA is None:
        return html.Div([
            html.H2("No data file found."),
            html.P(f"Expected: {JSON_PATH}"),
            html.P("Run experiments/tiger/tiger_test.py first."),
        ])

    metadata = DATA.get("metadata", {})
    total_events = len(DATA["events"])

    # Initial state: first event (usually sim_start)
    nodes, edges, active, created = rebuild_state(DATA["events"], 0)
    initial_elements = to_cytoscape_elements(nodes, edges, active, created, meta=metadata)

    return html.Div([
        html.H2("Robust POMCP Explorer", style={"marginBottom": "5px"}),
        html.P(
            f"Problem: {metadata.get('problem', '?')}  |  "
            f"Budget: {metadata.get('budget', '?')}  |  "
            f"Horizon: {metadata.get('horizon', '?')}  |  "
            f"ρ_T={metadata.get('rho_T')}, ρ_Z={metadata.get('rho_Z')}  |  "
            f"Events: {total_events}",
            style={"color": "#666"}),

        # Hidden store for current event index
        dcc.Store(id="event-idx", data=0),

        html.Div(style={"display": "flex", "marginTop": "10px"}, children=[
            # Left: tree
            html.Div(style={"flex": "3", "border": "1px solid #ccc"}, children=[
                cyto.Cytoscape(
                    id="cyto",
                    elements=initial_elements,
                    layout={
                        "name": "dagre",
                        "rankDir": "TB",
                        "nodeSep": 30,
                        "rankSep": 60,
                    },
                    style={"width": "100%", "height": "750px"},
                    stylesheet=cyto_stylesheet,
                )
            ]),
            # Right: control + info
            html.Div(style={"flex": "2", "padding": "10px",
                            "maxHeight": "750px", "overflowY": "auto"}, children=[
                html.Div(id="event-header", style={"fontSize": "14px", "marginBottom": "10px"}),
                html.Div([
                    html.Button("⟵ Prev step", id="btn-prev-step", n_clicks=0),
                    html.Button("Next step ⟶", id="btn-next-step", n_clicks=0,
                                style={"marginLeft": "5px"}),
                ]),
                html.Div(style={"marginTop": "5px"}, children=[
                    html.Button("⟵ Prev sim", id="btn-prev-sim", n_clicks=0),
                    html.Button("Next sim ⟶", id="btn-next-sim", n_clicks=0,
                                style={"marginLeft": "5px"}),
                ]),
                html.Div(style={"marginTop": "5px"}, children=[
                    html.Button("⟵ Prev backup", id="btn-prev-bkp", n_clicks=0),
                    html.Button("Next backup ⟶", id="btn-next-bkp", n_clicks=0,
                                style={"marginLeft": "5px"}),
                ]),
                html.Hr(),
                html.Div(id="event-detail"),
                html.Hr(),
                html.H4("Selected node"),
                html.Div(id="node-detail"),
            ]),
        ]),
    ])


app.layout = build_layout()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(Output("event-idx", "data"),
              Input("btn-prev-step", "n_clicks"),
              Input("btn-next-step", "n_clicks"),
              Input("btn-prev-sim", "n_clicks"),
              Input("btn-next-sim", "n_clicks"),
              Input("btn-prev-bkp", "n_clicks"),
              Input("btn-next-bkp", "n_clicks"),
              State("event-idx", "data"),
              prevent_initial_call=True
              )
def navigate(p_step, n_step, p_sim, n_sim, p_bkp, n_bkp, current):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    events = DATA["events"]
    total = len(events)

    if trigger == "btn-prev-step":
        return max(0, current - 1)
    if trigger == "btn-next-step":
        return min(total - 1, current + 1)
    if trigger == "btn-prev-sim":
        return jump_to_prev_of_type(events, current, "sim_start")
    if trigger == "btn-next-sim":
        return jump_to_next_of_type(events, current, "sim_start")
    if trigger == "btn-prev-bkp":
        return jump_to_prev_of_type(events, current, "robust_backup")
    if trigger == "btn-next-bkp":
        return jump_to_next_of_type(events, current, "robust_backup")
    return no_update


@app.callback(Output("cyto", "elements"),
              Output("event-header", "children"),
              Output("event-detail", "children"),
              Input("event-idx", "data")
              )
def update_view(idx):
    events = DATA["events"]
    idx = max(0, min(idx, len(events) - 1))
    ev = events[idx]

    nodes, edges, active, created = rebuild_state(events, idx)
    elements = to_cytoscape_elements(nodes, edges, active, created, meta=DATA.get("metadata", {}))

    header = html.Div([
        html.B(f"Event #{idx + 1} of {len(events)}"),
        html.Span(f"  —  type: {ev['type']}", style={"color": "#666"}),
    ])

    detail = render_event_description(ev, DATA.get("metadata", {}))
    return elements, header, detail


@app.callback(Output("node-detail", "children"),
              Input("cyto", "tapNodeData")
              )
def show_node_info(data):
    raw = data["raw"] if data else None
    return render_node_detail(raw, DATA.get("metadata", {}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
