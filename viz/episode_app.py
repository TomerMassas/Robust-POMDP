"""
FrozenLake episode viewer — Tab A (grid view) v1.

Loads a recorded episode JSON (produced by record_episode.py) and renders
per-step:
  - belief heatmap over the grid (blue translucent overlay)
  - true agent position (red disc) with NSEW observation halo
  - chosen action arrow
  - Q-bars side panel (Q_robust per action) + step readout

Usage:
    python viz/episode_app.py --episode <path/to/episode.json>

Open http://localhost:8050 in a browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, ctx, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "frozenlake"))
from frozenlake_problem import (  # noqa: E402
    GOAL_INDEX,
    HOLE_INDICES,
    N_ACTIONS,
    N_COLS,
    N_GRID_CELLS,
    N_ROWS,
    START_INDEX,
    TERMINAL_STATE,
    build_R,
    idx_to_cell,
    obs_idx_to_bits,
)


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

ACTION_NAMES  = ["LEFT", "DOWN", "RIGHT", "UP"]
ACTION_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]  # L, D, R, U

# (dx, dy) in plot coords for the arrowhead end, relative to cell center.
# Reversed y-axis: north (UP) = smaller y, south (DOWN) = larger y.
ACTION_ARROW_DXDY = [
    (-0.38,  0.0),   # LEFT
    ( 0.0,   0.38),  # DOWN
    ( 0.38,  0.0),   # RIGHT
    ( 0.0,  -0.38),  # UP
]

CELL_BG_COLOR     = "white"
CELL_BORDER       = "#a0a0a0"
HOLE_FILL         = "#d4d4d4"
GOAL_BORDER       = "#d4af37"
START_BORDER      = "#2ca02c"
BELIEF_COLOR_RGB  = (200, 30, 60)
AGENT_DISC_COLOR  = "#000000"
HALO_ON_COLOR     = "#8b0000"   # obs bit = 1 (hole detected)
HALO_OFF_COLOR    = "#ffd0d0"   # obs bit = 0


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def cell_center(state_idx: int) -> tuple[float, float]:
    """Plot-coords center of a grid cell. Returns (x, y); y-axis is reversed."""
    r, c = idx_to_cell(state_idx)
    return c + 0.5, r + 0.5


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _add_cell_rect(fig: go.Figure, r: int, c: int,
                   *, fill: str | None, border: str, width: float) -> None:
    fig.add_shape(type="rect",
                  x0=c, y0=r, x1=c + 1, y1=r + 1,
                  line=dict(color=border, width=width),
                  fillcolor=fill,
                  layer="below"
                  )


def _add_belief_overlay(fig: go.Figure, belief: list[float], max_b: float) -> None:
    if max_b <= 0:
        return
    for s in range(N_GRID_CELLS):
        b = belief[s]
        if b <= 1e-6:
            continue
        alpha = 0.85 * (b / max_b)
        r, c  = idx_to_cell(s)
        fig.add_shape(type="rect",
                      x0=c, y0=r, x1=c + 1, y1=r + 1,
                      line=dict(width=0),
                      fillcolor=f"rgba({BELIEF_COLOR_RGB[0]}, "
                                f"{BELIEF_COLOR_RGB[1]}, "
                                f"{BELIEF_COLOR_RGB[2]}, {alpha:.3f})",
                      layer="below"
                      )


def _add_agent_disc(fig: go.Figure, state: int) -> None:
    if state == TERMINAL_STATE:
        return  # terminal has no grid coordinate
    x, y = cell_center(state)
    r = 0.18
    fig.add_shape(type="circle",
                  x0=x - r, y0=y - r, x1=x + r, y1=y + r,
                  line=dict(color=AGENT_DISC_COLOR, width=1),
                  fillcolor=AGENT_DISC_COLOR
                  )


def _add_nsew_halo(fig: go.Figure, state: int, obs: int | None) -> None:
    # obs=None at t=0 (no prior observation has been received yet).
    if state == TERMINAL_STATE or obs is None:
        return
    bits  = obs_idx_to_bits(obs)  # (N, S, E, W)
    x, y  = cell_center(state)
    off   = 0.32
    rad   = 0.07
    # (offset_x, offset_y) per direction. Reversed y: N is -y, S is +y.
    positions = [(0.0, -off),  # N
                 (0.0,  off),  # S
                 ( off, 0.0),  # E
                 (-off, 0.0)]  # W
    for bit, (dx, dy) in zip(bits, positions):
        color = HALO_ON_COLOR if bit == 1 else HALO_OFF_COLOR
        fig.add_shape(type="circle",
                      x0=x + dx - rad, y0=y + dy - rad,
                      x1=x + dx + rad, y1=y + dy + rad,
                      line=dict(color="#555555", width=0.5),
                      fillcolor=color
                      )


def _add_action_arrow(fig: go.Figure, state: int, action: int) -> None:
    # Suppress at absorbing states — the chosen action is arbitrary
    # (any action absorbs into terminal with the same outcome).
    if state == TERMINAL_STATE or state == GOAL_INDEX or state in HOLE_INDICES:
        return
    x0, y0 = cell_center(state)
    dx, dy = ACTION_ARROW_DXDY[action]
    color  = ACTION_COLORS[action]
    fig.add_annotation(x=x0 + dx, y=y0 + dy,
                       ax=x0,     ay=y0,
                       xref="x",  yref="y",
                       axref="x", ayref="y",
                       showarrow=True,
                       arrowhead=3,
                       arrowsize=1.8,
                       arrowwidth=3.5,
                       arrowcolor=color
                       )


ADVERSARY_ADD_COLOR = "#a0522d"  # sienna — adversary added mass here
ADVERSARY_REM_COLOR = "#00a0a0"  # teal   — adversary removed mass here

REWARD_COLOR_RGB   = (40, 180, 90)   # green — per-cell reward heatmap

_REWARD_PER_CELL = None


def _get_reward_per_cell():
    """Cache the per-cell reward (action-independent) at first call."""
    global _REWARD_PER_CELL
    if _REWARD_PER_CELL is None:
        _REWARD_PER_CELL = build_R()[:, 0]
    return _REWARD_PER_CELL


def _add_reward_overlay(fig: go.Figure) -> None:
    """Static reward heatmap. Green translucent overlay per cell with
    alpha proportional to r(s) / max grid reward. Independent of step.
    Grid-scenario-specific: relies on build_R from frozenlake_problem.
    """
    reward = _get_reward_per_cell()
    max_r  = max(reward[s] for s in range(N_GRID_CELLS))
    if max_r <= 0:
        return
    for s in range(N_GRID_CELLS):
        r_val = reward[s]
        if r_val <= 0:
            continue
        alpha    = 0.7 * (r_val / max_r)
        row, col = idx_to_cell(s)
        fig.add_shape(type="rect",
                      x0=col, y0=row, x1=col + 1, y1=row + 1,
                      line=dict(width=0),
                      fillcolor=f"rgba({REWARD_COLOR_RGB[0]}, "
                                f"{REWARD_COLOR_RGB[1]}, "
                                f"{REWARD_COLOR_RGB[2]}, {alpha:.3f})",
                      layer="below"
                      )


def _add_flow_field(fig: go.Figure, record: dict, agent_cell: int) -> None:
    """Grid-scenario-specific. Uses cell_center and TERMINAL_STATE.
    Future non-grid scenarios should substitute a layer-appropriate renderer.

    For each cell s with root-level sims that picked the planner's chosen
    action, find the modal next cell s'*, and draw an arrow s → s'* whose
    width encodes per-cell confidence (count[s, s'*] / total_from_s).
    Skips agent's own cell (covered by the bold action arrow) and
    no-movement transitions (s'* == s).
    """
    events = record.get("planner_events", [])
    if not events:
        return

    chosen = record["action"]
    pairs: dict[int, dict[int, int]] = {}
    for e in events:
        if e.get("type") != "sim_step":
            continue
        if e.get("depth") != 0:
            continue
        if e.get("action_chosen") != chosen:
            continue
        s      = e["state"]
        s_next = e["s_next"]
        if s == TERMINAL_STATE or s == agent_cell:
            continue
        pairs.setdefault(s, {}).setdefault(s_next, 0)
        pairs[s][s_next] += 1

    for s, next_counts in pairs.items():
        total      = sum(next_counts.values())
        s_star     = max(next_counts, key=next_counts.get)
        confidence = next_counts[s_star] / total      # 0..1
        if s_star == TERMINAL_STATE or s_star == s:
            continue  # MVP: skip terminal-bound + no-movement
        x0, y0 = cell_center(s)
        x1, y1 = cell_center(s_star)
        width  = 0.5 + confidence * (4.0 - 0.5)
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                           xref="x", yref="y",
                           axref="x", ayref="y",
                           showarrow=True,
                           arrowhead=2,
                           arrowsize=1.0,
                           arrowwidth=width,
                           arrowcolor="#444"
                           )


def _add_adversary_chevrons(fig: go.Figure, record: dict) -> None:
    """Grid-scenario-specific. Uses cell_center and TERMINAL_STATE.
    Future non-grid scenarios need a layer-appropriate adversary renderer.

    For each next-state cell s' in S_in at the root, draw a small colored
    dot encoding the net belief-weighted mass shift the adversary applied
    under the chosen action a_t. Mulberry = mass added (worse for agent),
    teal = mass removed. Aggregates across source states s in S_in,
    weighted by belief b(s).

    Belief in robust_backup events is aligned with S_in (length |S_in|),
    not full N_STATES — see compute_belief at robust_pomcp.py:482-488.
    """
    events = record.get("planner_events", [])
    if not events:
        return

    root_id = next((e["active_node"] for e in events
                    if e.get("type") == "sim_start"), None)
    if root_id is None:
        return

    chosen  = record["action"]
    backups = [e for e in events
               if e.get("type") == "robust_backup"
               and e.get("history_node_id") == root_id
               and e.get("action") == chosen]
    if not backups:
        return

    last_backup = backups[-1]
    S_in_list   = last_backup["S_in"]
    belief      = last_backup["belief"]            # |S_in|-indexed
    trans_lp    = last_backup["trans_lp_results"]

    state_to_w = {s: belief[j] for j, s in enumerate(S_in_list)}
    delta_total: dict[int, float] = {}
    for lp in trans_lp:
        s = lp["state"]
        w = state_to_w.get(s, 0.0)
        if w <= 0:
            continue
        nom = lp["nominal_p"]
        opt = lp["optimal_p"]
        if len(opt) != len(S_in_list) or len(nom) != len(S_in_list):
            continue                                # defensive guard
        for i, s_prime in enumerate(S_in_list):
            delta_total[s_prime] = delta_total.get(s_prime, 0.0) \
                                   + w * (opt[i] - nom[i])

    for s_prime, delta in delta_total.items():
        if s_prime == TERMINAL_STATE:
            continue
        if abs(delta) <= 0.01:
            continue                                # absorbs LP noise
        r, c   = idx_to_cell(s_prime)
        x      = c + 0.78
        y      = r + 0.78
        rad    = 0.04 + min(abs(delta), 1.0) * (0.10 - 0.04)
        color  = ADVERSARY_ADD_COLOR if delta > 0 else ADVERSARY_REM_COLOR
        fig.add_shape(type="circle",
                      x0=x - rad, y0=y - rad,
                      x1=x + rad, y1=y + rad,
                      line=dict(color="#333333", width=0.5),
                      fillcolor=color
                      )


def build_grid_figure(record: dict, halo_obs: int | None,
                      show_flow: bool = True,
                      show_adversary: bool = True,
                      show_reward: bool = True) -> go.Figure:
    fig = go.Figure()

    # Layer 0: empty cells
    for r in range(N_ROWS):
        for c in range(N_COLS):
            _add_cell_rect(fig, r, c, fill=CELL_BG_COLOR, border=CELL_BORDER, width=1)

    # Layer 1: holes
    for hole_idx in HOLE_INDICES:
        r, c = idx_to_cell(hole_idx)
        _add_cell_rect(fig, r, c, fill=HOLE_FILL, border=CELL_BORDER, width=1)

    # Layer 2: goal cell border + star
    gr, gc = idx_to_cell(GOAL_INDEX)
    _add_cell_rect(fig, gr, gc, fill=None, border=GOAL_BORDER, width=3)
    fig.add_annotation(x=gc + 0.5, y=gr + 0.5,
                       text="★", showarrow=False,
                       font=dict(size=26, color=GOAL_BORDER)
                       )

    # Layer 3: start cell border
    sr, sc = idx_to_cell(START_INDEX)
    _add_cell_rect(fig, sr, sc, fill=None, border=START_BORDER, width=2)

    # Layer 3.5: reward heatmap (static, opt-in).
    if show_reward:
        _add_reward_overlay(fig)

    # Layer 4: belief (per-step max so each frame is legible)
    step_max = max(record["belief"][:N_GRID_CELLS]) or 1.0
    _add_belief_overlay(fig, record["belief"], step_max)

    # Layer 4.5: flow field — planner's modal next-cell per belief-supported state.
    if show_flow:
        _add_flow_field(fig, record, agent_cell=record["state"])

    # Layer 4.6: adversary chevrons — robust LP transition mass shifts.
    if show_adversary:
        _add_adversary_chevrons(fig, record)

    # Layer 5: agent disc + halo + action arrow
    _add_agent_disc(fig, record["state"])
    _add_nsew_halo(fig, record["state"], halo_obs)
    _add_action_arrow(fig, record["state"], record["action"])

    # Terminal mass readout (off-grid annotation)
    b_term = record["belief"][TERMINAL_STATE] if len(record["belief"]) > N_GRID_CELLS else 0.0
    if b_term > 1e-3:
        fig.add_annotation(x=N_COLS, y=N_ROWS + 0.4,
                           xref="x", yref="y",
                           text=f"terminal mass: {b_term:.3f}",
                           showarrow=False, xanchor="right",
                           font=dict(size=11, color="#555")
                           )

    fig.update_layout(
        xaxis=dict(range=[-0.1, N_COLS + 0.1],
                   showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[N_ROWS + 0.6, -0.1],   # reversed
                   showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        width=620,
        height=620,
        showlegend=False,
    )
    return fig


def build_qbar_figure(record: dict) -> go.Figure:
    q = record["q_robust"]
    chosen = record["action"]

    # None → 0.0 for display; greyed.
    values  = [v if v is not None else 0.0 for v in q]
    colors  = [ACTION_COLORS[a] if q[a] is not None else "#bbbbbb"
               for a in range(N_ACTIONS)]
    # Highlight chosen: keep its color; dim the others slightly.
    for a in range(N_ACTIONS):
        if a != chosen and q[a] is not None:
            colors[a] = _dim_hex(ACTION_COLORS[a], 0.45)

    fig = go.Figure(go.Bar(
        x=values,
        y=ACTION_NAMES,
        orientation="h",
        marker=dict(color=colors,
                    line=dict(color="#333333", width=1.2),
                    ),
        text=[f"{v:.4f}" if q[a] is not None else "—"
              for a, v in enumerate(values)],
        textposition="outside",
        hovertemplate="%{y}: Q_robust=%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Q_robust at root", font=dict(size=14)),
        xaxis=dict(title="value", fixedrange=True),
        yaxis=dict(autorange="reversed", fixedrange=True),
        margin=dict(l=60, r=80, t=30, b=30),
        width=440,
        height=240,
        showlegend=False,
    )
    fig.update_traces(cliponaxis=False)
    return fig


def _dim_hex(hex_color: str, factor: float) -> str:
    """Return rgba string with reduced alpha (factor in [0,1])."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{factor:.2f})"


# ---------------------------------------------------------------------------
# Step readout text
# ---------------------------------------------------------------------------

def build_step_readout(record: dict, halo_obs: int | None,
                       cumulative: float, total_steps: int
                       ) -> html.Div:
    state      = record["state"]
    a          = record["action"]
    is_at_goal = (state == GOAL_INDEX)
    is_at_hole = (state in HOLE_INDICES)

    rows: list = [
        html.Div(f"t = {record['t']} / {total_steps - 1}",
                 style={"fontWeight": "bold", "fontSize": "16px",
                        "marginBottom": "8px", "color": "#222"}),
    ]

    if is_at_goal or is_at_hole:
        label, color = (("REACHED GOAL", "#0a8a0a") if is_at_goal
                        else ("FELL IN HOLE", "#aa0000"))
        rows.append(html.Div(label,
                             style={"color": color, "fontWeight": "bold",
                                    "fontSize": "15px"}))
        rows.append(html.Div("next step absorbs into terminal",
                             style={"color": "#888", "fontSize": "11px",
                                    "marginBottom": "8px",
                                    "fontStyle": "italic"}))
    else:
        rows.append(html.Div([
            html.Span("action: ", style={"color": "#666"}),
            html.Span(ACTION_NAMES[a],
                      style={"color": ACTION_COLORS[a], "fontWeight": "bold"}),
        ]))

    if halo_obs is None:
        obs_text = "—   (initial belief, no obs yet)"
        obs_color = "#999"
    else:
        bits      = obs_idx_to_bits(halo_obs)
        obs_text  = "".join(str(b) for b in bits)
        obs_color = "#222"

    rows.extend([
        html.Div([
            html.Span("obs (NSEW): ", style={"color": "#666"}),
            html.Span(obs_text,
                      style={"fontFamily": "monospace", "color": obs_color}),
        ]),
        html.Div([
            html.Span("reward: ", style={"color": "#666"}),
            html.Span(f"{record['reward']:+.4f}",
                      style={"fontFamily": "monospace"}),
        ]),
        html.Div([
            html.Span("cumulative: ", style={"color": "#666"}),
            html.Span(f"{cumulative:+.4f}",
                      style={"fontFamily": "monospace", "color": "#222",
                             "fontWeight": "bold"}),
        ]),
        html.Div(f"true state: {record['state']} → {record['next_state']}",
                 style={"fontSize": "11px", "color": "#999",
                        "marginTop": "6px"}),
    ])

    return html.Div(rows, style={"padding": "4px 0", "lineHeight": "1.6"})


# ---------------------------------------------------------------------------
# Header helpers (chips + result badge)
# ---------------------------------------------------------------------------

def make_chip(label: str, value, *, accent: str | None = None) -> html.Div:
    return html.Div([
        html.Span(label,
                  style={"fontSize": "10px", "color": "#888",
                         "textTransform": "uppercase",
                         "letterSpacing": "0.05em"}),
        html.Span(str(value),
                  style={"fontSize": "13px", "marginLeft": "6px",
                         "fontFamily": "ui-monospace, monospace",
                         "color": accent or "#333",
                         "fontWeight": "600"}),
    ], style={"display": "inline-flex", "alignItems": "center",
              "padding": "4px 10px", "marginRight": "6px",
              "marginBottom": "6px", "backgroundColor": "#fafafa",
              "border": "1px solid #e3e3e3", "borderRadius": "12px"})


def result_badge(episode: list[dict]) -> html.Span:
    if not episode:
        return html.Span()
    last       = episode[-1]
    last_state = last["state"]
    next_state = last["next_state"]
    if next_state == TERMINAL_STATE:
        if last_state == GOAL_INDEX:
            label, bg, fg = "REACHED GOAL", "#e6f8e6", "#0a7a0a"
        elif last_state in HOLE_INDICES:
            label, bg, fg = "FELL IN HOLE", "#fde6e6", "#a00000"
        else:
            label, bg, fg = "ABSORBED",     "#f0f0f0", "#555"
    else:
        label, bg, fg = "TIMED OUT", "#fff3d6", "#9a6900"
    return html.Span(label, style={
        "padding": "4px 12px", "marginLeft": "12px",
        "backgroundColor": bg, "color": fg,
        "borderRadius": "12px", "fontWeight": "bold",
        "fontSize": "12px", "verticalAlign": "middle",
        "letterSpacing": "0.05em",
    })


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

CARD_STYLE = {
    "backgroundColor": "white",
    "padding":         "16px",
    "borderRadius":    "8px",
    "boxShadow":       "0 2px 8px rgba(0,0,0,0.06)",
    "border":          "1px solid #e8e8e8",
}

BUTTON_STYLE = {
    "padding":         "8px 18px",
    "fontSize":        "14px",
    "cursor":          "pointer",
    "border":          "1px solid #ccc",
    "borderRadius":    "4px",
    "backgroundColor": "white",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True,
                        help="Path to episode JSON from record_episode.py")
    parser.add_argument("--port",    type=int, default=8050)
    args = parser.parse_args()

    with args.episode.open() as f:
        data = json.load(f)
    metadata = data["metadata"]
    episode  = data["episode"]
    n_steps  = len(episode)

    # Cumulative returns per step for the readout.
    cum_returns: list[float] = []
    running = 0.0
    for rec in episode:
        running += rec["reward"]
        cum_returns.append(running)

    chips = html.Div([
        make_chip("scenario", metadata['scenario']),
        make_chip("h", metadata['horizon']),
        make_chip("plan_h",
                  metadata.get('plan_horizon') or metadata['horizon']),
        make_chip("budget", metadata['budget']),
        make_chip("p", metadata.get('p', 0.6)),
        make_chip("ε", metadata.get('epsilon', 0.05)),
        make_chip("ρ_T", metadata['rho_T']),
        make_chip("ρ_Z", metadata['rho_Z']),
        make_chip("η_obs", metadata['eta_obs']),
        make_chip("η_trans", metadata['eta_trans']),
        make_chip("seed", metadata['seed']),
        make_chip("total_R", f"{metadata['total_reward']:+.4f}",
                  accent="#0a7a0a" if metadata['total_reward'] > 0 else "#333"),
    ], style={"marginTop": "10px"})

    app = Dash(__name__)
    app.title = f"FrozenLake episode — {args.episode.name}"

    app.layout = html.Div([
        # Header
        html.Div(
            html.Div([
                html.H2([
                    "FrozenLake — ",
                    html.Span(args.episode.name,
                              style={"fontWeight": "normal", "color": "#666",
                                     "fontSize": "16px"}),
                    result_badge(episode),
                ], style={"margin": 0, "color": "#222",
                          "fontFamily": "system-ui, sans-serif",
                          "fontSize": "22px"}),
                chips,
            ], style={"maxWidth": "1180px", "margin": "0 auto"}),
            style={"backgroundColor": "white", "padding": "18px 24px",
                   "borderBottom": "1px solid #e0e0e0"}
        ),

        # Main row: grid (left) + side panel (right)
        html.Div([
            html.Div([
                dcc.Graph(id="grid-fig",
                          config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "marginRight": "16px"}),

            html.Div([
                html.Div(id="step-readout"),
                html.Div(style={"height": "12px"}),
                dcc.Graph(id="qbar-fig",
                          config={"displayModeBar": False}),
                html.Div([
                    # Flow toggle + its explanation.
                    dcc.Checklist(
                        id="flow-toggle",
                        options=[{
                            "label": html.Span([
                                html.Span("→", style={"color": "#444",
                                                       "fontSize": "16px",
                                                       "fontWeight": "bold",
                                                       "marginRight": "6px",
                                                       "verticalAlign": "middle"}),
                                " flow field",
                            ]),
                            "value": "flow",
                        }],
                        value=["flow"],
                        style={"fontSize": "13px"}
                    ),
                    html.Div(
                        "grey arrow: planner's modal next-cell prediction",
                        style={"fontSize": "11px", "color": "#666",
                               "marginLeft": "24px", "marginTop": "2px"}
                    ),
                    # Adversary toggle + its 2-line explanation.
                    dcc.Checklist(
                        id="adv-toggle",
                        options=[{
                            "label": html.Span([
                                html.Span(style={"display": "inline-block",
                                                 "width": "10px", "height": "10px",
                                                 "backgroundColor": ADVERSARY_ADD_COLOR,
                                                 "borderRadius": "50%",
                                                 "marginRight": "2px",
                                                 "verticalAlign": "middle"}),
                                html.Span(style={"display": "inline-block",
                                                 "width": "10px", "height": "10px",
                                                 "backgroundColor": ADVERSARY_REM_COLOR,
                                                 "borderRadius": "50%",
                                                 "marginRight": "6px",
                                                 "verticalAlign": "middle"}),
                                " adversary chevrons",
                            ]),
                            "value": "adv",
                        }],
                        value=["adv"],
                        style={"fontSize": "13px", "marginTop": "10px"}
                    ),
                    html.Div([
                        html.Div([
                            html.Span(style={"display": "inline-block",
                                             "width": "8px", "height": "8px",
                                             "backgroundColor": ADVERSARY_ADD_COLOR,
                                             "borderRadius": "50%",
                                             "marginRight": "8px",
                                             "verticalAlign": "middle"}),
                            "sienna: cell worse than nominal",
                        ]),
                        html.Div([
                            html.Span(style={"display": "inline-block",
                                             "width": "8px", "height": "8px",
                                             "backgroundColor": ADVERSARY_REM_COLOR,
                                             "borderRadius": "50%",
                                             "marginRight": "8px",
                                             "verticalAlign": "middle"}),
                            "teal:   cell better than nominal",
                        ]),
                    ], style={"fontSize": "11px", "color": "#666",
                              "marginLeft": "24px", "marginTop": "2px",
                              "lineHeight": "1.6"}),
                    # Reward toggle + its explanation.
                    dcc.Checklist(
                        id="reward-toggle",
                        options=[{
                            "label": html.Span([
                                html.Span(style={"display": "inline-block",
                                                 "width": "12px", "height": "12px",
                                                 "backgroundColor": f"rgba({REWARD_COLOR_RGB[0]},"
                                                                    f"{REWARD_COLOR_RGB[1]},"
                                                                    f"{REWARD_COLOR_RGB[2]},0.7)",
                                                 "marginRight": "8px",
                                                 "verticalAlign": "middle"}),
                                " reward heatmap",
                            ]),
                            "value": "reward",
                        }],
                        value=["reward"],   # on by default
                        style={"fontSize": "13px", "marginTop": "10px"}
                    ),
                    html.Div(
                        "green fill: per-cell reward (darker = higher)",
                        style={"fontSize": "11px", "color": "#666",
                               "marginLeft": "24px", "marginTop": "2px"}
                    ),
                ], style={"marginTop": "12px",
                          "paddingTop": "10px",
                          "borderTop": "1px solid #eee"}),
                html.Div(id="flow-availability-note",
                         style={"fontSize": "11px", "color": "#999",
                                "fontStyle": "italic",
                                "marginTop": "4px"}),
            ], style={**CARD_STYLE, "minWidth": "470px"}),
        ], style={"display": "flex", "flexDirection": "row",
                  "alignItems": "flex-start", "maxWidth": "1180px",
                  "margin": "20px auto 0 auto"}),

        # Slider + buttons
        html.Div(
            html.Div([
                html.Button("◀ Prev", id="btn-prev", n_clicks=0,
                            style={**BUTTON_STYLE, "marginRight": "16px"}),
                html.Div([
                    dcc.Slider(id="step-slider",
                               min=0, max=max(0, n_steps - 1), step=1, value=0,
                               marks={t: str(t) for t in range(n_steps)},
                               tooltip={"placement": "bottom"}
                               ),
                ], style={"flex": "1"}),
                html.Button("Next ▶", id="btn-next", n_clicks=0,
                            style={**BUTTON_STYLE, "marginLeft": "16px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            style={**CARD_STYLE, "maxWidth": "1180px",
                   "margin": "16px auto 24px auto"}
        ),
    ], style={"backgroundColor": "#f4f5f7", "minHeight": "100vh",
              "fontFamily": "system-ui, -apple-system, sans-serif",
              "margin": 0, "padding": "0 0 24px 0"})

    @app.callback(
        Output("grid-fig",               "figure"),
        Output("qbar-fig",               "figure"),
        Output("step-readout",           "children"),
        Output("step-slider",            "value"),
        Output("flow-availability-note", "children"),
        Input("step-slider",             "value"),
        Input("btn-prev",                "n_clicks"),
        Input("btn-next",                "n_clicks"),
        Input("flow-toggle",             "value"),
        Input("adv-toggle",              "value"),
        Input("reward-toggle",           "value"),
    )
    def _update(slider_val, _prev_clicks, _next_clicks,
                flow_values, adv_values, reward_values):
        trig = ctx.triggered_id
        if trig == "btn-prev":
            step_idx = max(0, (slider_val or 0) - 1)
        elif trig == "btn-next":
            step_idx = min(n_steps - 1, (slider_val or 0) + 1)
        else:
            step_idx = slider_val or 0
        rec            = episode[step_idx]
        halo_obs       = episode[step_idx - 1]["obs"] if step_idx > 0 else None
        show_flow      = "flow"   in (flow_values   or [])
        show_adversary = "adv"    in (adv_values    or [])
        show_reward    = "reward" in (reward_values or [])

        # Short, intuitive availability notes.
        notes: list[str] = []
        events = rec.get("planner_events")
        if not events:
            notes.append("events stripped — re-record without --strip-events")
        else:
            root_id = next((e["active_node"] for e in events
                            if e.get("type") == "sim_start"), None)
            any_robust = any(
                any(lp.get("radius", 0) > 0
                    for lp in e.get("trans_lp_results", []))
                for e in events
                if e.get("type") == "robust_backup"
                and e.get("history_node_id") == root_id
                and e.get("action") == rec["action"]
            )
            if not any_robust:
                if metadata.get("rho_T", 0.0) > 0:
                    notes.append("no hole-adjacent cells in current support")
                else:
                    notes.append("ρ_T = 0 — try --robust")
        note = [html.Div(line) for line in notes] if notes else ""

        return (
            build_grid_figure(rec, halo_obs, show_flow,
                              show_adversary, show_reward),
            build_qbar_figure(rec),
            build_step_readout(rec, halo_obs,
                               cum_returns[step_idx], n_steps),
            step_idx,
            note,
        )

    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
