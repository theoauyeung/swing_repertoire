"""Standalone swing-card and usage-heatmap generator.

Usage:
    python src/uncommited/swing_cards.py             # swing cards for Arraez + Judge
    python src/uncommited/swing_cards.py --heatmap   # usage heatmaps for Arraez + Judge
    python src/uncommited/swing_cards.py "Name"      # swing card for one hitter
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Wedge, Circle
from matplotlib.colors import LinearSegmentedColormap
import dataframe_image as dfi

ROOT  = next(p for p in [Path(__file__).parents[2], Path.cwd(), Path.cwd().parent]
             if (p / 'data').exists())
DATA  = ROOT / 'data'
PLOTS = ROOT / 'results' / 'plots'

# ---------------------------------------------------------------------------
# Colour palette (light mode)
# ---------------------------------------------------------------------------
BG, INK, MUT = 'white', '#1a1a1a', '#888888'
GRID         = '#dddddd'
LIME         = '#2a6800'
ORNG         = '#b86a18'
RED          = '#cc2222'
TAN          = '#b89050'

FT_DIV = LinearSegmentedColormap.from_list('ft_div', ['#30a2da', '#FFFFFF', '#fc4f30'])
FASTBALLS = {'FF', 'SI', 'FC', 'FT', 'FA'}

# ---------------------------------------------------------------------------
# Data loading (lazy — only what each function needs)
# ---------------------------------------------------------------------------
_swc    = None
_league = None


def _load_swc():
    global _swc, _league
    if _swc is not None:
        return _swc
    swings = pd.read_parquet(DATA / 'swings_model.parquet')
    assign = pd.read_parquet(DATA / 'cluster_assignments.parquet',
                             columns=['play_id', 'cluster', 'resp_max'])
    _swc   = swings.merge(assign, on='play_id', how='inner')
    _league = {c: _swc[c].mean() for c in
               ['bat_speed', 'vert_attack_angle', 'horz_attack_angle_pull',
                'swing_path_tilt', 'swing_length']}
    return _swc


# ---------------------------------------------------------------------------
# Metric panel renderers
# ---------------------------------------------------------------------------
def _off(ax):
    ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


def _p_aa(ax, r):
    _off(ax); ax.set_aspect('equal'); ax.set_xlim(-1.15, 1.25); ax.set_ylim(-0.6, 1.05)
    a = np.radians(r.vaa)
    ax.add_patch(Circle((-0.85, 0), 0.14, color=TAN, zorder=3))
    ax.plot([-0.85, 1.05], [0, 0], color=INK, ls=(0, (4, 3)), lw=1.4, alpha=.8)
    ax.add_patch(FancyArrowPatch((-0.85, 0), (-0.85 + 1.9*np.cos(a), 1.9*np.sin(a)),
                 arrowstyle='-|>', mutation_scale=17, color=RED, lw=3, zorder=4))
    ax.text(0.5, 0.94, f"{r.vaa:.0f}°", transform=ax.transAxes, ha='center', va='top',
            color=RED, fontsize=15, fontweight='bold')


def _p_dir(ax, r):
    _off(ax); ax.set_aspect('equal'); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.0, 1.15)
    s = -1.0 if getattr(r, 'stand', 'R') == 'L' else 1.0
    ang = np.radians(r.haa) * s
    ax.plot([-0.9, 0.9], [-0.25, -0.25], color=TAN, lw=4, solid_capstyle='round')
    ax.plot([0, 0], [-0.25, 0.95], color=MUT, lw=1.1, ls=(0, (3, 3)), alpha=0.7)
    ax.add_patch(Circle((0, -0.25), 0.09, color='#3a5', zorder=3))
    ax.add_patch(FancyArrowPatch((0, -0.25), (-np.sin(ang)*1.05, -0.25 + np.cos(ang)*1.05),
                 arrowstyle='-|>', mutation_scale=16, color=RED, lw=3, zorder=4))
    pull_x, oppo_x = ((-1.12, 0.72) if s > 0 else (0.72, -1.12))
    ax.text(pull_x, 0.55, 'PULL', color=LIME, fontsize=7.5, fontweight='bold')
    ax.text(oppo_x, 0.55, 'OPPO', color=MUT,  fontsize=7.5, fontweight='bold')
    hand_lbl = 'LHH' if getattr(r, 'stand', 'R') == 'L' else 'RHH'
    ax.text(0.97, 0.97, hand_lbl, transform=ax.transAxes, ha='right', va='top',
            color=MUT, fontsize=6.5, style='italic')
    ax.text(0.5, 1.02, f"{abs(r.haa):.0f}° {'PULL' if r.haa >= 0 else 'OPPO'}",
            transform=ax.transAxes, ha='center', va='top',
            color=LIME, fontsize=13, fontweight='bold')


def _p_tilt(ax, r):
    _off(ax); ax.set_aspect('equal'); ax.set_xlim(-0.2, 2.1); ax.set_ylim(-1.5, 0.6)
    t = r.tilt
    ax.add_patch(Wedge((0, 0), 1.5, -t, 0, facecolor=ORNG, alpha=.30))
    ax.plot([0, 2.0], [0, 0], color=INK, ls=(0, (4, 3)), lw=1.3, alpha=.8)
    ax.plot([0, 2.0*np.cos(np.radians(t))], [0, -2.0*np.sin(np.radians(t))],
            color=LIME, lw=3.2, solid_capstyle='round')
    ax.text(1.75, 0.12, 'FLAT', color=MUT, fontsize=7)
    ax.text(1.55, -1.35, 'STEEP', color=MUT, fontsize=7)
    ax.text(0.5, 0.98, f"{r.tilt:.0f}°", transform=ax.transAxes, ha='center', va='top',
            color=ORNG, fontsize=15, fontweight='bold')


def _p_speed(ax, r):
    _off(ax)
    dev = r.speed - _league['bat_speed']
    ax.set_xlim(-20, 20); ax.set_ylim(0, 1)
    col = LIME if dev >= 0 else RED
    ax.barh(0.45, dev, height=0.32, left=0, color=col, align='center', linewidth=0, zorder=3)
    ax.axvline(0, color=INK, lw=1.8, ls=(0, (2, 2)), zorder=4)
    ax.text(0.5, 0.96, f"{r.speed:.0f} mph", transform=ax.transAxes, ha='center', va='top',
            color=INK, fontsize=14, fontweight='bold')
    ax.text(0.5, 0.06, f"{dev:+.0f} vs. avg", transform=ax.transAxes, ha='center', va='bottom',
            color=col, fontsize=8, fontweight='bold')


def _p_length(ax, r):
    _off(ax)
    dev = r.length - _league['swing_length']
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(0, 1)
    col = LIME if dev >= 0 else RED
    ax.barh(0.45, dev, height=0.32, left=0, color=col, align='center', linewidth=0, zorder=3)
    ax.axvline(0, color=INK, lw=1.8, ls=(0, (2, 2)), zorder=4)
    ax.text(0.5, 0.96, f"{r.length:.2f} ft", transform=ax.transAxes, ha='center', va='top',
            color=INK, fontsize=14, fontweight='bold')
    ax.text(0.5, 0.06, f"{dev:+.2f} vs. avg", transform=ax.transAxes, ha='center', va='bottom',
            color=col, fontsize=8, fontweight='bold')


# Column definitions: (header, renderer, subtitle)
_LABEL_W = 0.75
_CELL_W  = 1.35
_CELL_H  = 1.75
_HDR_H   = 0.42
_GAP_W   = 0.4
_TOP_H   = 0.45
_BOT_H   = 0.1
_NC      = 5


def _cols():
    return [
        ('ATTACK\nANGLE',     _p_aa,
         f"MLB {_league['vert_attack_angle']:.0f}°"),
        ('ATTACK\nDIRECTION', _p_dir,
         f"MLB {abs(_league['horz_attack_angle_pull']):.0f}° "
         f"{'PULL' if _league['horz_attack_angle_pull'] >= 0 else 'OPPO'}"),
        ('SWING\nTILT',       _p_tilt,
         f"MLB {_league['swing_path_tilt']:.0f}°"),
        ('BAT\nSPEED',        _p_speed,
         f"MLB avg {_league['bat_speed']:.0f} mph"),
        ('SWING\nLENGTH',     _p_length,
         f"MLB avg {_league['swing_length']:.2f} ft"),
    ]


def swing_cards(name, save=None):
    """Shapes as rows, metrics as columns. Two side-by-side halves for k > 3."""
    swc = _load_swc()
    d   = swc[swc.batter_full_name.str.contains(name, case=False, na=False)]
    who = d.batter_full_name.unique()
    if len(who) != 1:
        raise ValueError(f'{name!r} matches {list(who)} — be more specific')
    full, stand = who[0], d.batter_stand.mode().iloc[0]

    g = (d.groupby('cluster').agg(
            n=('play_id', 'size'),
            tilt=('swing_path_tilt', 'mean'),
            speed=('bat_speed', 'mean'),
            vaa=('vert_attack_angle', 'mean'),
            haa=('horz_attack_angle_pull', 'mean'),
            length=('swing_length', 'mean'),
         ).reset_index())
    g['weight'] = g.n / g.n.sum()
    g['stand']  = stand
    g = g.sort_values('weight', ascending=False).reset_index(drop=True)
    k = len(g)

    cols4      = _cols()
    n_halves   = 2 if k > 3 else 1
    n_per_half = (k + 1) // 2 if n_halves == 2 else k
    halves = ([g.iloc[:n_per_half].reset_index(drop=True),
               g.iloc[n_per_half:].reset_index(drop=True)]
              if n_halves == 2 else [g.reset_index(drop=True)])

    half_w = _LABEL_W + _NC * _CELL_W
    fig_w  = half_w * n_halves + _GAP_W * (n_halves - 1)
    n_rows = n_per_half
    fig_h  = _TOP_H + _HDR_H + n_rows * _CELL_H + _BOT_H

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 1 - 0.2 / fig_h,
             f"{full}   ·   Bats {stand}   ·   {k} swing shapes",
             ha='center', va='top', fontsize=13, fontweight='bold', color=INK)

    for h, half_g in enumerate(halves):
        nh = len(half_g)
        x0_in    = h * (half_w + _GAP_W)
        gs_left  = (x0_in + _LABEL_W) / fig_w
        gs_right = (x0_in + half_w) / fig_w
        gs_top   = 1 - (_TOP_H + _HDR_H) / fig_h
        gs_bot   = _BOT_H / fig_h

        gs = gridspec.GridSpec(nh, _NC, figure=fig,
                               left=gs_left, right=gs_right,
                               top=gs_top, bottom=gs_bot,
                               wspace=0.06, hspace=0.45)

        cell_w_fig = (gs_right - gs_left) / _NC
        for c, (title, fn, avg) in enumerate(cols4):
            cx = gs_left + (c + 0.5) * cell_w_fig
            fig.text(cx, 1 - (_TOP_H + _HDR_H * 0.35) / fig_h,
                     title.replace('\n', ' '),
                     ha='center', va='center', fontsize=8,
                     fontweight='bold', color='#333333')
            if avg:
                fig.text(cx, 1 - (_TOP_H + _HDR_H * 0.82) / fig_h,
                         avg, ha='center', va='center', fontsize=6.5, color=MUT)

        row_h_fig = (gs_top - gs_bot) / nh
        label_cx  = (x0_in + _LABEL_W * 0.5) / fig_w

        for row_i, (_, r) in enumerate(half_g.iterrows()):
            row_cy = gs_top - (row_i + 0.5) * row_h_fig
            fig.text(label_cx, row_cy + 0.016,
                     f"shape {int(r.cluster)}",
                     ha='center', va='bottom', fontsize=9,
                     fontweight='bold', color=INK)
            fig.text(label_cx, row_cy - 0.016,
                     f"{r.weight * 100:.0f}%",
                     ha='center', va='top', fontsize=8.5, color=MUT)
            for c, (title, fn, avg) in enumerate(cols4):
                ax = fig.add_subplot(gs[row_i, c])
                fn(ax, r)

    if save:
        fig.savefig(save, dpi=130, facecolor=BG, bbox_inches='tight')
        print(f'wrote {save}')
    return fig


# ---------------------------------------------------------------------------
# Usage heatmap
# ---------------------------------------------------------------------------
def usage_heatmap(name, save=None):
    """Cluster-usage heatmap by game state (P(cluster | state), columns sum to 100%)."""
    swc = _load_swc()
    d   = swc[swc.batter_full_name.str.contains(name, case=False, na=False)]
    names = d.batter_full_name.unique()
    if len(names) == 0:
        raise ValueError(f'no batter matches {name!r}')
    if len(names) > 1:
        raise ValueError(f'{name!r} matches {len(names)} batters: {list(names)}')
    full = names[0]
    k    = int(d.cluster.max()) + 1

    b, s, o, pt = d.balls, d.strikes, d.outs_when_up, d.pitch_type
    on1, on2, on3 = d.on_1b_id.notna(), d.on_2b_id.notna(), d.on_3b_id.notna()
    is_fb = pt.isin(FASTBALLS)

    masks = {
        'All':     None,
        '0-0':     (b == 0) & (s == 0),
        '2 strk':  s == 2,
        '0-2':     (b == 0) & (s == 2),
        'Full':    (b == 3) & (s == 2),
        'vs FB':   is_fb,
        'vs Sec':  pt.notna() & ~is_fb,
        'Empty':   ~(on1 | on2 | on3),
        'RISP':    on2 | on3,
        '0 out':   o == 0,
        '1 out':   o == 1,
        '2 out':   o == 2,
    }

    def dist(m):
        return ((d if m is None else d[m]).cluster
                .value_counts(normalize=True).reindex(range(k), fill_value=0) * 100)

    usage = pd.DataFrame({nm: dist(m) for nm, m in masks.items()})
    usage.index.name = 'cluster'

    styled = (usage.style
              .background_gradient(cmap=FT_DIV, axis=0, low=0.1, high=0.1)
              .format('{:.0f}%')
              .set_caption(f'{full} - cluster usage % by game state')
              .set_table_styles([
                  {'selector': '',
                   'props': [('background-color', 'white'), ('color', '#3c3c3c'),
                             ('font-family', 'sans-serif'), ('border-collapse', 'collapse')]},
                  {'selector': 'th',
                   'props': [('background-color', 'white'), ('color', '#3c3c3c'),
                             ('border', '1px solid #cbcbcb'), ('font-weight', 'bold')]},
                  {'selector': 'td', 'props': [('border', '1px solid #ececec')]},
                  {'selector': 'caption',
                   'props': [('caption-side', 'top'), ('color', '#3c3c3c'),
                             ('font-size', '13px'), ('font-weight', 'bold'),
                             ('padding-bottom', '6px')]},
              ]))

    if save:
        dfi.export(styled, str(save), table_conversion='matplotlib')
        print(f'wrote {save}')
    return styled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    heatmap_mode = '--heatmap' in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith('--')]

    cards_dir = PLOTS / 'swing_cards'
    hm_dir    = PLOTS / 'usage_heatmap'
    cards_dir.mkdir(parents=True, exist_ok=True)
    hm_dir.mkdir(parents=True, exist_ok=True)

    _load_swc()  # preload once

    targets = names if names else ['Arraez', 'Judge']

    if heatmap_mode:
        for name in targets:
            slug = name.lower().replace(' ', '_')
            usage_heatmap(name, save=hm_dir / f'usage_heatmap_{slug}.png')
    else:
        for name in targets:
            slug = name.lower().replace(' ', '_')
            fig = swing_cards(name, save=cards_dir / f'swing_cards_{slug}.png')
            plt.close(fig)
