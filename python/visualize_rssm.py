"""Visualize RSSM imagined rollouts against ground truth (stage-2 gate).

Loads a trained world-model checkpoint, filters the posterior over a seed
window of a REAL episode, then rolls the PRIOR forward with the episode's
actual actions and decodes the predicted grids. Renders ground-truth vs
imagined frame strips — the visual test of DREAMERV4_update.md jalon S2
("rollouts imaginés à 10 steps restent cohérents visuellement").

Usage:
    .venv/Scripts/python.exe python/visualize_rssm.py \
        --run data/worldmodel/<stamp> --episode data/human/session1/episode_0003.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rssm import WorldModel, one_hot_actions

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "worldmodel"

# grid channel -> RGB tint for the composite rendering
CHANNEL_COLORS = {
    0: (0.55, 0.55, 0.55),   # walls
    1: (0.35, 0.25, 0.15),   # obstacles
    2: (0.1, 0.2, 0.4),      # pits
    3: (0.0, 1.0, 0.0),      # player
    4: (1.0, 0.15, 0.15),    # enemies
    5: (1.0, 1.0, 0.0),      # projectiles
    6: (0.2, 0.5, 1.0),      # pickups
    7: (0.9, 0.9, 1.0),      # doors
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="worldmodel run dir (default: most recent)")
    parser.add_argument("--episode", required=True, help="episode .npz file")
    parser.add_argument("--seed-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def resolve_run(run: str | None) -> Path:
    if run:
        return Path(run)
    candidates = sorted(
        (d for d in DEFAULT_ROOT.glob("*")
         if list(d.glob("worldmodel_*.pt"))),
        key=lambda d: max(p.stat().st_mtime for p in d.glob("worldmodel_*.pt")),
    )
    if not candidates:
        raise SystemExit(f"No worldmodel checkpoints under {DEFAULT_ROOT}")
    return candidates[-1]


def composite(frame: np.ndarray) -> np.ndarray:
    """(C,H,W) float grid -> (H,W,3) RGB overlay."""
    c, h, w = frame.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    for ch, color in CHANNEL_COLORS.items():
        if ch >= c:
            continue
        layer = np.clip(frame[ch], 0.0, 1.0)[..., None]
        img = np.maximum(img, layer * np.asarray(color, dtype=np.float32))
    return img


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = resolve_run(args.run)
    ckpts = sorted(run_dir.glob("worldmodel_*.pt"))
    payload = torch.load(ckpts[-1], map_location=device)
    cfg = payload["config"]
    model = WorldModel(
        action_nvec=cfg["action_nvec"],
        grid_channels=cfg["grid_channels"],
        grid_hw=tuple(cfg["grid_hw"]),
        player_features=cfg["player_features"],
        max_entities=cfg["max_entities"],
        entity_dim=cfg["entity_dim"],
        free_nats=cfg.get("free_nats", 3.0),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    print(f"Loaded {ckpts[-1].name} (trained step {payload['step']})")

    episode = np.load(args.episode, allow_pickle=False)
    grids = episode["grid"]
    players = episode["player"]
    actions = episode["action"]
    meta = json.loads(str(episode["meta"]))

    seed_len = args.seed_len
    horizon = args.horizon
    total_needed = seed_len + horizon
    if len(grids) < total_needed:
        raise SystemExit(
            f"Episode has {len(grids)} steps < seed({seed_len}) + "
            f"horizon({horizon})."
        )

    with torch.no_grad():
        obs_grids = torch.as_tensor(
            grids[:seed_len], dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_players = torch.as_tensor(
            players[:seed_len], dtype=torch.float32, device=device
        ).unsqueeze(0)
        obs_entities = torch.as_tensor(
            episode["entities"][:seed_len], dtype=torch.float32, device=device
        ).unsqueeze(0)

        embeds = model.encoder(obs_grids, obs_players, obs_entities)
        act_seed = one_hot_actions(
            torch.as_tensor(actions[:seed_len], device=device),
            cfg["action_nvec"],
        ).unsqueeze(0)
        hiddens, _priors, _posts, samples = model.core.observe(
            embeds, act_seed
        )
        h = hiddens[:, -1]
        z = samples[:, -1]

        imagined = []
        player_pred_all = []
        state_dim = model.core.deter + model.core.stoch
        for t in range(horizon):
            action_t = one_hot_actions(
                torch.as_tensor(
                    actions[seed_len + t: seed_len + t + 1], device=device
                ),
                cfg["action_nvec"],
            )
            inp = torch.cat([z, action_t], dim=-1)
            h = model.core.cell(inp, h)
            prior_logits = model.core.prior_net(h)
            z = torch.nn.functional.one_hot(
                prior_logits.argmax(dim=-1), model.core.stoch
            ).float()
            state = torch.cat([h, z], dim=-1).reshape(state_dim)
            grid_pred = (
                model.grid_head(state)
                .reshape(cfg["grid_channels"], *cfg["grid_hw"])
                .cpu().numpy()
            )
            player_pred = model.player_head(state).cpu().numpy()
            imagined.append(grid_pred)
            player_pred_all.append(player_pred)

    imagined = np.stack(imagined)
    truth = grids[seed_len: total_needed]
    pos_slice = slice(14, 16)
    pos_err = float(
        np.mean(
            (np.stack(player_pred_all)[:, pos_slice]
             - players[seed_len: total_needed][:, pos_slice]) ** 2
        )
    )

    fig, axes = plt.subplots(
        2, horizon, figsize=(1.6 * horizon, 4.2)
    )
    fig.suptitle(
        f"RSSM rollout — {Path(args.episode).name} ({meta.get('episode_reason')}) "
        f"| seed {seed_len}, horizon {horizon}\n"
        f"player pos MSE over horizon: {pos_err:.5f}",
        fontsize=9,
    )
    for t in range(horizon):
        axes[0, t].imshow(composite(truth[t]), interpolation="nearest")
        axes[0, t].set_title(f"t+{t + 1}", fontsize=7)
        axes[1, t].imshow(composite(imagined[t]), interpolation="nearest")
        for ax in (axes[0, t], axes[1, t]):
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("réel", fontsize=8)
    axes[1, 0].set_ylabel("imaginé", fontsize=8)
    fig.subplots_adjust(wspace=0.05, hspace=0.12)

    out_path = Path(args.out) if args.out else run_dir / (
        f"rollout_{Path(args.episode).stem}.png"
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
