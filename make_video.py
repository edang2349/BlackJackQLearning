"""
Records the best trained agent playing Blackjack to videos/blackjack_agent.mp4.

    python make_video.py                # 12 hands
    python make_video.py --hands 20 --seed 3

Needs results/best_agent.json (written by experiments.py) and ffmpeg
(on PATH, or `pip install imageio-ffmpeg`).

Blackjack's own render only shows the player sum and the dealer's up card,
so a caption strip is added under each frame with the agent's action, its
Q-values, and the outcome of the hand.
"""

import argparse
import json
import os
import shutil
import subprocess

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from qlearning import ACTION_NAMES, QLearningAgent, make_env

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 10
HOLD = {"deal": 12, "action": 12, "result": 20}  # frames each kind of moment is shown
CAPTION_H = 120
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
FONT_BOLD = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it or run `pip install imageio-ffmpeg`")


def load_agent(path):
    with open(path) as f:
        d = json.load(f)
    agent = QLearningAgent(alpha=d["alpha"], gamma=d["gamma"], q_init=d["q_init"], epsilon=0.0)
    for s, v in d["q"]:
        agent.q[tuple(s)] = np.array(v, dtype=float)
    return agent, d["config"]


def captioned(frame, lines):
    """Frame from env.render() with a caption strip below it."""
    h, w, _ = frame.shape
    img = Image.new("RGB", (w, h + CAPTION_H), (32, 40, 28))
    img.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(img)
    y = h + 10
    for i, text in enumerate(lines):
        font = ImageFont.truetype(FONT_BOLD if i == 0 else FONT, 17 if i == 0 else 14)
        draw.text((14, y), text, fill=(255, 255, 255) if i == 0 else (215, 222, 200), font=font)
        y += 26 if i == 0 else 21
    return np.asarray(img)


def hand_str(cards):
    return " ".join("A" if c == 1 else str(c) for c in cards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--agent", default=os.path.join(HERE, "results", "best_agent.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "videos", "blackjack_agent.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    agent, cfg = load_agent(args.agent)
    env = make_env(render_mode="rgb_array")
    game = env.unwrapped
    obs, _ = env.reset(seed=args.seed)
    title = "Q-learning agent (α=%g, γ=%g) - greedy policy" % (cfg["alpha"], cfg["gamma"])

    frames = []
    record = {"win": 0, "draw": 0, "loss": 0}
    total = 0.0
    for hand in range(1, args.hands + 1):
        if hand > 1:
            obs, _ = env.reset()
        head = "Hand %d/%d    total reward so far: %+g" % (hand, args.hands, total)
        frames += [captioned(env.render(), [head, title,
                   "Dealt: %s    (sum %d, dealer shows %s)" % (hand_str(game.player), obs[0],
                                                               "A" if obs[1] == 1 else obs[1])])] * HOLD["deal"]
        while True:
            q = agent.q[obs]
            action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            line = "Agent: %s    Q(stick)=%+.2f  Q(hit)=%+.2f" % (ACTION_NAMES[action].upper(), q[0], q[1])
            if not (terminated or truncated):
                frames += [captioned(env.render(), [head, line,
                           "Player cards: %s" % hand_str(game.player)])] * HOLD["action"]
                continue
            total += reward
            outcome = "WIN" if reward > 0 else "DRAW" if reward == 0 else "LOSS"
            record[outcome.lower()] += 1
            if obs[0] > 21:
                detail = "Player busts with %d (%s)" % (obs[0], hand_str(game.player))
            else:
                dealer_sum = sum(game.dealer) + (10 if 1 in game.dealer and sum(game.dealer) + 10 <= 21 else 0)
                detail = "Dealer: %s = %d%s" % (hand_str(game.dealer), dealer_sum,
                                                " (bust)" if dealer_sum > 21 else "")
            frames += [captioned(env.render(), ["%s  %s (%+g)" % (head.split("    ")[0], outcome, reward),
                                                line, detail])] * HOLD["result"]
            break
    env.close()

    h, w, _ = frames[0].shape
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (w, h), "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print("Wrote %s (%d hands: %d W / %d D / %d L, total reward %+g, %.0fs)"
          % (args.out, args.hands, record["win"], record["draw"], record["loss"], total, len(frames) / FPS))


if __name__ == "__main__":
    main()
