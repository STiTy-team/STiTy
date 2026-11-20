"""Train the whisper-head language head on cached encoder frames (short_features.py).

Train = features/{fleurs,yodas}_train_*.pt, dev = features/{fleurs,yodas}_dev_*.pt. Batches are
drawn with weights that balance (language, duration bucket). Early stopping on dev accuracy for
clips under 1 s; the best head is saved with its language order.

usage: uv run --project bench python core/research/lid_module/short_train.py FEATURE_DIR OUT.pt
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from core.components.langid.audio.head import LidHead  # noqa: E402

LANGS = ["ko", "en", "ja", "es", "zh", "fr", "de"]
EDGES = [0.6, 1.0, 1.5]


def bucket(d):
    return sum(d >= e for e in EDGES)


def load(paths):
    frames, offs, labels, durs = [], [0], [], []
    for p in paths:
        d = torch.load(p)
        if d["lang"] not in LANGS or not len(d["n_frames"]):
            continue
        frames.append(d["frames"])
        for n in d["n_frames"].tolist():
            offs.append(offs[-1] + n)
        labels += [LANGS.index(d["lang"])] * len(d["n_frames"])
        durs += d["dur"].tolist()
    return torch.cat(frames), torch.tensor(offs), torch.tensor(labels), torch.tensor(durs)


def batch(frames, offs, idx, device):
    lens = (offs[idx + 1] - offs[idx]).tolist()
    T = max(lens)
    x = torch.zeros(len(idx), T, frames.shape[1], dtype=torch.float32)
    m = torch.zeros(len(idx), T, dtype=torch.bool)
    for k, (i, n) in enumerate(zip(idx.tolist(), lens)):
        x[k, :n] = frames[offs[i] : offs[i] + n].float()
        m[k, :n] = True
    return x.to(device), m.to(device)


@torch.no_grad()
def evaluate(head, data, device, bs=512):
    frames, offs, labels, durs = data
    head.eval()
    pred = []
    for i in range(0, len(labels), bs):
        idx = torch.arange(i, min(i + bs, len(labels)))
        x, m = batch(frames, offs, idx, device)
        pred.append(head(x, m).argmax(-1).cpu())
    pred = torch.cat(pred)
    ok = pred == labels
    short = durs < 1.0
    return ok[short].float().mean().item(), ok.float().mean().item()


def main(feat_dir, out):
    device = "cuda"
    fd = Path(feat_dir)
    train = load(sorted(fd.glob("*_train_*.pt")))
    dev = load(sorted(fd.glob("*_dev_*.pt")))
    frames, offs, labels, durs = train
    print(f"train {len(labels)} crops, dev {len(dev[2])} crops", flush=True)
    keys = labels * 10 + torch.tensor([bucket(d) for d in durs.tolist()])
    counts = {k: (keys == k).sum().item() for k in keys.unique().tolist()}
    weights = torch.tensor([1.0 / counts[k] for k in keys.tolist()])
    torch.manual_seed(0)
    head = LidHead(len(LANGS)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    steps = max(1, len(labels) // 256)
    best, best_epoch, patience = -1.0, 0, 5
    log = []
    for epoch in range(1, 31):
        head.train()
        idx_all = torch.multinomial(weights, steps * 256, replacement=True)
        total = 0.0
        for s in range(steps):
            idx = idx_all[s * 256 : (s + 1) * 256]
            x, m = batch(frames, offs, idx, device)
            loss = loss_fn(head(x, m), labels[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        short_acc, all_acc = evaluate(head, dev, device)
        log.append(
            {"epoch": epoch, "loss": total / steps, "dev_short": short_acc, "dev_all": all_acc}
        )
        print(json.dumps(log[-1]), flush=True)
        if short_acc > best:
            best, best_epoch = short_acc, epoch
            head.save(
                out,
                LANGS,
                {
                    "encoder": "openai/whisper-small",
                    "epoch": epoch,
                    "dev_short": short_acc,
                    "dev_all": all_acc,
                    "train_crops": len(labels),
                },
            )
        elif epoch - best_epoch >= patience:
            break
    print(f"TRAIN_DONE best dev_short {best:.4f} at epoch {best_epoch} -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
