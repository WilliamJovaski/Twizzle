import numpy as np

def load(name, split):
    d = np.load("predictions/" + name + "_" + split + ".npz", allow_pickle=True)
    return d["probs_calibrated"], d["labels"], d["clip_ids"]

def stats(probs, labels, m):
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    correct = probs.argmax(1) == labels
    conf = margin >= m
    acc = correct[conf].mean() if conf.any() else float("nan")
    return conf.mean(), acc

data = {}
for split in ["val", "test"]:
    px, lx, ix = load("x3d", split)
    pr, lr, ir = load("r2plus1d", split)
    same = bool((ix == ir).all()) and bool((lx == lr).all())
    print(split, "| urutan klip kedua model sama:", same)
    if not same:
        raise SystemExit("STOP: urutan klip beda, kirim hasil ini ke Kiro")
    data[split] = (px, pr, lx)

print("")
print("=== Tabel margin Fused (VALIDATION) ===")
px, pr, y = data["val"]
pf = 0.5 * px + 0.5 * pr
chosen = None
for m in np.round(np.arange(0.0, 0.96, 0.05), 2):
    cov, acc = stats(pf, y, m)
    print("margin", m, "| yakin:", round(cov * 100, 1), "% | benar saat yakin:", round(acc * 100, 1), "%")
    if chosen is None and acc >= 0.95:
        chosen = float(m)
print("")
print(">>> MARGIN FUSED TERPILIH:", chosen)

print("")
print("=== Cek di TEST (data yang belum pernah dilihat) ===")
px, pr, y = data["test"]
for name, p, m in [("x3d", px, 0.40), ("r2plus1d", pr, 0.15), ("fused", 0.5 * px + 0.5 * pr, chosen)]:
    cov, acc = stats(p, y, m)
    print(name, "| margin", m, "| yakin:", round(cov * 100, 1), "% | ragu:", round((1 - cov) * 100, 1), "% | benar saat yakin:", round(acc * 100, 1), "%")
