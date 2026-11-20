import os
import sys
import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pipeline

root = sys.argv[1] if len(sys.argv) > 1 else "tes_video"
model = sys.argv[2] if len(sys.argv) > 2 else "fast"

names = pipeline.class_names()
lower = {n.lower(): n for n in names}
exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")

items = []
for label in sorted(os.listdir(root)):
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        continue
    if label.lower() not in lower:
        print("LEWATI folder (bukan nama kelas):", label)
        continue
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith(exts):
            items.append((p, lower[label.lower()]))

print("Jumlah klip:", len(items), "| model:", model)
print("")

modes = pipeline.RESIZE_MODES
correct = {m: 0 for m in modes}
agree = 0
total = 0

for path, label in items:
    try:
        frames = pipeline.read_frames(path)
    except Exception as e:
        print("GAGAL BACA", path, e)
        continue
    preds = {}
    for m in modes:
        clip = pipeline.make_clip(frames, m)
        result = pipeline.predict(clip, model)
        preds[m] = result[0][0]
        if result[0][0] == label:
            correct[m] += 1
    total += 1
    if len(set(preds.values())) == 1:
        agree += 1
    line = os.path.basename(path).ljust(24) + "asli=" + label.ljust(12)
    for m in modes:
        mark = "OK" if preds[m] == label else "SALAH"
        line += " | " + m + "=" + preds[m] + " (" + mark + ")"
    print(line)

print("")
print("===== RINGKASAN =====")
if total == 0:
    raise SystemExit("Tidak ada klip yang berhasil diproses.")
for m in modes:
    print(m.ljust(8), "benar", correct[m], "dari", total, "=", round(100 * correct[m] / total, 1), "%")
print("Ketiga cara resize memberi tebakan sama pada", agree, "dari", total, "klip")
