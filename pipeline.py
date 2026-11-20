import json
import sys
import numpy as np
import onnxruntime as ort

import preprocess

EXPORT_DIR = "export"
_cache = {}

def load_model(name):
    if name not in _cache:
        with open(EXPORT_DIR + "/" + name + "/deployment_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        sess = ort.InferenceSession(EXPORT_DIR + "/" + name + "/model.onnx",
                                    providers=["CPUExecutionProvider"])
        _cache[name] = (sess, meta)
    return _cache[name]

def class_names():
    return load_model("x3d")[1]["class_names"]

def prepare_input(clip, meta):
    """clip: uint8 (16,160,160,3) -> float32 (1,3,16,160,160), dinormalisasi."""
    x = clip.astype(np.float32) / 255.0
    mean = np.array(meta["mean"], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(meta["std"], dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (3, 0, 1, 2))
    return x[None].astype(np.float32)

def calibrated_softmax(logits, temperature):
    z = logits / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def model_probs(name, clip):
    sess, meta = load_model(name)
    logits = sess.run(None, {"clip": prepare_input(clip, meta)})[0]
    return calibrated_softmax(logits, meta["temperature"])[0]

def predict(clip, model="fast"):
    """fast = X3D | accurate = Fused (X3D + R(2+1)D, bobot 0.5)."""
    if model == "fast":
        probs = model_probs("x3d", clip)
    elif model == "accurate":
        probs = 0.5 * model_probs("x3d", clip) + 0.5 * model_probs("r2plus1d", clip)
    else:
        raise ValueError("model harus 'fast' atau 'accurate'")
    order = np.argsort(-probs)
    names = class_names()
    return [(names[i], float(probs[i])) for i in order]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Pemakaian: python pipeline.py video.mp4 [fast|accurate]")
    video = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "fast"
    clip = preprocess.sample_frames(video)
    if clip is None:
        raise SystemExit("Video tidak bisa dibaca: " + video)
    result = predict(clip, model)
    print("Video :", video)
    print("Model :", model)
    print("")
    print("HASIL PREDIKSI:", result[0][0])
    print("")
    for rank, (name, p) in enumerate(result, 1):
        print(str(rank) + ". " + name.ljust(12) + str(round(p * 100, 1)) + " %")

