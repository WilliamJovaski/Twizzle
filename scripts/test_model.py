import json, time
import numpy as np
import onnxruntime as ort

for name in ["x3d", "r2plus1d"]:
    print("")
    print("===== " + name + " =====")
    with open("export/" + name + "/deployment_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    print("Isi meta:", list(meta.keys()))

    sess = ort.InferenceSession("export/" + name + "/model.onnx",
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print("Input :", inp.name, inp.shape)
    print("Output:", out.name, out.shape)

    x = np.random.rand(1, 3, 16, 160, 160).astype(np.float32)
    sess.run(None, {inp.name: x})
    t = time.perf_counter()
    logits = sess.run(None, {inp.name: x})[0]
    print("Shape hasil:", logits.shape)
    print("Waktu (ms):", round((time.perf_counter() - t) * 1000))
