import time
import numpy as np
import onnxruntime as ort

for name in ["x3d", "r2plus1d"]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession("export/" + name + "/model.onnx", so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 16, 160, 160).astype(np.float32)
    sess.run(None, {inp: x})
    sess.run(None, {inp: x})
    times = []
    for i in range(10):
        t = time.perf_counter()
        sess.run(None, {inp: x})
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    print(name, "| median ms:", round(times[5]), "| min:", round(times[0]), "| max:", round(times[-1]))
