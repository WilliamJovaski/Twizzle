import base64
import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool

import pipeline
import preprocess

MAX_SECONDS = 5.5   # batas 5 detik + toleransi kecil
MAX_UPLOAD_MB = 50
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLE_VIDEO_PATH = BASE_DIR / "contoh.mp4"

# Muat model sekali saat aplikasi start, supaya request pertama tidak lambat.
pipeline.load_model("x3d")
pipeline.load_model("r2plus1d")

MODE_MAP = {
    "Fast (X3D, cepat)": "fast",
    "Accurate (Fused X3D + R(2+1)D, lebih akurat, lebih lambat)": "accurate",
}
DEFAULT_MODE = "Fast (X3D, cepat)"

def classify(video_path, mode_label):
    if video_path is None:
        raise gr.Error("Upload video dulu.")

    info = preprocess.probe_video(video_path)
    if info is None:
        raise gr.Error("Video tidak bisa dibuka. Coba pakai format mp4.")

    dur = info["duration_s"]
    if dur == dur and dur > MAX_SECONDS:   # dur == dur bernilai False kalau dur = NaN
        raise gr.Error("Video terlalu panjang (" + str(round(dur, 1)) + " detik). Maksimal 5 detik, satu elemen saja.")

    clip = preprocess.sample_frames(video_path)
    if clip is None:
        raise gr.Error("Frame video tidak bisa dibaca.")

    mode = MODE_MAP[mode_label]
    t0 = time.perf_counter()
    result = pipeline.predict(clip, mode)
    elapsed = time.perf_counter() - t0

    top_name, top_p = result[0]
    summary = "## Hasil prediksi: " + top_name + " (" + str(round(top_p * 100, 1)) + "%)"

    label_dict = {name: p for name, p in result}
    table = [[i + 1, name, str(round(p * 100, 1)) + " %"] for i, (name, p) in enumerate(result)]
    frames = [clip[i] for i in range(clip.shape[0])]
    info_text = "Waktu inferensi: " + str(round(elapsed, 2)) + " detik | Durasi video: " + str(round(dur, 1)) + " detik"

    return summary, label_dict, table, frames, info_text


def _encode_frames_b64(clip):
    encoded = []
    for i in range(clip.shape[0]):
        rgb = clip[i]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
    return encoded


def _run_video_classification(video_path: str, mode: str):
    if mode not in ("fast", "accurate"):
        raise HTTPException(status_code=400, detail="Mode must be 'fast' or 'accurate'.")

    info = preprocess.probe_video(video_path)
    if info is None:
        raise HTTPException(status_code=400, detail="Unable to open video. Please use a valid MP4 file.")

    dur = info["duration_s"]
    if dur == dur and dur > MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Video is too long ({round(dur, 1)}s). Maximum allowed is 5 seconds containing a single element.",
        )

    clip = preprocess.sample_frames(video_path)
    if clip is None:
        raise HTTPException(status_code=400, detail="Unable to decode video frames.")

    t0 = time.perf_counter()
    result = pipeline.predict(clip, mode)
    elapsed = time.perf_counter() - t0

    predictions = [
        {"rank": i + 1, "name": name, "prob": p, "pct": round(p * 100.0, 2)}
        for i, (name, p) in enumerate(result)
    ]

    return {
        "predictions": predictions,
        "elapsed_s": round(elapsed, 2),
        "duration_s": round(dur, 2) if dur == dur else 0.0,
        "frames_b64": _encode_frames_b64(clip),
    }


# ---------------------------------------------------------------------------
# FastAPI + Gradio App
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Twizzle AI Skating Analytics")
web_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@web_app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@web_app.get("/api/sample-video")
async def get_sample_video():
    if not SAMPLE_VIDEO_PATH.is_file():
        raise HTTPException(status_code=404, detail="Sample video (contoh.mp4) was not found.")
    return FileResponse(str(SAMPLE_VIDEO_PATH), media_type="video/mp4")


@web_app.post("/api/classify")
async def api_classify(
    mode: str = Form("fast"),
    use_sample: str = Form("false"),
    video: UploadFile | None = File(None),
):
    if use_sample.lower() == "true":
        if not SAMPLE_VIDEO_PATH.is_file():
            raise HTTPException(status_code=404, detail="Sample video (contoh.mp4) was not found.")
        return await run_in_threadpool(_run_video_classification, str(SAMPLE_VIDEO_PATH), mode)

    if video is None:
        raise HTTPException(status_code=400, detail="Please upload a video file first.")

    suffix = Path(video.filename or "clip.mp4").suffix or ".mp4"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)
    try:
        limit = MAX_UPLOAD_MB * 1024 * 1024
        content = await video.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail="File terlalu besar (maksimal " + str(MAX_UPLOAD_MB) + " MB).")
        with open(tmp_path, "wb") as f:
            f.write(content)
        return await run_in_threadpool(_run_video_classification, tmp_path, mode)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass



with gr.Blocks(title="Twizzle") as demo:
    gr.Markdown("# Twizzle: Klasifikasi Elemen Figure Skating")
    gr.Markdown("Upload klip pendek (maksimal 5 detik) yang berisi satu elemen: satu lompatan, satu putaran, atau satu sequence.")
    with gr.Row():
        with gr.Column():
            video = gr.Video(label="Video klip (maks. 5 detik)")
            mode = gr.Radio(list(MODE_MAP.keys()), value=DEFAULT_MODE, label="Mode")
            btn = gr.Button("Klasifikasi", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            top3 = gr.Label(num_top_classes=3, label="3 tebakan teratas")
            with gr.Accordion("Lihat semua 11 kelas", open=False):
                table = gr.Dataframe(headers=["Peringkat", "Gerakan", "Persentase"], interactive=False)
            info_box = gr.Markdown()
    gallery = gr.Gallery(label="16 frame yang dianalisis model", columns=8, height="auto")

    btn.click(classify, inputs=[video, mode], outputs=[summary, top3, table, gallery, info_box], api_name="classify")


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        # Berjalan di Hugging Face Space: pakai launcher Gradio bawaan
        demo.launch()
    else:
        # Berjalan di laptop: UI custom + Gradio di /gradio
        web_app = gr.mount_gradio_app(web_app, demo, path="/gradio")
        port = int(os.environ.get("PORT", 7860))
        print("Twizzle berjalan di: http://127.0.0.1:" + str(port))
        print("Classic Gradio UI di: http://127.0.0.1:" + str(port) + "/gradio")
        uvicorn.run(web_app, host="0.0.0.0", port=port)
