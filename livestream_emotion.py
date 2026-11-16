"""Face/audio presence and emotion (valence, arousal, dominance) inventory for livestream recordings.

For every recording in a folder:
  1. ffprobe: duration and which streams exist.
  2. Record the first detected face and first audible 100 ms window.
     Times are in seconds using the decoded frame and audio window indices.
     Audible windows can contain music or other non-speech sounds.
     Sample video frames (default 3 fps) and detect faces with OpenCV YuNet.
     If a face is found in fewer than --min-face-ratio (default 5%) of the sampled
     frames, the recording is marked status="no_face" and emotion results are omitted.
  3. Otherwise:
       Face:  valence + arousal (and 8-class expression) per frame with HSEmotion
              (enet_b0_8_va_mtl) on the largest face. Scale: about -1 .. 1.
       Audio: arousal + dominance + valence per window with audeering's
              wav2vec2-large-robust-12-ft-emotion-msp-dim. Scale: about 0 .. 1.
              Windows quieter than --silence-db are skipped.
     Means and SDs go into the summary CSV; optional per-frame / per-window CSVs
     go into --timeseries-dir.

Requirements
  FFmpeg (ffmpeg + ffprobe) on PATH
  pip install numpy opencv-python onnxruntime hsemotion-onnx torch transformers
  YuNet face detector weights (pass with --yunet):
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

Example
  python livestream_emotion.py recordings/ results/summary.csv \
      --yunet models/face_detection_yunet_2023mar.onnx --timeseries-dir results/timeseries
"""

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".flv", ".ts"}
AUDIO_SR = 16000
AUDIO_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

COLUMNS = [
    "file", "duration_seconds", "has_video", "has_audio",
    "first_face_time_s", "first_audio_time_s",
    "frames_sampled", "face_frames", "face_presence_ratio",
    "face_valence_mean", "face_valence_sd",
    "face_arousal_mean", "face_arousal_sd",
    "face_dominant_expression",
    "audio_windows", "audio_active_ratio",
    "audio_valence_mean", "audio_valence_sd",
    "audio_arousal_mean", "audio_arousal_sd",
    "audio_dominance_mean", "audio_dominance_sd",
    "status", "notes",
]


# --------------------------------------------------------------------------- probing

def probe(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=60,
    )
    return json.loads(result.stdout)


# --------------------------------------------------------------------------- decoding

def _stream_ffmpeg(cmd, chunk_bytes, min_bytes):
    """Yield fixed-size chunks of ffmpeg stdout; kill ffmpeg if the consumer stops early."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            buf = proc.stdout.read(chunk_bytes)
            if len(buf) < min_bytes:
                break
            yield buf
    finally:
        proc.kill()
        proc.wait()


def iter_frames(path, fps, box):
    """Yield (time_s, BGR frame) letterboxed into a box x box square (works for portrait streams)."""
    vf = (
        f"fps={fps},scale={box}:{box}:force_original_aspect_ratio=decrease,"
        f"pad={box}:{box}:(ow-iw)/2:(oh-ih)/2"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    size = box * box * 3
    for i, buf in enumerate(_stream_ffmpeg(cmd, size, size)):
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(box, box, 3)


def iter_audio_windows(path, window_s, min_window_s=1.0):
    """Yield (start_s, mono float32 16 kHz samples). Trailing windows shorter than min_window_s are dropped."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "pipe:1",
    ]
    size = int(window_s * AUDIO_SR) * 4
    for i, buf in enumerate(_stream_ffmpeg(cmd, size, int(min_window_s * AUDIO_SR) * 4)):
        yield i * window_s, np.frombuffer(buf, np.float32)


# --------------------------------------------------------------------------- face

class FaceAnalyzer:
    def __init__(self, yunet_path, box, score_threshold, min_face_px):
        import cv2
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (box, box), score_threshold, 0.3, 5000
        )
        self.fer = HSEmotionRecognizer(model_name="enet_b0_8_va_mtl")
        self.min_face_px = min_face_px

    def analyze(self, frame):
        """Return (face_found, expression, valence, arousal); emotion fields are None
        when no face is found or the largest face is too small to score reliably."""
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return False, None, None, None

        x, y, w, h = faces[np.argmax(faces[:, 2] * faces[:, 3]), :4]
        if min(w, h) < self.min_face_px:
            return True, None, None, None

        # Crop with a 15% margin, clipped to the frame.
        mx, my = 0.15 * w, 0.15 * h
        H, W = frame.shape[:2]
        x0, y0 = max(int(x - mx), 0), max(int(y - my), 0)
        x1, y1 = min(int(x + w + mx), W), min(int(y + h + my), H)
        if x1 <= x0 or y1 <= y0:
            return True, None, None, None

        rgb = self.cv2.cvtColor(frame[y0:y1, x0:x1], self.cv2.COLOR_BGR2RGB)
        expression, scores = self.fer.predict_emotions(rgb, logits=False)
        # enet_b0_8_va_mtl: 8 expression scores, then valence, then arousal.
        return True, expression, float(scores[-2]), float(scores[-1])


# --------------------------------------------------------------------------- audio

class AudioAnalyzer:
    def __init__(self, device):
        import torch
        from torch import nn
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, x):
                x = torch.tanh(self.dense(self.dropout(x)))
                return self.out_proj(self.dropout(x))

        class EmotionModel(Wav2Vec2PreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values)[0].mean(dim=1)
                return self.classifier(hidden)

        self.torch = torch
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL)
        self.model = EmotionModel.from_pretrained(AUDIO_MODEL).to(device).eval()

    def predict(self, samples):
        """Return (arousal, dominance, valence) for one window."""
        x = self.processor(samples, sampling_rate=AUDIO_SR)["input_values"][0]
        x = self.torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            out = self.model(x)[0].cpu().numpy()
        return float(out[0]), float(out[1]), float(out[2])


def rms_dbfs(samples):
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


# --------------------------------------------------------------------------- helpers

def mean_sd(values):
    if not values:
        return "", ""
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return round(float(arr.mean()), 4), round(sd, 4)


def write_timeseries(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# --------------------------------------------------------------------------- per recording

def analyze_recording(path, input_dir, args, face_analyzer, get_audio_analyzer):
    rel = path.relative_to(input_dir).as_posix()
    row = {"file": rel}

    try:
        metadata = probe(path)
    except subprocess.CalledProcessError:
        return {**row, "status": "probe_failed"}
    except subprocess.TimeoutExpired:
        return {**row, "status": "probe_timed_out"}
    except json.JSONDecodeError:
        return {**row, "status": "invalid_metadata"}

    stream_types = {s.get("codec_type") for s in metadata.get("streams", [])}
    has_video, has_audio = "video" in stream_types, "audio" in stream_types
    row.update(
        duration_seconds=metadata.get("format", {}).get("duration", ""),
        has_video=has_video,
        has_audio=has_audio,
    )
    if has_audio:
        windows = iter_audio_windows(path, 0.1, min_window_s=0.1)
        try:
            for start, samples in windows:
                if rms_dbfs(samples) >= args.silence_db:
                    row["first_audio_time_s"] = round(start, 3)
                    break
        finally:
            windows.close()

    if not has_video:
        return {**row, "status": "no_video"}

    # ---- face presence + face emotion (single decoding pass)
    frames = face_frames = 0
    valences, arousals, expressions, face_ts = [], [], Counter(), []
    for t, frame in iter_frames(path, args.fps, args.frame_size):
        frames += 1
        found, expr, val, aro = face_analyzer.analyze(frame)
        face_frames += found
        if found and "first_face_time_s" not in row:
            row["first_face_time_s"] = round(t, 3)
        if val is not None:
            valences.append(val)
            arousals.append(aro)
            expressions[expr] += 1
        face_ts.append([round(t, 3), int(found), expr or "", val if val is not None else "",
                        aro if aro is not None else ""])

    if frames == 0:
        return {**row, "status": "no_frames_decoded"}

    ratio = face_frames / frames
    row.update(frames_sampled=frames, face_frames=face_frames,
               face_presence_ratio=round(ratio, 4))

    if ratio < args.min_face_ratio:
        return {**row, "status": "no_face",
                "notes": f"face in {ratio:.1%} of frames (< {args.min_face_ratio:.0%})"}

    notes = []
    row["face_valence_mean"], row["face_valence_sd"] = mean_sd(valences)
    row["face_arousal_mean"], row["face_arousal_sd"] = mean_sd(arousals)
    row["face_dominant_expression"] = expressions.most_common(1)[0][0] if expressions else ""
    if not valences:
        notes.append("faces too small for emotion scoring")

    stem = rel.replace("/", "__").rsplit(".", 1)[0]
    if args.timeseries_dir:
        write_timeseries(args.timeseries_dir / f"{stem}_face.csv",
                         ["time_s", "face_found", "expression", "valence", "arousal"], face_ts)

    # ---- audio emotion
    if not has_audio:
        return {**row, "status": "ok_no_audio", "notes": "; ".join(notes)}

    audio = get_audio_analyzer()
    windows = 0
    a_val, a_aro, a_dom, audio_ts = [], [], [], []
    for start, samples in iter_audio_windows(path, args.audio_window):
        windows += 1
        level = rms_dbfs(samples)
        if level < args.silence_db:
            audio_ts.append([round(start, 3), round(level, 1), 0, "", "", ""])
            continue
        aro, dom, val = audio.predict(samples)
        a_val.append(val)
        a_aro.append(aro)
        a_dom.append(dom)
        audio_ts.append([round(start, 3), round(level, 1), 1, val, aro, dom])

    row["audio_windows"] = windows
    row["audio_active_ratio"] = round(len(a_val) / windows, 4) if windows else 0
    row["audio_valence_mean"], row["audio_valence_sd"] = mean_sd(a_val)
    row["audio_arousal_mean"], row["audio_arousal_sd"] = mean_sd(a_aro)
    row["audio_dominance_mean"], row["audio_dominance_sd"] = mean_sd(a_dom)

    if args.timeseries_dir:
        write_timeseries(args.timeseries_dir / f"{stem}_audio.csv",
                         ["start_s", "rms_dbfs", "active", "valence", "arousal", "dominance"],
                         audio_ts)

    status = "ok" if a_val else "ok_silent_audio"
    return {**row, "status": status, "notes": "; ".join(notes)}


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=Path, help="Folder of recordings")
    parser.add_argument("output_csv", type=Path, help="Path for the summary CSV")
    parser.add_argument("--yunet", type=Path, required=True,
                        help="Path to face_detection_yunet_2023mar.onnx")
    parser.add_argument("--timeseries-dir", type=Path,
                        help="Also write per-frame / per-window CSVs here")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Frames sampled per second (default 3)")
    parser.add_argument("--frame-size", type=int, default=640,
                        help="Frames are letterboxed into this square size (default 640)")
    parser.add_argument("--min-face-ratio", type=float, default=0.05,
                        help="Below this share of frames with a face -> no_face (default 0.05)")
    parser.add_argument("--face-score", type=float, default=0.7,
                        help="YuNet detection confidence threshold (default 0.7)")
    parser.add_argument("--min-face-px", type=int, default=32,
                        help="Smallest face side (px, in the letterboxed frame) used for emotion")
    parser.add_argument("--audio-window", type=float, default=5.0,
                        help="Audio window length in seconds (default 5)")
    parser.add_argument("--silence-db", type=float, default=-40.0,
                        help="Windows below this RMS level (dBFS) count as silent (default -40)")
    parser.add_argument("--device", default=None,
                        help="torch device for the audio model (default: cuda if available)")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input folder does not exist: {input_dir}")
    for tool in ("ffprobe", "ffmpeg"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is unavailable. Load or install FFmpeg first.")
    if args.output_csv.suffix.lower() != ".csv":
        parser.error("The output path must end in .csv.")
    if not args.yunet.is_file():
        parser.error(f"YuNet model not found: {args.yunet}")
    if not 0 <= args.min_face_ratio <= 1:
        parser.error("--min-face-ratio must be between 0 and 1.")

    videos = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        parser.error(f"No supported video files found in {input_dir}")

    face_analyzer = FaceAnalyzer(args.yunet, args.frame_size, args.face_score, args.min_face_px)

    audio_holder = {}

    def get_audio_analyzer():  # loaded only when a recording actually needs it
        if "model" not in audio_holder:
            import torch
            device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            audio_holder["model"] = AudioAnalyzer(device)
        return audio_holder["model"]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output_csv.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video.relative_to(input_dir)}", flush=True)
            try:
                row = analyze_recording(video, input_dir, args, face_analyzer, get_audio_analyzer)
            except Exception as exc:  # keep the batch going; record what broke
                row = {"file": video.relative_to(input_dir).as_posix(),
                       "status": "analysis_failed", "notes": repr(exc)[:300]}
            writer.writerow(row)
            output.flush()
            counts[row["status"]] += 1

    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"Checked {len(videos)} recordings ({summary}). Summary: {args.output_csv}")


if __name__ == "__main__":
    main()