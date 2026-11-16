# Emotion-analysis
# Facial and Vocal Emotion in E-commerce Livestreams

This project examines how streamers’ facial and vocal emotional expressions relate to viewer engagement and sales in e-commerce livestreams. The Python script processes recordings in batches and is intended for use on UWA’s Kaya HPC cluster.

## What the script measures

- **Face.** FFmpeg samples video at 3 frames per second. OpenCV YuNet detects faces, and HSEmotion scores the largest eligible face in each sampled frame for expression, valence, and arousal.
- **Audio.** FFmpeg converts the audio to 16 kHz mono. A pretrained wav2vec2 model scores valence, arousal, and dominance in non-overlapping 5-second windows that meet the −40 dBFS threshold.
- **First appearance.** `first_face_time_s` is the time of the first sampled frame with a detected face. `first_audio_time_s` is the start of the first 0.1-second window at or above −40 dBFS. Both are in seconds relative to the start of the decoded stream. A missing detection is blank. Audible music or other sounds can trigger the audio measure.

If faces appear in fewer than 5% of sampled frames, the recording is flagged `no_face`. The summary retains any detected onset times, and emotion outputs are omitted for that recording.

## Setup

Use Python 3 and make sure both `ffmpeg` and `ffprobe` are on your PATH. In your Python environment, install:

```bash
python -m pip install numpy opencv-python onnxruntime hsemotion-onnx torch transformers
```

Download the [YuNet ONNX weights](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) and save them under `models/` for the example below. The emotion model weights must also be available to the Python environment when the script runs.

## Run a pilot

From the repository root, run this on a small set of recordings first. Replace the example paths with your approved UWA research storage paths.

```bash
python livestream_emotion.py recordings/ results/summary.csv \
  --yunet models/face_detection_yunet_2023mar.onnx \
  --timeseries-dir results/timeseries
```

Use a new summary CSV path for each run. For all available options, run `python livestream_emotion.py --help`.

## Outputs

| Output | Contents |
| --- | --- |
| `summary.csv` | One row per recording, including duration, stream availability, first detection times, coverage ratios, emotion means and standard deviations, and processing status. |
| `*_face.csv` | Optional per-frame timestamps, face detection flags, expression labels, valence, and arousal. |
| `*_audio.csv` | Optional per-window start times, sound levels, activity flags, valence, arousal, and dominance. |

Minute-level aggregation and matching to engagement or sales data are downstream steps. The face and audio measures are computed independently, so review pilot recordings to assess whether they represent the same presenter.

Keep recordings and generated CSVs in approved UWA research storage. Use this repository for code and documentation.



