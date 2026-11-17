# Gesture Slide Clicker

Control presentation slides with hand gestures in front of your webcam.

| Gesture | Action |
|---|---|
| point right (index finger or flat hand) | next slide |
| point left (index finger or flat hand) | previous slide |
| open palm (fingers up) | pause / resume |
| anything else | nothing |

## How it works

```
webcam (OpenCV, mirrored)
  -> 21 hand landmarks (MediaPipe HandLandmarker)
  -> features: relative to wrist, scaled by palm length, + finger extension & direction
  -> kNN classifier (scikit-learn)
  -> smoothing: confidence >= 0.8 for 15 frames in a row, 1 s cooldown, once per hold
  -> arrow key (pyautogui) or Keynote command (AppleScript)
```

Only landmark coordinates are stored, never images or video.

## Setup (Python 3.11, macOS or Windows)

**macOS**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows** (PowerShell, 64-bit Python 3.11 from python.org)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Then, on both:**
```bash
pip install -r requirements.txt
python setup_model.py        # downloads the MediaPipe hand model into models/
python src/train.py          # ~5 min, creates models/gesture_model.joblib
```

Don't install `opencv-python` alongside `opencv-contrib-python` (installed by
MediaPipe): they overwrite each other's `cv2` folder.

**Permissions**
- macOS (System Settings → Privacy & Security), for the app you run Python from
  (Terminal / VS Code), then restart it:
  - **Camera**
  - **Accessibility**: for key presses; without it the slides silently don't move
  - **Automation → Keynote**: asked once when using `--keynote`
- Windows: Settings → Privacy & security → **Camera** → allow desktop apps.
  Key presses need no extra permission. They don't reach apps running as
  administrator unless Python runs as administrator too.

## Usage

```bash
python src/landmarks.py          # camera demo with hand landmarks
python src/live.py --dry-run     # prints actions, presses nothing
python src/live.py               # presses arrow keys: click on your slides after starting
python src/live.py --keynote     # macOS only: controls Keynote directly (no focus needed)
```
Arrow keys work with PowerPoint, Keynote, Google Slides, PDF viewers and so on.
If the wrong camera opens, add `--camera 1`.

Tuning: `--threshold` (default 0.8), `--frames` (default 15 ≈ 0.5 s),
`--cooldown` (default 1.0 s), `--scale` (preview size). If it fires while you
talk, try `--threshold 0.9 --frames 20`; if it misses gestures, try `--threshold 0.7`.

## Record your own data

```bash
python src/record.py --person name --gesture point_right --hand right --distance near --session 1 --note index
python src/record.py --summary
```
- `--gesture`: `point_right`, `point_left`, `open_palm`, `none`
- `--hand`: `left` / `right`; `--distance`: `near` / `mid` / `far`
- SPACE starts a take (3 s countdown + 15 s recording), `u` undoes the last take, `q` quits

Rows are appended to `data/gestures.csv`: tags + 63 coordinates per frame.
Retrain with `python src/train.py` afterwards.

## Data

- **Own recordings:** 3 people, 69 takes, ~29k frames (`data/gestures.csv`).
- **[HaGRIDv2](https://github.com/hukenovs/hagrid)** landmark annotations, 19.7k
  people (`data/hagrid.csv`): `palm`/`stop` → open palm, `no_gesture` + other gestures → none.
  It also has synthetic sideways points made by rotating upright "index up" and
  flat-palm hands by ~90°. HaGRID's own `point` class points at the camera, so it's
  used as none.

Not needed to run the project, since `hagrid.csv` is included. To rebuild it
(719 MB download, annotations only; macOS/Linux shell):
```bash
mkdir -p data/hagrid_raw
curl -L -C - -o data/hagrid_raw/annotations.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
python src/import_hagrid.py
```

## Model

`python src/train.py` compares kNN, Random Forest and MLP with leave-one-person-out
over our 3 people and saves the best one. Current model: **kNN (k=15), own data +
HaGRID**, about 90% accuracy on a person it hasn't seen (per frame, before smoothing).
Left vs right is almost never confused; most errors are normal hand movements
read as a point. The smoothing filters those out.

`python src/evaluate.py` saves a confusion matrix to `models/confusion_matrix.png`.
Full numbers: `models/results.md`, `models/evaluation.md`.

## Live testing

To measure it with new people, an operator marks events on the camera window
(`f` = false trigger, `m` = missed gesture, `q` = save):
```bash
python src/live.py --test name --kind gestures --dry-run       # called-out gestures
python src/live.py --test name --kind talking --dry-run        # just talking, no gestures
python src/live.py --test name --kind presentation --keynote   # real talk
python src/live.py --summary
```
Results go to `results/`. The marking keys only work while the camera window
has focus, so a real-presentation test with marking needs `--keynote` (macOS).
Elsewhere, use the `gestures` and `talking` tests.

## Limitations

- Trained on only 3 people recorded in one room, so other people, lighting or
  cameras may work worse.
- Talking with your hands can still trigger a slide change occasionally.
- Developed on macOS. Windows is supported, but we haven't tested it on a
  Windows machine yet. `--keynote` is macOS only.

## Project structure

```
setup_model.py      downloads the MediaPipe hand model
src/
  landmarks.py      camera (mirrored) + MediaPipe wrapper
  record.py         data recording tool
  import_hagrid.py  HaGRID annotations -> data/hagrid.csv
  features.py       landmark normalization + extra features
  classifier.py     predict(landmarks) -> (gesture, confidence)
  train.py          model comparison + training
  evaluate.py       confusion matrix
  live.py           real-time clicker (+ live test mode)
  live_test.py      live test logging
data/               landmark CSVs
models/             trained model + results
```

## Credits

- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [HaGRIDv2](https://github.com/hukenovs/hagrid) ([paper](https://arxiv.org/abs/2412.01508)), license: variant of CC BY-SA 4.0
