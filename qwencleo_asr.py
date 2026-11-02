#!/usr/bin/env python
"""
qwencleo_asr.py
Use QwenCleo-ASR (Qwen3-ASR-1.7B fine-tuned for Egyptian Arabic) instead of Whisper.

Three commands
--------------
1. eval        Zero-shot WER/CER on your dataset's validation/test split, using the
               SAME subset selection and Arabic normalization as train_asr.py, so
               the numbers are directly comparable with your Whisper results.

    python qwencleo_asr.py eval --max-eval 2000
    python qwencleo_asr.py eval --max-eval 2000 --split test
    python qwencleo_asr.py eval --max-eval 2000 ^
        --whisper-preds .\\whisper-medium-egyptian-asr-20k\\predictions_eval_only\\validation_predictions.jsonl

2. transcribe  Files / folders / globs (optionally scored against .txt references
               or a --manifest CSV/JSONL with columns: audio,text)

    python qwencleo_asr.py transcribe clip.wav
    python qwencleo_asr.py transcribe my_test_set\\ --output results.jsonl

3. mic         Record from the microphone, transcribe, optionally collect references

    python qwencleo_asr.py mic --ask-reference

Install (use a SEPARATE virtual environment: qwen-asr pins its own
transformers version and may break your Whisper environment):
    pip install -U qwen-asr datasets jiwer soundfile tqdm librosa
    pip install sounddevice          # only for the mic command
    pip install librosa              # only for mp3/m4a input (needs ffmpeg)
"""

import argparse
import csv
import glob
import json
import os
import queue
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.5
MAX_AUDIO_SECONDS = 30.0
MIN_CHUNK_SECONDS = 0.2
SILENCE_PEAK = 1e-3            # chunks quieter than this are skipped (avoids hallucinations)
DEFAULT_MODEL = "mohammedaly22/QwenCleo-ASR"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac",
              ".wma", ".opus", ".webm", ".mp4"}
TEXT_COLUMN_CANDIDATES = [
    "text", "sentence", "transcription", "transcript", "normalized_text",
]


# --------------------------------------------------
# Arabic normalization (identical to train_asr.py)
# --------------------------------------------------

def normalize_ar(text: str) -> str:
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)  # diacritics, tatweel
    text = re.sub("[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s]", " ", text)                      # punctuation
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------
# Audio helpers
# --------------------------------------------------

def get_audio(audio):
    """(mono float32 array, sampling rate) from a dict or a decoder object."""
    if isinstance(audio, dict) and "array" in audio:
        return np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        arr = samples.data.numpy()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        return arr.astype(np.float32), samples.sample_rate

    raise TypeError(f"Unsupported audio format: {type(audio)}")


def decode_audio_field(a):
    """Decode a datasets Audio(decode=False) value with soundfile.

    Avoids torchcodec/FFmpeg entirely (they are fragile on Windows).
    Returns a mono float32 array at 16 kHz.
    """
    import io
    import soundfile as sf

    if isinstance(a, dict) and (a.get("bytes") or a.get("path")):
        src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
        arr, sr = sf.read(src, dtype="float32", always_2d=True)
        return resample(arr.mean(axis=1), sr, SAMPLE_RATE)

    arr, sr = get_audio(a)          # already-decoded formats
    return resample(arr, sr, SAMPLE_RATE)


def resample(arr: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if sr == target:
        return arr

    try:
        import torch
        import torchaudio.functional as AF
        return AF.resample(torch.from_numpy(arr), sr, target).numpy()
    except Exception:
        pass

    try:
        import librosa
        return librosa.resample(arr, orig_sr=sr, target_sr=target).astype(np.float32)
    except Exception:
        pass

    print("WARNING: torchaudio/librosa not found; using basic linear resampling "
          "(pip install librosa for better quality).")
    n = max(1, int(round(len(arr) * target / sr)))
    return np.interp(np.linspace(0, len(arr) - 1, n),
                     np.arange(len(arr)), arr).astype(np.float32)


def load_audio(path) -> np.ndarray:
    """Read any supported audio file as mono float32 at 16 kHz."""
    path = str(path)
    arr, sr = None, None

    try:
        import soundfile as sf
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        arr = arr.mean(axis=1)
    except Exception as sf_error:
        try:
            import librosa
            arr, sr = librosa.load(path, sr=None, mono=True)
        except Exception as lb_error:
            raise RuntimeError(
                f"Could not read '{path}'. Use .wav/.flac, or install "
                f"librosa + ffmpeg for mp3/m4a. "
                f"(soundfile: {sf_error}; librosa: {lb_error})"
            )
    return resample(arr, sr, SAMPLE_RATE)


def split_audio(arr: np.ndarray, max_s: float, search_s: float = 4.0,
                frame_ms: int = 20):
    """Split long audio into <= max_s chunks, cutting at the quietest point
    near each boundary so words are not chopped in half."""
    max_len = int(max_s * SAMPLE_RATE)
    if len(arr) <= max_len:
        return [arr]

    frame = int(SAMPLE_RATE * frame_ms / 1000)
    chunks, start = [], 0
    while start < len(arr):
        end = min(start + max_len, len(arr))
        if end < len(arr):
            lo = max(start + max_len - int(search_s * SAMPLE_RATE),
                     start + SAMPLE_RATE)
            seg = arr[lo:end]
            n = len(seg) // frame
            if n > 0:
                frames = seg[: n * frame].reshape(n, frame)
                energy = np.sqrt((frames ** 2).mean(axis=1))
                if n >= 5:
                    energy = np.convolve(energy, np.ones(5) / 5, mode="same")
                end = lo + int(np.argmin(energy)) * frame + frame // 2
        chunks.append(arr[start:end])
        start = end

    min_len = int(MIN_CHUNK_SECONDS * SAMPLE_RATE)
    return [c for c in chunks if len(c) >= min_len]


# --------------------------------------------------
# QwenCleo-ASR wrapper
# --------------------------------------------------

class QwenCleoTranscriber:
    def __init__(self, model_id=DEFAULT_MODEL, device="auto", language="Arabic",
                 batch_size=8, max_new_tokens=256, chunk_seconds=30.0):
        import torch
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise SystemExit("Missing package. Install it with:  pip install -U qwen-asr")
        try:
            import soundfile  # noqa: F401
        except ImportError:
            raise SystemExit("Missing package. Install it with:  pip install soundfile")

        self.torch = torch
        self.language = language if language and language.lower() != "auto" else None
        self.batch_size = batch_size
        self.chunk_seconds = chunk_seconds

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        print(f"Loading {model_id} on {device} ({str(dtype).replace('torch.', '')})")
        kwargs = dict(dtype=dtype, device_map=device)
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_id, max_new_tokens=max_new_tokens, **kwargs)
        except TypeError:
            # Older qwen-asr versions do not accept max_new_tokens here.
            self.model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)

    def _run(self, wav_paths):
        results = self.model.transcribe(audio=wav_paths, language=self.language)
        return [r.text for r in results]

    def transcribe_arrays(self, arrays):
        """List of 16 kHz mono float32 arrays -> list of strings (same order)."""
        import soundfile as sf

        out = [""] * len(arrays)
        usable = [(i, a) for i, a in enumerate(arrays)
                  if a.size and float(np.abs(a).max()) > SILENCE_PEAK]

        with tempfile.TemporaryDirectory() as tmp:
            for s in range(0, len(usable), self.batch_size):
                part = usable[s:s + self.batch_size]
                paths = []
                for i, a in part:
                    p = os.path.join(tmp, f"{i}.wav")
                    sf.write(p, a, SAMPLE_RATE)
                    paths.append(p)
                for (i, _), text in zip(part, self._run(paths)):
                    out[i] = (text or "").strip()
        return out

    def warmup(self):
        """One dummy pass so the first real timing is not inflated by CUDA setup."""
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        self.transcribe_arrays([0.1 * np.sin(2 * np.pi * 220 * t)])

    def transcribe(self, audio: np.ndarray):
        """One (possibly long) recording -> (text, seconds_taken)."""
        start = time.perf_counter()
        chunks = split_audio(audio, self.chunk_seconds)
        texts = self.transcribe_arrays(chunks)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return " ".join(t for t in texts if t), elapsed


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_metrics(refs, hyps):
    """WER/CER on raw and Arabic-normalized text. Returns {} if no references."""
    import jiwer

    pairs = [(r, h) for r, h in zip(refs, hyps) if r and r.strip()]
    if not pairs:
        return {}

    out = {"n": len(pairs)}
    r_raw = [r for r, _ in pairs]
    h_raw = [h if h.strip() else " " for _, h in pairs]
    out["wer_raw"] = jiwer.wer(r_raw, h_raw)
    out["cer_raw"] = jiwer.cer(r_raw, h_raw)

    norm = [(normalize_ar(r), normalize_ar(h)) for r, h in pairs]
    norm = [(r, h if h else " ") for r, h in norm if r]
    if norm:
        out["wer"] = jiwer.wer([r for r, _ in norm], [h for _, h in norm])
        out["cer"] = jiwer.cer([r for r, _ in norm], [h for _, h in norm])
    return out


def print_metrics(name, m):
    if not m:
        return
    print(f"  {name:<22} n={m['n']:<5} "
          f"WER={m.get('wer', float('nan')):.3f}  "
          f"CER={m.get('cer', float('nan')):.3f}   "
          f"(raw: WER={m['wer_raw']:.3f}, CER={m['cer_raw']:.3f})")


def worst_cases(refs, hyps, k):
    """The k samples with the highest per-sample normalized CER."""
    import jiwer
    scored = []
    for i, (r, h) in enumerate(zip(refs, hyps)):
        nr, nh = normalize_ar(r), normalize_ar(h)
        if not nr:
            continue
        scored.append((jiwer.cer(nr, nh if nh else " "), i, r, h))
    scored.sort(reverse=True)
    return scored[:k]


# --------------------------------------------------
# Command 1: eval on the dataset
# --------------------------------------------------

def find_text_column(column_names):
    for name in TEXT_COLUMN_CANDIDATES:
        if name in column_names:
            return name
    raise KeyError(f"No transcript column in {column_names}. Use --text-column.")


def prepare_eval_split(ds, max_samples, text_col, duration_col, seed):
    """Same subset as train_asr.py: shuffle(seed) -> select -> filter.

    (train_asr.py additionally drops clips whose labels exceed 448 Whisper
    tokens; that almost never removes anything for conversational speech.)
    """
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    if duration_col and duration_col in ds.column_names:
        ds = ds.filter(
            lambda dur, text: (
                dur is not None and MIN_AUDIO_SECONDS <= dur <= MAX_AUDIO_SECONDS
                and isinstance(text, str) and len(text.strip()) > 0),
            input_columns=[duration_col, text_col], desc="Filtering")
    else:
        ds = ds.filter(lambda text: isinstance(text, str) and len(text.strip()) > 0,
                       input_columns=[text_col], desc="Filtering")
    return ds


def load_jsonl_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_eval(args):
    from datasets import Audio, load_dataset
    from tqdm.auto import tqdm

    print(f"Loading dataset {args.dataset} ...")
    dataset = load_dataset(args.dataset, args.config)
    if args.split not in dataset:
        raise KeyError(f"Split '{args.split}' not in {list(dataset.keys())}")

    first = dataset[args.split]
    text_col = args.text_column or find_text_column(first.column_names)
    audio_col = args.audio_column
    ds = first.cast_column(audio_col, Audio(decode=False))  # decode ourselves
    ds = prepare_eval_split(ds, args.max_eval, text_col, args.duration_column, args.seed)
    n = len(ds)
    print(f"Evaluating on {n:,} clips from '{args.split}' (seed {args.seed})")

    tf = QwenCleoTranscriber(args.model, "cpu" if args.cpu else "auto", args.language,
                             args.batch_size, args.max_new_tokens)
    print("Warming up...")
    tf.warmup()

    refs, hyps, total_audio = [], [], 0.0
    t0 = time.perf_counter()
    for start in tqdm(range(0, n, args.batch_size), desc="Transcribing"):
        rows = ds[start:start + args.batch_size]
        arrays = []
        for a in rows[audio_col]:
            arr = decode_audio_field(a)
            arrays.append(arr[: int(MAX_AUDIO_SECONDS * SAMPLE_RATE)])
            total_audio += len(arrays[-1]) / SAMPLE_RATE
        hyps.extend(tf.transcribe_arrays(arrays))
        refs.extend(t if isinstance(t, str) else "" for t in rows[text_col])
    elapsed = time.perf_counter() - t0

    # ---- results ----
    print("\n" + "=" * 60)
    print(f"Results on '{args.split}' ({n:,} clips, {total_audio / 60:.1f} min audio, "
          f"{elapsed / 60:.1f} min processing, RTF {elapsed / max(total_audio, 1e-9):.3f})")
    print("=" * 60)
    results = {"split": args.split, "n_clips": n, "seed": args.seed,
               "rtf": elapsed / max(total_audio, 1e-9)}

    qwen_m = compute_metrics(refs, hyps)
    print_metrics("QwenCleo-ASR", qwen_m)
    results["qwencleo"] = qwen_m

    if args.whisper_preds:
        w = load_jsonl_predictions(args.whisper_preds)
        if len(w) != n:
            print(f"\nWARNING: Whisper predictions file has {len(w):,} rows but this "
                  f"run has {n:,}. Use the same --max-eval/--seed/--split. Skipping "
                  f"the side-by-side comparison.")
        else:
            agree = np.mean([normalize_ar(r) == normalize_ar(x.get("reference", ""))
                             for r, x in zip(refs, w)])
            if agree < 0.99:
                print(f"\nWARNING: only {agree:.0%} of references match the Whisper "
                      f"file, so the clips are not aligned. Skipping the comparison.")
            else:
                w_hyps = [x.get("prediction", "") for x in w]
                w_m = compute_metrics(refs, w_hyps)
                print_metrics("Your Whisper model", w_m)
                results["whisper"] = w_m

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"{args.split}_predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for i, (r, h) in enumerate(zip(refs, hyps)):
            f.write(json.dumps({"index": i, "reference": r, "prediction": h},
                               ensure_ascii=False) + "\n")
    res_path = os.path.join(args.output_dir, f"{args.split}_results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved predictions to {pred_path}")
    print(f"Saved metrics to {res_path}")

    if args.show_errors > 0:
        print(f"\nWorst {args.show_errors} clips by character error rate "
              f"(check whether the REFERENCE looks wrong):")
        for cer, i, r, h in worst_cases(refs, hyps, args.show_errors):
            print(f"\n  [{i}] CER={cer:.2f}\n    ref : {r}\n    pred: {h}")


# --------------------------------------------------
# Inputs for transcribe / mic
# --------------------------------------------------

def collect_audio_files(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            files += sorted(Path(g) for g in glob.glob(item)
                            if Path(g).suffix.lower() in AUDIO_EXTS)
    seen, unique = set(), []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def load_manifest(path):
    """CSV (audio,text) or JSONL ({"audio":..., "text":...}) -> {abs path: text}."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        with open(p, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    refs = {}
    for row in rows:
        audio = row.get("audio") or row.get("path") or row.get("file")
        text = row.get("text")
        if text is None:
            text = row.get("reference") or row.get("sentence")
        if audio and text is not None:
            ap = Path(audio)
            if not ap.is_absolute():
                ap = p.parent / ap
            refs[str(ap.resolve())] = str(text).strip()
    return refs


def find_reference(path: Path, manifest: dict):
    key = str(path.resolve())
    if key in manifest:
        return manifest[key]
    sibling = path.with_suffix(".txt")
    if sibling.exists():
        return sibling.read_text(encoding="utf-8-sig").strip()
    return None


class ResultWriter:
    """Writes records to .jsonl as they arrive, or to .csv when finished."""

    def __init__(self, path):
        self.path = path
        self.is_csv = bool(path) and path.lower().endswith(".csv")
        self.rows = []
        self.file = None
        if path and not self.is_csv:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.file = open(path, "w", encoding="utf-8")

    def add(self, record):
        self.rows.append(record)
        if self.file:
            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()
        elif self.path and self.is_csv and self.rows:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fields = []
            for r in self.rows:
                for k in r:
                    if k not in fields:
                        fields.append(k)
            with open(self.path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(self.rows)
        if self.path and self.rows:
            print(f"\nSaved {len(self.rows)} results to {self.path}")


def process_audio(audio, tf):
    duration = len(audio) / SAMPLE_RATE
    text, secs = tf.transcribe(audio)
    return {
        "duration_s": round(duration, 2),
        "process_s": round(secs, 3),
        "rtf": round(secs / duration, 3) if duration > 0 else None,
        "text": text,
        "text_normalized": normalize_ar(text),
    }


def show_record(rec, header):
    print(f"\n{header}")
    print(f"  {rec['duration_s']:.1f}s audio | {rec['process_s']:.2f}s to transcribe "
          f"| real-time factor {rec['rtf']}")
    print(f"  QwenCleo  : {rec['text'] or '(nothing detected)'}")
    if rec.get("reference"):
        print(f"  reference : {rec['reference']}")


def summarize(records):
    if not records:
        return
    total_audio = sum(r["duration_s"] for r in records)
    total_proc = sum(r["process_s"] for r in records)
    print("\n" + "=" * 60)
    print(f"Files/recordings: {len(records)} | audio: {total_audio:.1f}s | "
          f"processing: {total_proc:.1f}s | "
          f"overall real-time factor: {total_proc / max(total_audio, 1e-9):.3f}")
    refs = [r.get("reference") for r in records]
    if any(refs):
        print("\nAccuracy against references (normalized Arabic):")
        print_metrics("QwenCleo-ASR", compute_metrics(
            [r or "" for r in refs], [rec["text"] for rec in records]))


# --------------------------------------------------
# Command 2: transcribe files
# --------------------------------------------------

def cmd_transcribe(args):
    files = collect_audio_files(args.inputs)
    if not files:
        print("No audio files found for:", args.inputs)
        return

    tf = QwenCleoTranscriber(args.model, "cpu" if args.cpu else "auto", args.language,
                             args.batch_size, args.max_new_tokens, args.chunk_seconds)
    print("Warming up...")
    tf.warmup()

    manifest = load_manifest(args.manifest) if args.manifest else {}
    writer = ResultWriter(args.output)
    records = []
    try:
        for i, path in enumerate(files, 1):
            try:
                audio = load_audio(path)
            except Exception as e:
                print(f"\n[{i}/{len(files)}] {path}: SKIPPED ({e})")
                continue

            rec = process_audio(audio, tf)
            rec["audio"] = str(path)
            ref = find_reference(path, manifest)
            if ref:
                rec["reference"] = ref
            show_record(rec, f"[{i}/{len(files)}] {path}")
            writer.add(rec)
            records.append(rec)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        summarize(records)
        writer.close()


# --------------------------------------------------
# Command 3: microphone
# --------------------------------------------------

def record_until_enter(device=None):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Microphone mode needs: pip install sounddevice")

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    sr = SAMPLE_RATE
    try:
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                callback=callback, device=device)
    except Exception:
        sr = int(sd.query_devices(device, "input")["default_samplerate"])
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                callback=callback, device=device)

    with stream:
        input("  Recording... press Enter to stop. ")

    chunks = []
    while not q.empty():
        chunks.append(q.get())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks)[:, 0]
    return resample(audio, sr, SAMPLE_RATE)


def cmd_mic(args):
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("pip install soundfile")

    tf = QwenCleoTranscriber(args.model, "cpu" if args.cpu else "auto", args.language,
                             args.batch_size, args.max_new_tokens, args.chunk_seconds)
    print("Warming up...")
    tf.warmup()

    os.makedirs(args.recordings_dir, exist_ok=True)
    print("\nMicrophone mode. Speak in Egyptian Arabic, close to the mic.")
    if args.ask_reference:
        print("After each recording you can type the correct sentence to measure accuracy.")

    writer = ResultWriter(args.output)
    records = []
    try:
        while True:
            cmd = input("\nPress Enter to start recording (q + Enter to quit): ").strip().lower()
            if cmd in ("q", "quit", "exit"):
                break

            audio = record_until_enter(args.mic_device)
            if len(audio) < 0.3 * SAMPLE_RATE:
                print("  Too short, try again.")
                continue

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(args.recordings_dir, f"rec_{stamp}.wav")
            sf.write(wav_path, audio, SAMPLE_RATE)

            rec = process_audio(audio, tf)
            rec["audio"] = wav_path

            if args.ask_reference:
                ref = input("  Type the correct sentence (Enter to skip): ").strip()
                if ref:
                    rec["reference"] = ref
                    Path(wav_path).with_suffix(".txt").write_text(ref, encoding="utf-8")

            show_record(rec, f"Recording {len(records) + 1}")
            writer.add(rec)
            records.append(rec)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        summarize(records)
        writer.close()


# --------------------------------------------------
# CLI
# --------------------------------------------------

def add_model_args(p):
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Hugging Face id or local folder of the model")
    p.add_argument("--language", default="Arabic",
                   help="Language hint ('Arabic', or 'auto' to detect)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Clips decoded together")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--cpu", action="store_true", help="Force CPU (very slow)")


def build_parser():
    parser = argparse.ArgumentParser(
        description="QwenCleo-ASR for Egyptian Arabic: evaluate, transcribe, record.")
    sub = parser.add_subparsers(dest="command", required=True)

    # eval
    e = sub.add_parser("eval", help="Zero-shot WER/CER on your dataset")
    add_model_args(e)
    e.add_argument("--dataset", default="MAdel121/arabic-egy-cleaned")
    e.add_argument("--config", default=None)
    e.add_argument("--split", default="validation")
    e.add_argument("--text-column", default=None)
    e.add_argument("--audio-column", default="audio")
    e.add_argument("--duration-column", default="duration")
    e.add_argument("--max-eval", type=int, default=2000,
                   help="0 = whole split. Use the SAME value you trained with.")
    e.add_argument("--seed", type=int, default=42,
                   help="Must match train_asr.py (42) to evaluate the same clips")
    e.add_argument("--whisper-preds", default=None, metavar="JSONL",
                   help="validation_predictions.jsonl from your Whisper run, for a "
                        "side-by-side comparison on the same clips")
    e.add_argument("--show-errors", type=int, default=10,
                   help="Print the N worst clips (0 = off)")
    e.add_argument("--output-dir", default="./qwencleo_eval")
    e.set_defaults(func=cmd_eval)

    # transcribe
    t = sub.add_parser("transcribe", help="Transcribe audio files / folders")
    add_model_args(t)
    t.add_argument("inputs", nargs="+", help="Audio files, folders, or glob patterns")
    t.add_argument("--manifest", default=None,
                   help="CSV/JSONL with columns/keys: audio,text (references)")
    t.add_argument("--output", default="transcriptions.jsonl",
                   help="Results file (.jsonl or .csv; '' to disable)")
    t.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="Long audio is cut into chunks of at most this length")
    t.set_defaults(func=cmd_transcribe)

    # mic
    m = sub.add_parser("mic", help="Record from the microphone")
    add_model_args(m)
    m.add_argument("--mic-device", type=int, default=None,
                   help="Input device index (see: python -m sounddevice)")
    m.add_argument("--ask-reference", action="store_true",
                   help="Type the correct sentence after each recording")
    m.add_argument("--recordings-dir", default="recordings")
    m.add_argument("--output", default="mic_transcriptions.jsonl")
    m.add_argument("--chunk-seconds", type=float, default=30.0)
    m.set_defaults(func=cmd_mic)

    return parser


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()