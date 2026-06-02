# -*- coding: utf-8 -*-
"""NeMo MSDD 도입부 화자분리 — 격리 마이크로서비스 (FastAPI).

운영 voice-api venv 와 **분리된** NeMo venv 에서 구동한다(torch 버전 충돌 회피,
VRAM 격리). 메인 voice-api 의 app/services/hybrid_diarization.py 가 HTTP 로 호출.

구동 (NeMo venv 에서):
  cd /home/gdash/_poc && \
  HF_TOKEN=<token> ./nemo_venv/bin/python -m uvicorn \
    --app-dir /home/gdash/project/Uncounted-root/uncounted-voice-api/scripts \
    nemo_service:app --host 127.0.0.1 --port 8009

의존: nemo-toolkit[asr], soundfile, fastapi, uvicorn, torch (NeMo venv).
설정 자산: diar_telephonic.yaml (NeMo 공식 telephonic config).

반환: {status, turns:[{start,end,nemo_spk}], embeddings:{nemo_spk:[256-dim]}}
  embeddings = 각 화자 최장 turn 의 WeSpeaker 임베딩 (메인의 코사인 ID매핑용, PoC2.5).
PII: 오디오 경로만 받고 transcript/실명 미반환. 화자라벨·시간·임베딩 벡터만.
"""
import contextlib
import glob
import json
import os
import shutil
import time

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel

app = FastAPI(title="Uncounted-NeMo-Diarization-Service")

CONFIG_PATH = os.environ.get("NEMO_DIAR_CONFIG", "/home/gdash/_poc/diar_telephonic.yaml")
BASE_OUT_DIR = os.environ.get("NEMO_SERVICE_OUT", "/home/gdash/_poc/nemo_service_out")
EMB_MIN_SEC = 0.7   # 임베딩 최소 길이(짧으면 중심확장)

# 프로세스 간 GPU 락 — voice-api(STT)와 동시 추론 방지(VRAM peak 충돌 회피).
# voice-api 의 app/services/gpu_process_lock.py 와 같은 파일 경로/게이트를 공유한다.
_GPU_LOCK_ENABLED = os.environ.get("VOICE_GPU_PROCESS_LOCK_ENABLED", "false").strip().lower() == "true"
_GPU_LOCK_PATH = os.environ.get("VOICE_GPU_PROCESS_LOCK_PATH", "/tmp/voice_gpu.lock")
try:
    _GPU_LOCK_TIMEOUT = float(os.environ.get("VOICE_GPU_PROCESS_LOCK_TIMEOUT", "120"))
except ValueError:
    _GPU_LOCK_TIMEOUT = 120.0


@contextlib.contextmanager
def _gpu_lock():
    """voice-api 와 공유하는 프로세스 간 GPU 락. 게이트 OFF/미가용 시 no-op."""
    if not _GPU_LOCK_ENABLED:
        yield
        return
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        yield
        return
    lock = FileLock(_GPU_LOCK_PATH, timeout=_GPU_LOCK_TIMEOUT)
    acquired = False
    try:
        try:
            lock.acquire(); acquired = True
        except Timeout:
            pass
        yield
    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                pass

_emb_model = None


def _get_emb_model():
    """WeSpeaker 임베딩 모델 lazy load (titanet 와 별개, pyannote/wespeaker)."""
    global _emb_model
    if _emb_model is None:
        from pyannote.audio import Model
        repo = os.environ.get("VOICE_DIARIZATION_WESPEAKER_REPO",
                              "pyannote/wespeaker-voxceleb-resnet34-LM")
        tok = os.environ.get("HF_TOKEN")
        _emb_model = Model.from_pretrained(repo, token=tok) if tok else Model.from_pretrained(repo)
        _emb_model.eval()
        if torch.cuda.is_available():
            _emb_model.to(torch.device("cuda"))
    return _emb_model


def _embed(audio: np.ndarray, sr: int) -> list[float] | None:
    if len(audio) < int(EMB_MIN_SEC * sr):
        return None
    try:
        model = _get_emb_model()
        dev = next(model.parameters()).device
        t = torch.from_numpy(audio.astype(np.float32)).reshape(1, 1, -1).to(dev)
        with torch.no_grad():
            out = model(t)
        emb = out.detach().cpu().numpy().astype("float32").reshape(-1)
        n = float(np.linalg.norm(emb))
        return (emb / n).tolist() if n > 1e-12 else None
    except Exception:
        return None


class DiarizeRequest(BaseModel):
    audio_path: str
    window_seconds: float = 30.0   # PoC sweet spot


@app.get("/health")
def health():
    return {"status": "ok", "cuda": torch.cuda.is_available()}


@app.post("/api/diarize/intro")
def diarize_intro(req: DiarizeRequest):
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=404, detail="audio not found")

    audio, sr = sf.read(req.audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    intro = audio[: int(req.window_seconds * sr)]

    task_id = f"task_{int(time.time() * 1000)}"      # 버그수정: time.time (os.time 아님)
    task_dir = os.path.join(BASE_OUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    slice_wav = os.path.join(task_dir, "intro.wav")
    sf.write(slice_wav, intro, sr)

    try:
        manifest = os.path.join(task_dir, "manifest.json")
        with open(manifest, "w") as f:
            f.write(json.dumps({
                "audio_filepath": slice_wav, "offset": 0, "duration": None,
                "label": "infer", "text": "-", "num_speakers": 2,
                "rttm_filepath": None, "uem_filepath": None,
            }) + "\n")

        cfg = OmegaConf.load(CONFIG_PATH)
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.verbose = False
        cfg.diarizer.manifest_filepath = manifest
        cfg.diarizer.out_dir = task_dir
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True
        cfg.diarizer.ignore_overlap = False

        # voice-api STT 와 GPU 동시 사용 방지(프로세스 간 락). 게이트 OFF 면 no-op.
        from nemo.collections.asr.models import ClusteringDiarizer
        with _gpu_lock():
            ClusteringDiarizer(cfg=cfg).diarize()

        rttms = glob.glob(f"{task_dir}/**/*.rttm", recursive=True)
        if not rttms:
            raise RuntimeError("no RTTM produced")
        turns = []
        for line in open(rttms[0]):
            p = line.split()
            if p and p[0] == "SPEAKER":
                st = round(float(p[3]), 2)
                turns.append({"start": st, "end": round(st + float(p[4]), 2), "nemo_spk": p[7]})
        turns.sort(key=lambda x: x["start"])

        # PoC2.5: 화자별 최장 turn 으로 대표 임베딩 (코사인 ID매핑용)
        longest: dict[str, tuple[float, float]] = {}
        for t in turns:
            spk = t["nemo_spk"]
            dur = t["end"] - t["start"]
            if spk not in longest or dur > (longest[spk][1] - longest[spk][0]):
                longest[spk] = (t["start"], t["end"])
        embeddings: dict[str, list[float]] = {}
        for spk, (s, e) in longest.items():
            if e - s < EMB_MIN_SEC:
                c = (s + e) / 2
                s, e = max(0, c - EMB_MIN_SEC / 2), c + EMB_MIN_SEC / 2
            seg = intro[int(s * sr): int(e * sr)]
            emb = _embed(seg, sr)
            if emb is not None:
                embeddings[spk] = emb

        torch.cuda.empty_cache()
        return {"status": "success", "turns": turns, "embeddings": embeddings}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)
