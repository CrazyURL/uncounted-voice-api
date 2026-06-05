# -*- coding: utf-8 -*-
"""MOS 백필 — 발화 WAV에 non-intrusive MOS 추정(torchaudio SQUIM, CPU). 로드맵 #2.

POLQA/PESQ(intrusive)는 클린 원본 필요 → 우리 불가. SQUIM은 reference-free.
  - mos_score: SQUIM subjective MOS 추정(1-5). NMR=고정 클린참조(데이터 내 최고품질 1개).
  - mos_pesq : SQUIM objective PESQ 추정(reference-free).
GPU 무접촉(CPU). 재처리 완료 후 실행 권장(최종 WAV 기준). migration 20260605 선적용 필수.

사용: PYTHONPATH=. python3 scripts/analysis/mos_backfill.py [--limit N] [--apply]
  --apply 없으면 dry-run(측정만, DB write 0).
"""
import os, io, json, sys, urllib.request, argparse
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU 강제

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)   # 0=전체
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
os.environ.update(env)
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K, "Content-Type": "application/json"}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))
def PATCH(p, b): return urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, data=json.dumps(b).encode(), method="PATCH", headers=H), timeout=20).status

import torch, torchaudio
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
sub = SQUIM_SUBJECTIVE.get_model().to("cpu").eval()
obj = SQUIM_OBJECTIVE.get_model().to("cpu").eval()
import app.worker as W, boto3, soundfile as sf
from botocore.config import Config
W._s3 = boto3.client("s3", endpoint_url=W.S3_ENDPOINT_URL, aws_access_key_id=W.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=W.AWS_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4",
    request_checksum_calculation="when_required", response_checksum_validation="when_required"))

def load(sp):
    buf = io.BytesIO(); W._s3.download_fileobj(W.S3_AUDIO_BUCKET, sp, buf); buf.seek(0)
    a, sr = sf.read(buf)
    if a.ndim > 1: a = a.mean(axis=1)
    wav = torch.tensor(a, dtype=torch.float32).unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def seg_snr(wav, sr=16000, frame_ms=20):
    """로드맵 #3: true noise SNR(세그먼탈) — 활성음성 vs 하위10% 잡음바닥. crest factor 아님."""
    a = wav.squeeze(0).numpy().astype(np.float64)
    n = int(sr * frame_ms / 1000)
    if len(a) < n * 5: return None
    pw = np.array([float(np.mean(a[i:i+n] ** 2)) for i in range(0, len(a) - n, n)])
    pw = pw[pw > 0]
    if len(pw) < 5: return None
    noise = np.percentile(pw, 10); speech = np.mean(pw[pw >= np.percentile(pw, 60)])
    if noise <= 0 or speech <= 0: return None
    return 10 * np.log10(speech / noise)

# NMR(고정 클린참조) = 첫 발화 중 충분히 긴 것
ref_rows = GET("utterances?select=storage_path&storage_path=not.is.null&quality_grade=eq.A&limit=5")
NMR = None
for r in ref_rows:
    try:
        w = load(r["storage_path"])
        if w.shape[1] >= 16000: NMR = w; break
    except Exception: pass

q = "utterances?select=id,storage_path&storage_path=not.is.null&mos_score=is.null&order=id.asc&limit="
lim = args.limit if args.limit else 1000
done = 0; vals = []
off = 0
while True:
    rows = GET(q + str(lim) + (f"&offset={off}" if not args.limit else ""))
    if not rows: break
    for u in rows:
        try:
            w = load(u["storage_path"])
            with torch.no_grad():
                mos = float(sub(w, NMR if NMR is not None else w)[0])
                _, pesq, _ = obj(w)
                pesq = float(pesq[0])
            tsnr = seg_snr(w)
            vals.append(mos)
            if args.apply:
                body = {"mos_score": round(mos, 3), "mos_pesq": round(pesq, 3), "mos_method": "squim_v1"}
                if tsnr is not None: body["true_snr_db"] = round(tsnr, 2)
                PATCH(f"utterances?id=eq.{u['id']}", body)
            done += 1
        except Exception:
            pass
    if args.limit or len(rows) < lim: break
    off += lim
v = np.array(vals)
print(f"{'적용' if args.apply else 'dry-run'} 완료: {done}건 | MOS 중앙 {np.median(v):.2f} 평균 {v.mean():.2f} (n={len(v)})" if len(v) else "대상 없음")
