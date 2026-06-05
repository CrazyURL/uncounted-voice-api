# -*- coding: utf-8 -*-
"""노이즈 카테고리 백필 (clean/babble/street/static 등) — STT 강건성 메타.

PANNs(audio tagging) 로 발화 WAV 의 배경음 분류 → STT 환경별 강건성 학습용(빅테크 요구).
⚠️ 재처리 완료 후 실행. migration 20260605 선적용. 의존성: pip install panns_inference
   (첫 실행 시 모델 ~300MB 다운로드). 미설치 시 안내만.

사용: PYTHONPATH=. python3 scripts/analysis/noise_category_backfill.py [--limit N] [--apply] [--cpu]
"""
import os, io, json, urllib.request, argparse
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--apply", action="store_true")
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()
if args.cpu: os.environ["CUDA_VISIBLE_DEVICES"] = ""

env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
os.environ.update(env)
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K, "Content-Type": "application/json"}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))
def PATCH(p, b): return urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, data=json.dumps(b).encode(), method="PATCH", headers=H), timeout=20).status

try:
    from panns_inference import AudioTagging
    import torch
except ImportError:
    print("panns_inference 미설치 — `pip install panns_inference` 후 재실행 (모델 자동 다운로드).")
    raise SystemExit

# AudioSet 태그 → 통화 노이즈 카테고리 매핑
CAT = {
    "Speech": "clean", "Babble": "babble", "Crowd": "babble", "Hubbub, speech noise": "babble",
    "Traffic noise, roadway noise": "street", "Vehicle": "street", "Car": "street",
    "White noise": "static", "Pink noise": "static", "Static": "static", "Noise": "static",
    "Music": "music",
}
dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
at = AudioTagging(checkpoint_path=None, device=dev)
print(f"PANNs AudioTagging 로드 OK (device={dev})")
import app.worker as W, boto3, soundfile as sf
from botocore.config import Config
W._s3 = boto3.client("s3", endpoint_url=W.S3_ENDPOINT_URL, aws_access_key_id=W.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=W.AWS_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4",
    request_checksum_calculation="when_required", response_checksum_validation="when_required"))
labels = at.labels

def categorize(sp):
    buf = io.BytesIO(); W._s3.download_fileobj(W.S3_AUDIO_BUCKET, sp, buf); buf.seek(0)
    a, sr = sf.read(buf)
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != 32000:
        import torchaudio; a = torchaudio.functional.resample(torch.tensor(a, dtype=torch.float32), sr, 32000).numpy()
    clipwise, _ = at.inference(a[None, :])
    top = np.argsort(clipwise[0])[::-1][:5]
    for i in top:
        lab = labels[i]
        if lab in CAT and lab != "Speech":
            return CAT[lab], float(clipwise[0][i])
    return "clean", float(clipwise[0][top[0]])

lim = args.limit if args.limit else 1000
done = 0; off = 0; cats = {}
while True:
    rows = GET(f"utterances?select=id,storage_path&storage_path=not.is.null&noise_category=is.null&order=id.asc&limit={lim}" + (f"&offset={off}" if not args.limit else ""))
    if not rows: break
    for u in rows:
        try:
            cat, conf = categorize(u["storage_path"]); cats[cat] = cats.get(cat, 0) + 1
            if args.apply: PATCH(f"utterances?id=eq.{u['id']}", {"noise_category": cat, "noise_confidence": round(conf, 3)})
            done += 1
        except Exception: pass
    if args.limit or len(rows) < lim: break
    off += lim
print(f"{'적용' if args.apply else 'dry-run'} {done}건 | 분포: {cats}")
