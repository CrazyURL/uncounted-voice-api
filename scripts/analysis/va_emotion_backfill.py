# -*- coding: utf-8 -*-
"""V-A(Valence-Arousal) 차원 감정 백필 (EmotionML, audeering wav2vec2-dim).

카테고리 감정(emotion)에 2D 연속차원(valence/arousal/dominance) 추가 = 빅테크 EmotionML.
⚠️ 재처리 완료 후 실행(GPU 경합 회피). migration 20260605_add_advanced_labels 선적용.
모델은 첫 실행 시 자동 다운로드(~1GB).

사용: PYTHONPATH=. python3 scripts/analysis/va_emotion_backfill.py [--limit N] [--apply] [--cpu]
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

import torch, torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"  # arousal/dominance/valence

# ★audeering 커스텀 회귀 구조(모델카드). AutoModelForAudioClassification 은 헤드를 못 읽어
# 랜덤초기화(출력 쓰레기)됨 — 반드시 이 커스텀 클래스로 로드해야 회귀 헤드가 실린다.
class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, x):
        x = self.dropout(x); x = torch.tanh(self.dense(x)); x = self.dropout(x)
        return self.out_proj(x)

class _EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()
    def forward(self, input_values):
        h = self.wav2vec2(input_values)[0]
        h = torch.mean(h, dim=1)
        return self.classifier(h)

dev = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
proc = Wav2Vec2Processor.from_pretrained(MODEL)
model = _EmotionModel.from_pretrained(MODEL).to(dev).eval()
print(f"audeering V-A 모델 로드 OK (device={dev})")

import app.worker as W, boto3, soundfile as sf
from botocore.config import Config
W._s3 = boto3.client("s3", endpoint_url=W.S3_ENDPOINT_URL, aws_access_key_id=W.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=W.AWS_SECRET_ACCESS_KEY, config=Config(signature_version="s3v4",
    request_checksum_calculation="when_required", response_checksum_validation="when_required"))

def va(sp):
    buf = io.BytesIO(); W._s3.download_fileobj(W.S3_AUDIO_BUCKET, sp, buf); buf.seek(0)
    a, sr = sf.read(buf)
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != 16000:
        import torchaudio; a = torchaudio.functional.resample(torch.tensor(a, dtype=torch.float32), sr, 16000).numpy()
    inp = proc(a, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        out = model(inp).squeeze().cpu().numpy()  # [arousal, dominance, valence] 0~1
    aro, dom, val = float(out[0]), float(out[1]), float(out[2])
    return val * 2 - 1, aro, dom  # valence -1~+1, arousal/dominance 0~1

lim = args.limit if args.limit else 1000
done = 0; off = 0; vals = []
while True:
    rows = GET(f"utterances?select=id,storage_path&storage_path=not.is.null&emotion_valence=is.null&order=id.asc&limit={lim}" + (f"&offset={off}" if not args.limit else ""))
    if not rows: break
    for u in rows:
        try:
            v, ar, do = va(u["storage_path"]); vals.append(v)
            if args.apply:
                PATCH(f"utterances?id=eq.{u['id']}", {"emotion_valence": round(v,3), "emotion_arousal": round(ar,3), "emotion_dominance": round(do,3)})
            done += 1
        except Exception: pass
    if args.limit or len(rows) < lim: break
    off += lim
print(f"{'적용' if args.apply else 'dry-run'} {done}건 | valence 중앙 {np.median(vals):.2f} (n={len(vals)})" if vals else "대상 없음")
