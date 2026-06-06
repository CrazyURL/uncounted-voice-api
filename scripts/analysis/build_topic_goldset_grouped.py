# -*- coding: utf-8 -*-
"""topic 골드 재구성 — ②세그먼트ID 격리분할(누수차단) + ①실패클래스 병합.

- 그룹(=annotations 세그먼트) 단위로 train/val/holdout 분할 → 한 세그먼트의 발화가
  train·test에 동시 등장 못 함(GroupKFold 정신). 발화단위 0.77 거품 제거.
- 병합: 사회이슈 + 타 국가 이슈 → '시사/뉴스' (확산성 0.00 늪 제거).
- 출력: data/goldset/topic_grouped/{train,val,holdout}.csv  (text,label,label_group,seg_id,source)
        + holdout_segments.csv (seg_text,label,seg_id)  ← 세그먼트단위 평가용(발화 묶음)
"""
import os, json, glob, zipfile, collections, random, re, csv

AIHUB = None
for c in ["/mnt/c/Users/gdash/Downloads/AIHUB/주제별 텍스트 일상 대화 데이터",
          "/mnt/d/ai hub/AIHUB/주제별 텍스트 일상 대화 데이터"]:
    if os.path.isdir(c):
        AIHUB = c; break
OUT = "/home/gdash/project/Uncounted-root/uncounted-voice-api/data/goldset/topic_grouped"
_SP = re.compile(r"^[^:：]{1,12}\s*[:：]\s*")
random.seed(7)

# 20 subject (병합 적용). 사회이슈/타 국가 이슈 → 시사/뉴스
MERGE = {"사회이슈": "시사/뉴스", "타 국가 이슈": "시사/뉴스"}
BASE20 = ["가족", "건강", "게임", "계절/날씨", "교육", "교통", "군대", "미용", "반려동물",
          "방송/연예", "사회이슈", "상거래 전반", "스포츠/레저", "식음료", "여행", "연애/결혼",
          "영화/만화", "주거와 생활", "타 국가 이슈", "회사/아르바이트"]
NORM = {"상거래전반": "상거래 전반"}

def norm_subj(raw):
    s = (raw or "").strip()
    s = NORM.get(s.replace(" ", ""), s)
    if s not in BASE20:
        return None
    return MERGE.get(s, s)

def strip_prefix(t):
    return _SP.sub("", t).strip()

# --- 세그먼트 수집 ---
segments = []  # {seg_id, label, texts}
sid = 0
for zp in sorted(glob.glob(os.path.join(AIHUB, "**", "*.zip"), recursive=True)):
    try:
        zf = zipfile.ZipFile(zp)
    except Exception:
        continue
    for jn in zf.namelist():
        if not jn.lower().endswith(".json"):
            continue
        try:
            d = json.loads(zf.read(jn).decode("utf-8"))
        except Exception:
            continue
        for info in d.get("info", []):
            ann = info.get("annotations", {})
            lab = norm_subj(ann.get("subject"))
            if not lab:
                continue
            txts = [strip_prefix(ln.get("norm_text") or ln.get("text") or "") for ln in ann.get("lines", [])]
            txts = [t for t in txts if len(t) >= 2]
            if len(txts) < 3:
                continue
            segments.append({"seg_id": sid, "label": lab, "texts": txts})
            sid += 1

by = collections.defaultdict(list)
for s in segments:
    by[s["label"]].append(s)
print("세그먼트 수집:", {k: len(v) for k, v in by.items()})
labels = sorted(by)
print("클래스 수(병합후):", len(labels))

# --- 세그먼트 단위 격리분할 80/10/10 (클래스별 stratified) ---
tr_seg, va_seg, ho_seg = [], [], []
for lab, segs in by.items():
    random.shuffle(segs)
    n = len(segs); a, b = int(n * 0.8), int(n * 0.9)
    tr_seg += segs[:a]; va_seg += segs[a:b]; ho_seg += segs[b:]

# --- 발화단위 행 생성 + 클래스 균형(undersample, train 기준) ---
def seg_to_rows(segs):
    rows = []
    for s in segs:
        for t in s["texts"]:
            rows.append({"text": t, "label": s["label"], "label_group": s["label"],
                         "seg_id": s["seg_id"], "source": "aihub_020"})
    return rows

tr_rows = seg_to_rows(tr_seg); va_rows = seg_to_rows(va_seg); ho_rows = seg_to_rows(ho_seg)

def balance(rows, cap):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["label"]].append(r)
    mn = min(min(len(v) for v in g.values()), cap)
    out = []
    for lab in labels:
        random.shuffle(g[lab]); out += g[lab][:mn]
    random.shuffle(out)
    return out, mn

tr_bal, mn = balance(tr_rows, 600)   # 클래스당 600 → 격리 유지하며 CPU 학습 가능 규모
va_bal, _ = balance(va_rows, 100)
ho_bal, _ = balance(ho_rows, 100)
print("발화단위(격리·균형): train %d(클래스당%d) val %d hold %d" % (len(tr_bal), mn, len(va_bal), len(ho_bal)))

os.makedirs(OUT, exist_ok=True)
cols = ["text", "label", "label_group", "seg_id", "source"]
for name, rows in [("train", tr_bal), ("val", va_bal), ("holdout", ho_bal)]:
    with open(os.path.join(OUT, name + ".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

# --- 세그먼트단위 홀드아웃(발화 묶음) — 클래스 균형 ---
hog = collections.defaultdict(list)
for s in ho_seg:
    hog[s["label"]].append(s)
mns = min(len(v) for v in hog.values())
with open(os.path.join(OUT, "holdout_segments.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["seg_text", "label", "seg_id"])
    for lab in labels:
        random.shuffle(hog[lab])
        for s in hog[lab][:mns]:
            w.writerow([" ".join(s["texts"]), lab, s["seg_id"]])
print("세그먼트 홀드아웃: 클래스당 %d (총 %d)" % (mns, mns * len(labels)))
print("출력:", OUT)
