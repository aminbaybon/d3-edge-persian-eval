"""Limited zero-shot Persian classification eval for vllm-sr/d3-edge."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy

# NumPy 2 dropped np.Inf; conda SciPy still imports it.
if not hasattr(numpy, "Inf"):
    numpy.Inf = numpy.inf
    numpy.Infinity = numpy.inf
    numpy.PINF = numpy.inf
    numpy.NINF = -numpy.inf
    numpy.NaN = numpy.nan

from datasets import load_dataset
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
FARSTAIL_URL = (
    "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv"
)

PN_CATEGORY_MAP = {
    0: "economy",
    1: "roads_urban",
    2: "banking_insurance",
    3: "agriculture",
    4: "international",
    5: "oil_energy",
    6: "industry",
    7: "transportation",
    8: "science_technology",
    9: "local",
    10: "sports",
    11: "politics",
    12: "art_culture",
    13: "society",
    14: "health",
    15: "research",
    16: "education_university",
    17: "tourism",
}

PN_KEEP = {
    "economy",
    "international",
    "science_technology",
    "sports",
    "politics",
    "art_culture",
    "society",
    "health",
}


def _truncate(text: str, limit: int = 1200) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _choice_question(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {
        "label": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


def extract_choice(result: dict[str, Any], qid: str = "label") -> str:
    answers = result.get("answers", result)
    payload = answers.get(qid, answers)
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected answer payload: {payload!r}")
    for key in ("answer", "choice", "label", "prediction", "value", "id", "name"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and key in payload:
            return str(value)
    raise KeyError(f"Cannot parse answer: {payload}")


def stratified_sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["gold"])].append(row)
    rng = random.Random(seed)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    picked: list[dict[str, Any]] = []
    labels = list(by_label)
    while labels and len(picked) < n:
        next_labels = []
        rng.shuffle(labels)
        for label in labels:
            if not by_label[label]:
                continue
            picked.append(by_label[label].pop())
            if by_label[label]:
                next_labels.append(label)
            if len(picked) >= n:
                break
        labels = next_labels
    rng.shuffle(picked)
    return picked


def load_snappfood(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("ParsiAI/snappfood-sentiment-analysis", split="test")
    criteria = {
        "HAPPY": "نظر مثبت، راضی یا خوشحال درباره غذا یا سفارش",
        "SAD": "نظر منفی، ناراضی یا ناراحت درباره غذا یا سفارش",
    }
    rows = []
    for item in ds:
        gold = str(item["label"]).upper()
        if gold not in criteria:
            continue
        rows.append(
            {
                "state": _truncate(item["comment"]),
                "gold": gold,
                "questions": _choice_question(
                    "قطبیت این نظر غذا درباره سفارش چیست؟", criteria
                ),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "ParsiAI/snappfood-sentiment-analysis",
        "task": "binary_sentiment",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_deepsentipers(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("mteb/DeepSentiPers", split="test")
    # MTEB remap inspected on examples: 0 positive, 1 negative, 2 factual/neutral.
    mapping = {"0": "positive", "1": "negative", "2": "neutral"}
    criteria = {
        "negative": "نظر منفی یا ناراضی درباره محصول",
        "neutral": "جمله خنثی یا توصیف فنی بدون ارزیابی احساسی",
        "positive": "نظر مثبت یا راضی درباره محصول",
    }
    rows = []
    for item in ds:
        gold = mapping[str(item["label"])]
        rows.append(
            {
                "state": _truncate(item["text"]),
                "gold": gold,
                "questions": _choice_question(
                    "قطبیت این نظر محصول دیجیتال چیست؟", criteria
                ),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "mteb/DeepSentiPers",
        "task": "ternary_sentiment",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_parsinlu_sentiment(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset(
        "persiannlp/parsinlu_sentiment", name="parsinlu-repo", split="test_food"
    )
    collapse = {
        "-3": "none",
        "-2": "negative",
        "-1": "negative",
        "0": "neutral",
        "1": "positive",
        "2": "positive",
    }
    criteria = {
        "none": "درباره این جنبه هیچ احساسی بیان نشده",
        "negative": "احساس منفی نسبت به این جنبه",
        "neutral": "احساس خنثی نسبت به این جنبه",
        "positive": "احساس مثبت نسبت به این جنبه",
    }
    rows = []
    for item in ds:
        gold = collapse.get(str(item["label"]))
        if gold is None:
            continue
        state = (
            f"نظر: {_truncate(item['review'], 900)}\n"
            f"جنبه: {item.get('aspect') or ''}\n"
            f"پرسش: {item.get('question') or ''}"
        )
        rows.append(
            {
                "state": state,
                "gold": gold,
                "questions": _choice_question(
                    "احساس نویسنده نسبت به جنبه ذکرشده چیست؟", criteria
                ),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "persiannlp/parsinlu_sentiment",
        "task": "aspect_sentiment",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_farstail(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / "FarsTail-Test-word.csv"
    if not csv_path.exists():
        with urllib.request.urlopen(FARSTAIL_URL, timeout=60) as response:
            csv_path.write_bytes(response.read())
    criteria = {
        "e": "جمله دوم از جمله اول نتیجه می‌شود (entailment)",
        "c": "جمله دوم با جمله اول در تضاد است (contradiction)",
        "n": "جمله دوم نسبت به جمله اول خنثی است (neutral)",
    }
    rows = []
    raw = csv_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
    fieldnames = reader.fieldnames or []
    # Official file is tab-separated: premise, hypothesis, label
    if len(fieldnames) == 1:
        reader = csv.DictReader(io.StringIO(raw))
    for item in reader:
        values = list(item.values())
        if len(values) < 3:
            continue
        premise, hypothesis, label = values[0], values[1], str(values[2]).strip().lower()
        if label not in criteria:
            continue
        state = f"مقدمه: {_truncate(premise, 700)}\nفرضیه: {_truncate(hypothesis, 400)}"
        rows.append(
            {
                "state": state,
                "gold": label,
                "questions": _choice_question(
                    "رابطه منطقی فرضیه نسبت به مقدمه چیست؟", criteria
                ),
            }
        )
    if not rows:
        raise RuntimeError("FarsTail CSV parsed zero labeled rows")
    return stratified_sample(rows, n, seed), {
        "hf_id": "dml-qom/FarsTail (official Test-word.csv)",
        "task": "nli",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_parsinlu_entailment(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset(
        "persiannlp/parsinlu_entailment", name="parsinlu-repo", split="test"
    )
    criteria = {
        "e": "جمله دوم از جمله اول نتیجه می‌شود",
        "c": "جمله دوم با جمله اول در تضاد است",
        "n": "جمله دوم نسبت به جمله اول خنثی است",
    }
    rows = []
    for item in ds:
        gold = str(item["label"]).strip().lower()
        if gold not in criteria:
            continue
        state = (
            f"جمله ۱: {_truncate(item['sent1'], 700)}\n"
            f"جمله ۲: {_truncate(item['sent2'], 400)}"
        )
        rows.append(
            {
                "state": state,
                "gold": gold,
                "questions": _choice_question(
                    "رابطه منطقی جمله دوم نسبت به جمله اول چیست؟", criteria
                ),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "persiannlp/parsinlu_entailment",
        "task": "nli",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_parsinlu_paraphrase(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset(
        "persiannlp/parsinlu_query_paraphrasing",
        name="parsinlu-repo",
        split="test",
    )
    criteria = {
        "0": "دو پرسش معنای متفاوت دارند",
        "1": "دو پرسش هم‌معنا یا بازنویسی یکدیگرند",
    }
    rows = []
    for item in ds:
        gold = str(item["label"]).strip()
        if gold not in criteria:
            continue
        state = f"پرسش ۱: {_truncate(item['q1'], 400)}\nپرسش ۲: {_truncate(item['q2'], 400)}"
        rows.append(
            {
                "state": state,
                "gold": gold,
                "questions": _choice_question(
                    "آیا این دو پرسش هم‌معنا هستند؟", criteria
                ),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "persiannlp/parsinlu_query_paraphrasing",
        "task": "paraphrase",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_sid(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("MCINext/sid-classification", split="test")
    texts = {}
    rows_raw = []
    for item in ds:
        label = str(item["label_text"]).strip()
        texts[label] = label
        rows_raw.append((item["text"], label))
    criteria = {label: f"مقاله علمی در حوزه {label}" for label in sorted(texts)}
    rows = [
        {
            "state": _truncate(text, 1000),
            "gold": label,
            "questions": _choice_question(
                "این مقاله علمی به کدام حوزه تعلق دارد؟", criteria
            ),
        }
        for text, label in rows_raw
        if label in criteria
    ]
    return stratified_sample(rows, n, seed), {
        "hf_id": "MCINext/sid-classification",
        "task": "topic_classification",
        "labels": list(criteria),
        "chance": 1 / max(len(criteria), 1),
    }


def load_pn_summary(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("HooshvareLab/pn_summary", split="test", trust_remote_code=True)
    criteria = {
        "economy": "اقتصاد و بازار",
        "international": "اخبار بین‌الملل",
        "science_technology": "علم و فناوری",
        "sports": "ورزش",
        "politics": "سیاست",
        "art_culture": "هنر و فرهنگ",
        "society": "جامعه",
        "health": "سلامت و بهداشت",
    }
    rows = []
    for item in ds:
        raw = item.get("category")
        gold = PN_CATEGORY_MAP.get(int(raw), None) if raw is not None else None
        if gold not in PN_KEEP:
            continue
        title = item.get("title") or ""
        summary = (item.get("summary") or "").replace("[n]", " ")
        state = _truncate(f"{title}\n{summary}", 1000)
        rows.append(
            {
                "state": state,
                "gold": gold,
                "questions": _choice_question("موضوع این خبر چیست؟", criteria),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "HooshvareLab/pn_summary",
        "task": "news_topic",
        "labels": list(criteria),
        "chance": 1 / len(criteria),
    }


def load_parsinlu_mcqa(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("PartAI/ParsiNLU-multiple-choice", split="test")
    rows = []
    for item in ds:
        gold_idx = int(item["label"])
        options = [item[f"ending{i}"] for i in range(4)]
        if any(opt is None or str(opt).strip() == "" for opt in options):
            continue
        criteria = {str(i): str(options[i]) for i in range(4)}
        context = (item.get("context") or "").strip()
        question = item["question"]
        state = _truncate(f"{context}\n{question}" if context else question, 1100)
        rows.append(
            {
                "state": state,
                "gold": str(gold_idx),
                "questions": _choice_question("گزینه درست کدام است؟", criteria),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "PartAI/ParsiNLU-multiple-choice",
        "task": "multiple_choice_qa",
        "labels": ["0", "1", "2", "3"],
        "chance": 0.25,
    }


def load_persian_digits(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("Mehdinmz/persian-handwritten-digits", split="train")
    names = None
    if "label" in ds.features and getattr(ds.features["label"], "names", None):
        names = list(ds.features["label"].names)
    criteria = {
        str(i): f"رقم دست‌نویس فارسی {i} (شکل {eastern})"
        for i, eastern in enumerate("۰۱۲۳۴۵۶۷۸۹")
    }
    rows = []
    for item in ds:
        if "label" in item:
            gold = str(int(item["label"]))
        else:
            path = str(item["image"].filename if hasattr(item["image"], "filename") else "")
            gold = Path(path).parent.name
            if names and gold in names:
                gold = str(names.index(gold))
        if gold not in criteria:
            continue
        image = item["image"].convert("RGB")
        rows.append(
            {
                "state": "این تصویر یک رقم دست‌نویس فارسی است.",
                "images": [image],
                "gold": gold,
                "questions": _choice_question(
                    "این تصویر کدام رقم فارسی را نشان می‌دهد؟", criteria
                ),
            }
        )
        if len(rows) >= max(n * 12, 400):
            break
    return stratified_sample(rows, n, seed), {
        "hf_id": "Mehdinmz/persian-handwritten-digits",
        "task": "persian_digit_image",
        "labels": list(criteria),
        "chance": 0.1,
        "modality": "image",
    }


def load_persian_text_sentiment(n: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ds = load_dataset("SeyedAli/Persian-Text-Sentiment", split="test")
    criteria = {
        "0": "نظر منفی یا ناراضی",
        "1": "نظر مثبت یا راضی",
    }
    rows = []
    for item in ds:
        gold = str(int(item["label"]))
        if gold not in criteria:
            continue
        rows.append(
            {
                "state": _truncate(item["text"]),
                "gold": gold,
                "questions": _choice_question("قطبیت این نظر فارسی چیست؟", criteria),
            }
        )
    return stratified_sample(rows, n, seed), {
        "hf_id": "SeyedAli/Persian-Text-Sentiment",
        "task": "binary_sentiment",
        "labels": list(criteria),
        "chance": 0.5,
    }


DATASETS = {
    "snappfood": load_snappfood,
    "deepsentipers": load_deepsentipers,
    "parsinlu_sentiment": load_parsinlu_sentiment,
    "farstail": load_farstail,
    "parsinlu_entailment": load_parsinlu_entailment,
    "parsinlu_paraphrase": load_parsinlu_paraphrase,
    "sid_classification": load_sid,
    "pn_summary_news": load_pn_summary,
    "parsinlu_mcqa": load_parsinlu_mcqa,
    "persian_text_sentiment": load_persian_text_sentiment,
    "persian_digits": load_persian_digits,
}


def load_model(device: str):
    print(f"Loading vllm-sr/d3-edge on {device} ...", flush=True)
    started = time.time()
    model = AutoModel.from_pretrained(
        "vllm-sr/d3-edge",
        trust_remote_code=True,
        device=device,
        verify="none",
    )
    print(f"Model ready in {time.time() - started:.1f}s", flush=True)
    return model


def evaluate_dataset(
    model,
    name: str,
    rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    correct = 0
    details = []
    latencies = []
    errors = 0
    confusion: dict[str, int] = Counter()
    for index, row in enumerate(rows, start=1):
        started = time.time()
        try:
            result = model.system_one(
                state=row["state"],
                questions=row["questions"],
                images=row.get("images"),
            )
            pred = extract_choice(result)
            latency = time.time() - started
            latencies.append(latency)
            ok = pred == row["gold"]
            correct += int(ok)
            confusion[f"{row['gold']}->{pred}"] += 1
            details.append(
                {
                    "gold": row["gold"],
                    "pred": pred,
                    "ok": ok,
                    "latency_s": round(latency, 3),
                    "raw": result.get("answers"),
                }
            )
            print(
                f"  [{name} {index}/{len(rows)}] gold={row['gold']} pred={pred} "
                f"{'OK' if ok else 'ERR'} {latency:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - keep the run going
            errors += 1
            details.append({"gold": row["gold"], "error": str(exc)})
            print(f"  [{name} {index}/{len(rows)}] failed: {exc}", flush=True)
    answered = len(rows) - errors
    acc = (correct / answered) if answered else 0.0
    return {
        "name": name,
        **meta,
        "n": len(rows),
        "answered": answered,
        "errors": errors,
        "correct": correct,
        "accuracy": acc,
        "chance": meta.get("chance"),
        "delta_vs_chance": acc - float(meta.get("chance") or 0),
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "label_counts": dict(Counter(row["gold"] for row in rows)),
        "confusion": dict(confusion),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30, help="examples per text dataset")
    parser.add_argument("--image-limit", type=int, default=20, help="examples for digits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        help="subset of dataset keys",
    )
    parser.add_argument("--smoke", action="store_true", help="one example, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(args.device)

    if args.smoke:
        rows, meta = load_snappfood(1, args.seed)
        print(json.dumps(evaluate_dataset(model, "smoke", rows, meta), ensure_ascii=False, indent=2))
        return

    reports = []
    for name in args.datasets:
        loader = DATASETS[name]
        limit = args.image_limit if name == "persian_digits" else args.limit
        print(f"\n=== {name} (n={limit}) ===", flush=True)
        try:
            rows, meta = loader(limit, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {name}: {exc}", flush=True)
            reports.append({"name": name, "load_error": str(exc)})
            continue
        reports.append(evaluate_dataset(model, name, rows, meta))
        (RESULTS_DIR / "partial.json").write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    answered_reports = [r for r in reports if "accuracy" in r]
    macro = (
        sum(r["accuracy"] for r in answered_reports) / len(answered_reports)
        if answered_reports
        else None
    )
    summary = {
        "model": "vllm-sr/d3-edge",
        "device": args.device,
        "limit": args.limit,
        "image_limit": args.image_limit,
        "seed": args.seed,
        "macro_accuracy": macro,
        "datasets": reports,
    }
    out = RESULTS_DIR / "d3_edge_persian_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)
    print(f"Macro accuracy: {macro}", flush=True)
    for report in reports:
        if "accuracy" in report:
            print(
                f"  {report['name']}: {report['accuracy']:.3f} "
                f"({report['correct']}/{report['answered']}) "
                f"chance={report.get('chance'):.3f}",
                flush=True,
            )
        else:
            print(f"  {report['name']}: LOAD FAIL {report.get('load_error')}", flush=True)


if __name__ == "__main__":
    main()
