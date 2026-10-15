# Zero-shot Persian classification with `d3-edge`

[vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) (0.6B, Decision 3.0) was evaluated on **9 Persian classification benchmarks** with `system_one` choice questions. No fine-tuning.

| | |
|---|---|
| **Macro accuracy** | **55.0%** (9 datasets) |
| Examples per dataset | 20, stratified sample, seed=42 |
| Device | CPU (no GPU) |
| Date | 15 October 2026 |

## Results

| Dataset | Hub | Task | Correct | Accuracy | Chance | Δ |
|---|---|---|---|---|---|---|
| DeepSentiPers | [mteb/DeepSentiPers](https://huggingface.co/datasets/mteb/DeepSentiPers) | 3-class sentiment | 16/20 | **80.0%** | 33.3% | +46.7 |
| Persian-Text-Sentiment | [SeyedAli/Persian-Text-Sentiment](https://huggingface.co/datasets/SeyedAli/Persian-Text-Sentiment) | binary sentiment | 15/20 | **75.0%** | 50.0% | +25.0 |
| Snappfood | [ParsiAI/snappfood-sentiment-analysis](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis) | food-review sentiment | 14/20 | **70.0%** | 50.0% | +20.0 |
| ParsiNLU paraphrase | [persiannlp/parsinlu_query_paraphrasing](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing) | query paraphrase | 14/20 | **70.0%** | 50.0% | +20.0 |
| FarsTail | [dml-qom/FarsTail](https://github.com/dml-qom/FarsTail) | 3-way NLI | 11/20 | **55.0%** | 33.3% | +21.7 |
| ParsiNLU entailment | [persiannlp/parsinlu_entailment](https://huggingface.co/datasets/persiannlp/parsinlu_entailment) | 3-way NLI | 10/20 | **50.0%** | 33.3% | +16.7 |
| ParsiNLU MCQA | [PartAI/ParsiNLU-multiple-choice](https://huggingface.co/datasets/PartAI/ParsiNLU-multiple-choice) | 4-way QA | 7/20 | **35.0%** | 25.0% | +10.0 |
| ParsiNLU aspect sentiment | [persiannlp/parsinlu_sentiment](https://huggingface.co/datasets/persiannlp/parsinlu_sentiment) | aspect sentiment | 6/20 | **30.0%** | 25.0% | +5.0 |
| SID classification | [MCINext/sid-classification](https://huggingface.co/datasets/MCINext/sid-classification) | 8-way topic (FaMTEB) | 6/20 | **30.0%** | 12.5% | +17.5 |

**Total:** 180 text examples. Sentiment and paraphrase sit well above chance. NLI is middling. Academic topic classification and multiple-choice QA stay close to chance.

Two datasets were dropped: Persian handwritten digits (`Mehdinmz/persian-handwritten-digits`) after Hugging Face rate limits, and `HooshvareLab/pn_summary` because its loading script is broken. `SeyedAli/Persian-Text-Sentiment` replaced the news set.

---

## Correct and incorrect example per dataset

NLI labels: `e` = entailment, `c` = contradiction, `n` = neutral. For paraphrase and binary sentiment: `1` = positive / paraphrase, `0` = negative / not paraphrase. Source texts stay in Persian.

### 1. Snappfood — 70%

**Correct** — gold `SAD` → pred `SAD`

> بسیار آش بد بود آبکی و قسمتهای مربوط به گوشت و … تیکه تیکه بهم چسبیده بود. بدترین آش شله قلمکاری بود که توی عمرم خوردم.

**Incorrect** — gold `SAD` → pred `HAPPY`

> طعم و حجم غذا خوب بود و به موقع ارسال شد و تا اون موقع خیلی راضی بودم اما چند ساعت بعد از خوردن غذا دلپیچه و دل درد گرفتم هنوز حالم بده،

The first half is positive; the model missed the later stomachache.

### 2. DeepSentiPers — 80%

**Correct** — gold `positive` → pred `positive`

> واقعا بسيار عالي .

**Incorrect** — gold `positive` → pred `neutral`

> تا اينجاي کار به نظرم نسبت به قيمتش مي‌ارزه.

A mild positive (“worth the price”) was read as neutral.

### 3. ParsiNLU aspect sentiment — 30%

**Correct** — gold `positive` → pred `positive`

> نظر: مربای پرتقال شانا، طعم خاص خودشو داره، … بسیار دوست داشتنی و خوش‌مزه بود … جنبه: طعم

**Incorrect** — gold `negative` → pred `none`

> نظر: به بزرگی که تو تصویر دیده میشه نیس و کلا ۶۰گرم پودر قهوه که بوی سوختگی هم میده توش هست جنبه: طعم

The review complains about burnt smell and taste, but the model chose “no sentiment”. The `none` class is frequent here and the model leans toward it.

### 4. FarsTail — 55%

**Correct** — gold `c` → pred `c`

> Premise: more than 93% of the Middle East and North Africa population are Muslim. Hypothesis: that share is about 82%.

**Incorrect** — gold `c` → pred `n`

> Premise: Iraq attacked Mehrabad, Tabriz, Hamedan, Dezful, Bushehr, Shiraz, Isfahan, Ahvaz, Kermanshah, Sanandaj, and Urmia. Hypothesis: it attacked Mashhad, Bandar Abbas, Chabahar, and Yazd.

The hypothesis names different cities (contradiction); the model called it neutral.

### 5. ParsiNLU entailment — 50%

**Correct** — gold `n` → pred `n`

> Sentence 2 has no clear link to the ballet-scene details in sentence 1 (neutral).

**Incorrect** — gold `c` → pred `n`

> Sentence 1: technicals are armed with machine guns and missiles. Sentence 2: arming technicals is impossible.

Sentence 2 directly denies sentence 1; the model said neutral.

### 6. ParsiNLU paraphrase — 70%

**Correct** — gold `1` → pred `1`

> Q1: فردا در چه کشور هایی عید فطر است؟ Q2: در چه کشورهای فردا عید فطر است؟

**Incorrect** — gold `1` → pred `0`

> Q1: آهنگ تو از کجا پیدات شد که غصه هامو بردی؟ Q2: دانلود آهنگ تو از کجا پیدات شد گلاب؟

Both queries refer to the same song; the model treated them as unrelated.

### 7. SID classification — 30%

**Correct** — gold `کشاورزی و منابع طبیعی` → pred `کشاورزی و منابع طبیعی`

> نقش ترویج منابع طبیعی در احیاء و توسعه اماکن ماهی گیری ورزشی … رودخانه و دریاچه سد لار

**Incorrect** — gold `علوم انسانی` → pred `علمی تخصصی`

> ارایه مدل تحلیل پوششی داده ها … در تحلیل صورت های مالی شرکت های بورس

The paper is finance / management. The model often collapses into the vague “specialized science” bucket.

### 8. ParsiNLU MCQA — 35%

**Correct** — gold `1` (Printer) → pred `1`

> ........ از وسایل خروجی یک سیستم کامپیوتری می باشد.
>
> 0 Keyboard · **1 Printer** · 2 Scanner · 3 Mouse

**Incorrect** — gold `0` → pred `1`

> «سرمایه فرهنگی» به معنای : ..................................
>
> **0** Cultural values that organize collective cohesion and identity.
> 1 Capital from cultural production, distribution, and consumption.
> 2 Cultural and artistic traditions and reserves in a social system.
> 3 Investment in the cultural sphere.

### 9. Persian-Text-Sentiment — 75%

**Correct** — gold `1` (positive) → pred `1`

> اش خیلی خیلی شور بود - نون هنبرگر هم خیلی خشک بود ولی در کل خوب بودن منهای اش

**Incorrect** — gold `1` (positive) → pred `0`

> سوپ چون تند بود مناسب بیمار نبود باید نوع سوپ داشته باشن تا همه بتونن استفاده کنن

The gold label is positive (constructive feedback). The model keyed on “too spicy / not suitable” and predicted negative.

---

## Protocol

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("vllm-sr/d3-edge", trust_remote_code=True, device="cpu")
result = model.system_one(
    state=text,
    questions={
        "label": {
            "type": "choice",
            "instructions": "...",
            "criteria": {"class_a": "...", "class_b": "..."},
        }
    },
)
# result["answers"]["label"]["choice"]
```

- Transformers 5.17, official Hub weights, `verify=none`
- Instructions are in Persian
- Inputs truncated around 1,200 characters
- ParsiNLU sentiment collapsed to four classes: none / negative / neutral / positive
- DeepSentiPers uses the MTEB 3-class mapping (positive / negative / neutral)

Reproduce:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-eval.txt
.venv\Scripts\python eval_d3_persian.py --device cpu --limit 20
```

## Limitations

This is a smoke test, not a full leaderboard: n=20, CPU only, no few-shot. Scores will move if the seed or the question wording changes. Summary JSON: [`results/d3_edge_persian_eval_summary.json`](results/d3_edge_persian_eval_summary.json).

## Citation

```bibtex
@misc{d3_edge_2026,
  title        = {{d3-edge}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-edge}}
}
```
