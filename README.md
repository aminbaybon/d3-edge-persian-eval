# ارزیابی صفر-شات `d3-edge` روی طبقه‌بندی فارسی

مدل [vllm-sr/d3-edge](https://huggingface.co/vllm-sr/d3-edge) (۰٫۶B، Decision 3.0) روی **۹ بنچ‌مارک طبقه‌بندی فارسی** با `system_one` و سؤال‌های `choice` تست شد. فاین‌تیون نشده است.

| | |
|---|---|
| **Macro accuracy** | **55.0%** (۹ دیتاست) |
| نمونه در هر دیتاست | ۲۰، نمونه‌گیری طبقه‌ای، seed=42 |
| دستگاه | CPU (بدون GPU) |
| تاریخ | ۱۵ اکتبر ۲۰۲۶ |

## نتایج

| دیتاست | هاب | کار | درست | Accuracy | Chance | Δ |
|---|---|---|---|---|---|---|
| DeepSentiPers | [mteb/DeepSentiPers](https://huggingface.co/datasets/mteb/DeepSentiPers) | احساس ۳کلاسه | ۱۶/۲۰ | **80.0%** | 33.3% | +46.7 |
| Persian-Text-Sentiment | [SeyedAli/Persian-Text-Sentiment](https://huggingface.co/datasets/SeyedAli/Persian-Text-Sentiment) | احساس دودویی | ۱۵/۲۰ | **75.0%** | 50.0% | +25.0 |
| Snappfood | [ParsiAI/snappfood-sentiment-analysis](https://huggingface.co/datasets/ParsiAI/snappfood-sentiment-analysis) | احساس غذا | ۱۴/۲۰ | **70.0%** | 50.0% | +20.0 |
| ParsiNLU paraphrase | [persiannlp/parsinlu_query_paraphrasing](https://huggingface.co/datasets/persiannlp/parsinlu_query_paraphrasing) | بازنویسی پرسش | ۱۴/۲۰ | **70.0%** | 50.0% | +20.0 |
| FarsTail | [dml-qom/FarsTail](https://github.com/dml-qom/FarsTail) | NLI سه‌کلاسه | ۱۱/۲۰ | **55.0%** | 33.3% | +21.7 |
| ParsiNLU entailment | [persiannlp/parsinlu_entailment](https://huggingface.co/datasets/persiannlp/parsinlu_entailment) | NLI سه‌کلاسه | ۱۰/۲۰ | **50.0%** | 33.3% | +16.7 |
| ParsiNLU MCQA | [PartAI/ParsiNLU-multiple-choice](https://huggingface.co/datasets/PartAI/ParsiNLU-multiple-choice) | چهارگزینه‌ای | ۷/۲۰ | **35.0%** | 25.0% | +10.0 |
| ParsiNLU aspect sentiment | [persiannlp/parsinlu_sentiment](https://huggingface.co/datasets/persiannlp/parsinlu_sentiment) | احساس جنبه | ۶/۲۰ | **30.0%** | 25.0% | +5.0 |
| SID classification | [MCINext/sid-classification](https://huggingface.co/datasets/MCINext/sid-classification) | موضوع ۸کلاسه (FaMTEB) | ۶/۲۰ | **30.0%** | 12.5% | +17.5 |

**جمع:** ۱۸۰ نمونه متنی. احساس و paraphrase بالاتر از شانس‌اند. NLI متوسط است. موضوع علمی و چندگزینه‌ای نزدیک شانس می‌مانند.

دو دیتاست جا ماندند: ارقام دست‌نویس فارسی (`Mehdinmz/persian-handwritten-digits`) به‌خاطر rate-limit هاب، و اخبار `HooshvareLab/pn_summary` به‌خاطر اسکریپت خراب بارگذاری. به‌جای اخبار از `SeyedAli/Persian-Text-Sentiment` استفاده شد.

---

## مثال درست و غلط هر دیتاست

لیبل‌های NLI: `e` = entailment، `c` = contradiction، `n` = neutral. در paraphrase و احساس دودویی: `1` = مثبت / هم‌معنا، `0` = منفی / غیرهم‌معنا.

### 1. Snappfood — 70%

**درست** — gold `SAD` → pred `SAD`

> بسیار آش بد بود آبکی و قسمتهای مربوط به گوشت و … تیکه تیکه بهم چسبیده بود. بدترین آش شله قلمکاری بود که توی عمرم خوردم.

**غلط** — gold `SAD` → pred `HAPPY`

> طعم و حجم غذا خوب بود و به موقع ارسال شد و تا اون موقع خیلی راضی بودم اما چند ساعت بعد از خوردن غذا دلپیچه و دل درد گرفتم هنوز حالم بده،

نیمهٔ اول نظر مثبت است؛ مدل بخش دوم (دل‌درد) را از دست داد.

### 2. DeepSentiPers — 80%

**درست** — gold `positive` → pred `positive`

> واقعا بسيار عالي .

**غلط** — gold `positive` → pred `neutral`

> تا اينجاي کار به نظرم نسبت به قيمتش مي‌ارزه.

نظر مثبت ملایم با مدل به‌عنوان خنثی خوانده شد.

### 3. ParsiNLU aspect sentiment — 30%

**درست** — gold `positive` → pred `positive`

> نظر: مربای پرتقال شانا، طعم خاص خودشو داره، … بسیار دوست داشتنی و خوش‌مزه بود … جنبه: طعم

**غلط** — gold `negative` → pred `none`

> نظر: به بزرگی که تو تصویر دیده میشه نیس و کلا ۶۰گرم پودر قهوه که بوی سوختگی هم میده توش هست جنبه: طعم

شکایت از بو و طعم هست، ولی مدل «بدون احساس» زد. لیبل `none` در این دیتاست زیاد است و مدل به آن سوگیری دارد.

### 4. FarsTail — 55%

**درست** — gold `c` → pred `c`

> مقدمه: بیش از ۹۳٪ جمعیت خاورمیانه و شمال آفریقا مسلمان‌اند. فرضیه: این جمعیت حدود ۸۲٪ است.

**غلط** — gold `c` → pred `n`

> مقدمه: عراق به مهرآباد، تبریز، همدان، دزفول، بوشهر، شیراز، اصفهان، اهواز، کرمانشاه، سنندج و ارومیه حمله کرد. فرضیه: به مشهد، بندرعباس، چابهار و یزد حمله کرد.

فرضیه شهرهای دیگری می‌گوید (تضاد)، مدل آن را خنثی دید.

### 5. ParsiNLU entailment — 50%

**درست** — gold `n` → pred `n`

> جمله ۲ ربط روشنی به جزئیات صحنهٔ باله در جمله ۱ ندارد (خنثی).

**غلط** — gold `c` → pred `n`

> جمله ۱: تکنیکال‌ها با تیربار و موشک مسلح می‌شوند. جمله ۲: مسلح کردن تکنیکال‌ها غیرممکن است.

جمله ۲ صریحاً جمله ۱ را رد می‌کند؛ مدل خنثی گفت.

### 6. ParsiNLU paraphrase — 70%

**درست** — gold `1` → pred `1`

> پرسش ۱: فردا در چه کشور هایی عید فطر است؟ پرسش ۲: در چه کشورهای فردا عید فطر است؟

**غلط** — gold `1` → pred `0`

> پرسش ۱: آهنگ تو از کجا پیدات شد که غصه هامو بردی؟ پرسش ۲: دانلود آهنگ تو از کجا پیدات شد گلاب؟

هر دو همان ترانه‌اند؛ مدل آن‌ها را غیرهم‌معنا گرفت.

### 7. SID classification — 30%

**درست** — gold `کشاورزی و منابع طبیعی` → pred `کشاورزی و منابع طبیعی`

> نقش ترویج منابع طبیعی در احیاء و توسعه اماکن ماهی گیری ورزشی … رودخانه و دریاچه سد لار

**غلط** — gold `علوم انسانی` → pred `علمی تخصصی`

> ارایه مدل تحلیل پوششی داده ها … در تحلیل صورت های مالی شرکت های بورس

مقاله مالی/مدیریت است. مدل اغلب به برچسب مبهم «علمی تخصصی» می‌رود.

### 8. ParsiNLU MCQA — 35%

**درست** — gold `1` (Printer) → pred `1`

> ........ از وسایل خروجی یک سیستم کامپیوتری می باشد.
>
> 0 Keyboard · **1 Printer** · 2 Scanner · 3 Mouse

**غلط** — gold `0` → pred `1`

> «سرمایه فرهنگی» به معنای : ..................................
>
> **0** ارزش‌های فرهنگی سامان‌دهنده به انسجام و هویت جمعی است.
> 1 سرمایه حاصل از فعالیت‌های فرهنگی … است.
> 2 سنت‌ها و ذخائر فرهنگی …
> 3 سرمایه‌گذاری در عرصه فرهنگی است.

### 9. Persian-Text-Sentiment — 75%

**درست** — gold `1` (مثبت) → pred `1`

> اش خیلی خیلی شور بود - نون هنبرگر هم خیلی خشک بود ولی در کل خوب بودن منهای اش

**غلط** — gold `1` (مثبت) → pred `0`

> سوپ چون تند بود مناسب بیمار نبود باید نوع سوپ داشته باشن تا همه بتونن استفاده کنن

نظر سازنده است و در دیتاست مثبت برچسب خورده؛ مدل روی «تند / مناسب نبود» منفی زد.

---

## پروتکل

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

- Transformers 5.17، وزن رسمی هاب، `verify=none`
- دستورالعمل‌ها فارسی‌اند
- متن‌ها تا حدود ۱۲۰۰ نویسه بریده شدند
- ParsiNLU sentiment به ۴ کلاس جمع شد: none / negative / neutral / positive
- DeepSentiPers نسخهٔ MTEB سه‌کلاسه است (positive / negative / neutral)

اجرای مجدد:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-eval.txt
.venv\Scripts\python eval_d3_persian.py --device cpu --limit 20
```

## محدودیت‌ها

این یک اسموک‌تست است نه لیدربورد کامل: n=20، فقط CPU، بدون few-shot. عددها با تغییر seed یا دستورالعمل جابه‌جا می‌شوند. فایل خلاصه: [`results/d3_edge_persian_eval_summary.json`](results/d3_edge_persian_eval_summary.json).

## استناد

```bibtex
@misc{d3_edge_2026,
  title        = {{d3-edge}: A Multimodal Foundation Decision Model},
  author       = {{vLLM Semantic Router Team}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/vllm-sr/d3-edge}}
}
```
