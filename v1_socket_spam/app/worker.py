import asyncio
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import aioredis

# Redis config
REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

# Model mapping theo category
CATEGORY_MODEL_MAP = {
    "healthcare_insurance": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525",
    "energy_fuels": "Khoa/kompa-spam-filter-energy-fuels-update-0625",
    "electronic": "Khoa/kompa-spam-filter-electronic-update-0625",
    "fmcg": "Khoa/kompa-spam-filter-fmcg-update-0625",
    "fnb": "Khoa/kompa-spam-filter-fnb-update-0625",
    "logistic_delivery": "Khoa/kompa-spam-filter-logistics-delivery-update-0625",
    "bank": "Khoa/kompa-spam-filter-bank-update-0625",
    "finance": "Khoa/kompa-spam-filter-finance-update-0525",
    "ewallet": "Khoa/kompa-spam-filter-e-wallet-update-0625",
    "investment": "Khoa/kompa-spam-filter-investment-update-0625",
    "real_estate": "Khoa/kompa-spam-filter-real-estate-update-0525",
    "education": "Khoa/kompa-spam-filter-education-update-0625",
}

def truncate_text(text, tokenizer, max_tokens=256):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    token_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# Model registry cache
class ModelRegistry:
    def __init__(self):
        self.models = {}

    def get(self, category):
        if category in self.models:
            return self.models[category]
        if category not in CATEGORY_MODEL_MAP:
            raise ValueError(f"❌ No model found for category '{category}'")

        print(f"🔧 Loading spam model for category '{category}'")
        model_path = CATEGORY_MODEL_MAP[category]
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0,
            return_all_scores=True
        )
        self.models[category] = (classifier, tokenizer)
        return classifier, tokenizer

# Init model cache và language detector
registry = ModelRegistry()

lang_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
lang_tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection", use_fast=False)
lang_classifier = pipeline(
    "text-classification",
    model=lang_model,
    tokenizer=lang_tokenizer,
    device=-1,
    top_k=1
)

def predict_spam_and_language(text, category):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("⚠️ Invalid input text")

    classifier, tokenizer = registry.get(category)
    text = truncate_text(text, tokenizer)

    spam_result = classifier(text)
    if not spam_result or not spam_result[0]:
        raise ValueError("⚠️ spam_result is empty")
    top_spam = max(spam_result[0], key=lambda x: x['score'])
    spam_label = top_spam['label'] == 'LABEL_0'

    # lang_result = lang_classifier(text, top_k=1)
    # if isinstance(lang_result[0], dict):
    #     language = lang_result[0].get("label", "unknown")
    # elif isinstance(lang_result[0], list):
    #     language = lang_result[0][0].get("label", "unknown")
    # else:
    #     language = "unknown"

    return {
        "spam": spam_label,
        "lang": None
    }

# ===== Worker loop (async) =====
async def worker_loop():
    redis = await aioredis.from_url("redis://redis:6379", decode_responses=True)

    while True:
        try:
            packed = await redis.blpop(REDIS_REQUEST_QUEUE, timeout=10)
            if not packed:
                await asyncio.sleep(0.1)
                continue

            _, payload = packed
            task = json.loads(payload)

            job_id = task.get("job_id")
            text = task.get("text", "")
            meta = task.get("meta", {})
            category = meta.get("category", "")

            print(f"📥 job_id={job_id} | category={category}")

            try:
                prediction = await asyncio.to_thread(predict_spam_and_language, text, category)

                result = {
                    "id": meta.get("id", ""),
                    "topic": meta.get("topic", ""),
                    "topic_id": meta.get("topic_id", ""),
                    "title": meta.get("title", ""),
                    "content": meta.get("content", ""),
                    "description": meta.get("description", ""),
                    "sentiment": meta.get("sentiment", ""),
                    "site_name": meta.get("site_name", ""),
                    "site_id": meta.get("site_id", ""),
                    "type": meta.get("type", ""),
                    **prediction
                }

                print(f"✅ job_id={job_id} | spam={result['spam']} | lang={result['lang']}")

            except Exception as e:
                result = {"error": str(e)}
                print(f"❌ job_id={job_id} | Lỗi xử lý: {e}")
                print(f"📝 Text: {repr(text)}")

            await redis.rpush(REDIS_RESULT_QUEUE, json.dumps({
                "job_id": job_id,
                "result": result
            }))

        except Exception as e:
            print(f"🔥 Worker loop error: {e}")
            await asyncio.sleep(1)

# ===== Main Entrypoint =====
if __name__ == "__main__":
    asyncio.run(worker_loop())
