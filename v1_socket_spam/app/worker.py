import redis
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def truncate_text(text, tokenizer, max_tokens=256):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

# Redis config (fix host)
redis_conn = redis.Redis(
    host="redis",  # tên service trong docker-compose
    port=6379,
    db=0,
    decode_responses=True
)
REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

LABEL_MAPPING = {"LABEL_0": "non-spam", "LABEL_1": "spam"}

# Mapping mô hình theo category
CATEGORY_MODEL_MAP = {
    "telecom": "Khoa/kompa-spam-filter-telecomunication-internet-update-0625",
    "finance": "Khoa/kompa-spam-filter-finance-update-0525",
    "healthcare": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525",
    "e-commerce": "Khoa/kompa-spam-filter-e-commerce-update-0625",
}

# Model cache
class ModelRegistry:
    def __init__(self):
        self.models = {}

    def get_or_load_model(self, category):
        if category not in self.models:
            if category not in CATEGORY_MODEL_MAP:
                raise ValueError(f"No model configured for category: {category}")
            print(f"🔧 Loading model for category: {category}")
            model_path = CATEGORY_MODEL_MAP[category]
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=-1,
                return_all_scores=True
            )
            self.models[category] = (classifier, tokenizer)
        return self.models[category]

# Registry instance
registry = ModelRegistry()

# Language classifier (chung)
lang_classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=-1
)

# Vòng lặp worker
while True:
    try:
        packed = redis_conn.blpop(REDIS_REQUEST_QUEUE, timeout=5)
        if not packed:
            continue

        _, payload = packed
        task = json.loads(payload)
        text = task["text"]
        meta = task["meta"]
        category = meta.get("category", "")

        try:
            classifier, tokenizer = registry.get_or_load_model(category)
            text = truncate_text(text, tokenizer)

            spam_result = classifier(text)[0]
            spam_label = max(spam_result, key=lambda x: x['score'])['label']
            spam_score = max(spam_result, key=lambda x: x['score'])['score']

            lang_result = lang_classifier(text, top_k=1)[0]
            lang_label = lang_result['label']
            lang_score = lang_result['score']

            result = {
                "spam_label": LABEL_MAPPING.get(spam_label, spam_label),
                "spam_score": spam_score,
                "language": lang_label,
                "language_score": lang_score
            }

        except Exception as e:
            result = {"error": str(e)}

        redis_conn.rpush(REDIS_RESULT_QUEUE, json.dumps({
            "job_id": task["job_id"],
            "result": result
        }))

    except Exception as e:
        print(f"❌ Worker error: {e}")
