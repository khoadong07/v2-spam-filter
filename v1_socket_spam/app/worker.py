import redis
import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Redis config
redis_conn = redis.Redis(
    host="redis",
    port=6379,
    db=0,
    decode_responses=True
)

REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

# Mapping mô hình theo category
CATEGORY_MODEL_MAP = {
    "finance": "Khoa/kompa-spam-filter-finance-update-0525",
    "real_estate": "Khoa/kompa-spam-filter-real-estate-update-0525",
    "ewallet": "Khoa/kompa-spam-filter-e-wallet-update-0625",
    "healthcare_insurance": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525",
    "ecommerce": "Khoa/kompa-spam-filter-e-commerce-update-0625",
}

# Truncate text để giới hạn số tokens
def truncate_text(text, tokenizer, max_tokens=256):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    token_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# Cache mô hình phân loại spam theo category
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
            device=0,  # CPU
            return_all_scores=True
        )
        self.models[category] = (classifier, tokenizer)
        return classifier, tokenizer

registry = ModelRegistry()

# Load language detection pipeline 1 lần
lang_model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection", from_tf=False)
lang_tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection", use_fast=False)
lang_classifier = pipeline(
    "text-classification",
    model=lang_model,
    tokenizer=lang_tokenizer,
    device=-1,
    top_k=1
)

# Hàm inference chính
def predict_spam_and_language(text, category):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("⚠️ Invalid input text")

    classifier, tokenizer = registry.get(category)
    text = truncate_text(text, tokenizer)

    # Spam classification
    spam_result = classifier(text)
    if not spam_result or not spam_result[0]:
        raise ValueError("⚠️ spam_result is empty")
    top_spam = max(spam_result[0], key=lambda x: x['score'])
    spam_label = top_spam['label'] == 'LABEL_0' 

    # Language detection
    lang_result = lang_classifier(text, top_k=1)
    if isinstance(lang_result, list) and isinstance(lang_result[0], dict):
        language = lang_result[0].get("label", "unknown")
    elif isinstance(lang_result[0], list) and isinstance(lang_result[0][0], dict):
        language = lang_result[0][0].get("label", "unknown")
    else:
        language = "unknown"

    return {
        "spam": spam_label,
        "lang": "vietnamese" if language == 'vi' else language
    }

# ===== Worker loop =====
while True:
    try:
        packed = redis_conn.blpop(REDIS_REQUEST_QUEUE, timeout=5)
        if not packed:
            continue

        _, payload = packed
        task = json.loads(payload)

        job_id = task.get("job_id")
        text = task.get("text", "")
        meta = task.get("meta", {})
        category = meta.get("category", "")

        print(f"📥 job_id={job_id} | category={category}")

        try:
            prediction = predict_spam_and_language(text, category)

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

        redis_conn.rpush(REDIS_RESULT_QUEUE, json.dumps({
            "job_id": job_id,
            "result": result
        }))

    except Exception as e:
        print(f"🔥 Worker loop error: {e}")
