import asyncio
import json
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download
import redis.asyncio as redis

# Redis config
REDIS_REQUEST_QUEUE = "spam_request_queue"
REDIS_RESULT_QUEUE = "spam_result_queue"

# Local model paths
LOCAL_MODEL_DIR = "./models"
CATEGORY_MODEL_MAP = {
    "healthcare_insurance": {"repo_id": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-healthcare-insurance-update-0525")},
    "energy_fuels": {"repo_id": "Khoa/kompa-spam-filter-energy-fuels-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-energy-fuels-update-0625")},
    "electronic": {"repo_id": "Khoa/kompa-spam-filter-electronic-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-electronic-update-0625")},
    "fmcg": {"repo_id": "Khoa/kompa-spam-filter-fmcg-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-fmcg-update-0625")},
    "fnb": {"repo_id": "Khoa/kompa-spam-filter-fnb-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-fnb-update-0625")},
    "logistic_delivery": {"repo_id": "Khoa/kompa-spam-filter-logistics-delivery-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-logistics-delivery-update-0625")},
    "bank": {"repo_id": "Khoa/kompa-spam-filter-bank-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-bank-update-0625")},
    "finance": {"repo_id": "Khoa/kompa-spam-filter-finance-update-0525", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-finance-update-0525")},
    "ewallet": {"repo_id": "Khoa/kompa-spam-filter-e-wallet-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-e-wallet-update-0625")},
    "investment": {"repo_id": "Khoa/kompa-spam-filter-investment-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-investment-update-0625")},
    "real_estate": {"repo_id": "Khoa/kompa-spam-filter-real-estate-update-0525", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-real-estate-update-0525")},
    "education": {"repo_id": "Khoa/kompa-spam-filter-education-update-0625", "local_path": os.path.join(LOCAL_MODEL_DIR, "kompa-spam-filter-education-update-0625")},
}

def truncate_text(text, tokenizer, max_tokens=256):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    token_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
    return tokenizer.decode(token_ids, skip_special_tokens=True)

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def get(self, category):
        if category in self.models:
            return self.models[category]
        if category not in CATEGORY_MODEL_MAP:
            raise ValueError(f"❌ No model found for category '{category}'")

        print(f"🔧 Loading spam model for category '{category}'")
        model_info = CATEGORY_MODEL_MAP[category]
        model_path = model_info["local_path"]

        # Tải model từ Hugging Face nếu chưa có
        if not os.path.exists(model_path):
            print(f"📥 Downloading model for category '{category}' from {model_info['repo_id']}")
            snapshot_download(repo_id=model_info["repo_id"], local_dir=model_path, local_dir_use_symlinks=False)

        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0,
            return_all_scores=True
        )
        self.models[category] = (classifier, tokenizer)
        return classifier, tokenizer

registry = ModelRegistry()

# Tải model ngôn ngữ
LANG_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "xlm-roberta-base-language-detection")
if not os.path.exists(LANG_MODEL_PATH):
    print(f"📥 Downloading language model from papluca/xlm-roberta-base-language-detection")
    snapshot_download(repo_id="papluca/xlm-roberta-base-language-detection", local_dir=LANG_MODEL_PATH, local_dir_use_symlinks=False)

lang_model = AutoModelForSequenceClassification.from_pretrained(LANG_MODEL_PATH, local_files_only=True)
lang_tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_PATH, use_fast=False, local_files_only=True)
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
    top_spam = max(spam_result[0], key=lambda x: x['score'])
    spam_label = top_spam['label'] == 'LABEL_0'

    return {
        "spam": spam_label,
        "lang": None
    }

async def handle_task(redis_conn, task, semaphore):
    async with semaphore:
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
            print(f"❌ job_id={job_id} | Error: {e}")
            print(f"📝 Text: {repr(text)}")

        await redis_conn.rpush(REDIS_RESULT_QUEUE, json.dumps({
            "job_id": job_id,
            "result": result
        }))

async def worker_loop(concurrency=5):
    redis_conn = redis.Redis(host="redis", port=6379, decode_responses=True)
    semaphore = asyncio.Semaphore(concurrency)

    while True:
        try:
            packed = await redis_conn.blpop(REDIS_REQUEST_QUEUE, timeout=10)
            if not packed:
                await asyncio.sleep(0.1)
                continue

            _, payload = packed
            task = json.loads(payload)

            asyncio.create_task(handle_task(redis_conn, task, semaphore))

        except Exception as e:
            print(f"🔥 Worker loop error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(worker_loop(concurrency=5))