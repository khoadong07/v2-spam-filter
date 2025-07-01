import json
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAPPING = {"LABEL_0": "spam", "LABEL_1": "non-spam"}

CATEGORY_MODEL_MAP = {
    "finance": "Khoa/kompa-spam-filter-finance-update-0525",
    "real_estate": "Khoa/kompa-spam-filter-real-estate-update-0525",
    "ewallet": "Khoa/kompa-spam-filter-e-wallet-update-0625",
    "healthcare_insurance": "Khoa/kompa-spam-filter-healthcare-insurance-update-0525",
    "ecommerce": "Khoa/kompa-spam-filter-e-commerce-update-0625",
}

def truncate_text(text, tokenizer, max_tokens=254):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_ids = tokenizer.convert_tokens_to_ids(tokens[:max_tokens])
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)

# Language classifier
lang_classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=-1
)

# Cache models
_model_cache = {}

def load_model(category):
    if category in _model_cache:
        return _model_cache[category]
    if category not in CATEGORY_MODEL_MAP:
        raise ValueError(f"❌ Không có mô hình cho category: {category}")
    
    model_path = CATEGORY_MODEL_MAP[category]
    print(f"🔧 Loading model for category '{category}' from {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        return_all_scores=True
    )
    _model_cache[category] = (classifier, tokenizer)
    return classifier, tokenizer

# Hàm gọi inference đơn lẻ
def run_inference(text, category):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("⚠️ Text đầu vào không hợp lệ")

    classifier, tokenizer = load_model(category)
    text = truncate_text(text, tokenizer)

    spam_result = classifier(text)[0]
    spam_result = [r for r in spam_result if r["label"] in LABEL_MAPPING]
    if not spam_result:
        raise ValueError("⚠️ Kết quả không hợp lệ từ mô hình.")

    top_spam = max(spam_result, key=lambda x: x['score'])
    spam_label = top_spam['label']
    spam_score = top_spam['score']

    lang_result = lang_classifier(text, top_k=1)[0]
    lang_label = lang_result['label']
    lang_score = lang_result['score']

    return {
        "spam_label": LABEL_MAPPING.get(spam_label, spam_label),
        "spam_score": spam_score,
        "language": lang_label,
        "language_score": lang_score
    }

# ===== Test demo =====
if __name__ == "__main__":
    test_text = "Màn hình Gaming Galax Vivance 32Q (VI-32Q) 32 inch 2K QHD IPS 165Hz 1ms👉Màn hình 2k đẹp nét đến từng giây luôn >< 🎁 𝐓𝐚̣̆𝐧𝐠 𝐧𝐠𝐚𝐲 𝐜𝐡𝐮𝐨̣̂𝐭 𝐆𝐀𝐌𝐈𝐍𝐆 𝐭𝐫𝐢̣ 𝐠𝐢𝐚́ 𝟐𝐭𝐫 đ𝐨̂̀𝐧𝐠 🎁================== ⚡ Hổ Trợ Trả góp HD Saison & Home Credit⚡ Quẹt Thẻ Tín Dụng Mpos🚛 Ship cod toàn quốc: Nhận hàng kiểm tra hàng và thanh toán tiền cho nhân viên giao hàng🏡 Hữu Tài Computer PC Hi-End - Gaming Gear📌 Số 33/10 đường Phạm Thái Bường,. P4. Thành Phố Vĩnh Long. (đối diện cổng trương tiểu học Trần Đại Nghĩa)📌 CN2 số 44B Đinh Tiên Hoàng P8 (cách bến xe mới 300M hướng đi Cần Thơ) ☎ 0939.182727 - 0939.792727  (zalo để được tư vấn nhanh nhất ạ)🌍 https://maytinhvinhlong.vn/ "
    category = "finance"

    result = run_inference(test_text, category)
    print("✅ Inference result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
