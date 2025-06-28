import socketio
from aiohttp import web
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Tạo server Socket.IO
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

# Load spam classification model
spam_model = AutoModelForSequenceClassification.from_pretrained("Khoa/kompa-spam-filter-telecomunication-internet-update-0625", num_labels=2)
spam_tokenizer = AutoTokenizer.from_pretrained("Khoa/kompa-spam-filter-telecomunication-internet-update-0625", use_fast=False)
spam_classifier = pipeline(
    "text-classification",
    model=spam_model,
    tokenizer=spam_tokenizer,
    device=-1,
    return_all_scores=True
)

# Load language detection model
lang_classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=-1
)

# Ánh xạ nhãn spam cho dễ hiểu
LABEL_MAPPING = {
    "LABEL_0": "non-spam",
    "LABEL_1": "spam"
}

async def predict_spam_and_language(text):
    # Dự đoán spam
    spam_result = spam_classifier(text)
    spam_label = max(spam_result[0], key=lambda x: x['score'])['label']
    spam_score = max(spam_result[0], key=lambda x: x['score'])['score']
    
    # Dự đoán ngôn ngữ
    lang_result = lang_classifier(text, top_k=1)
    language = lang_result[0]['label']
    lang_score = lang_result[0]['score']
    
    return {
        "spam_label": LABEL_MAPPING.get(spam_label, spam_label),
        "spam_score": spam_score,
        "language": language,
        "language_score": lang_score
    }

@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")

@sio.event
async def predict(sid, data):
    text = data.get('text', '')
    if not text:
        await sio.emit('result', {'error': 'No text provided'}, to=sid)
        return
    
    try:
        result = await predict_spam_and_language(text)
        await sio.emit('result', result, to=sid)
    except Exception as e:
        await sio.emit('result', {'error': str(e)}, to=sid)

if __name__ == '__main__':
    web.run_app(app, port=5000)