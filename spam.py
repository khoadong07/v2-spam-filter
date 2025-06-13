import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, DetectorFactory
from torch.utils.data import DataLoader, Dataset
import logging
from tqdm import tqdm
from vncorenlp import VnCoreNLP
import re
import json
from typing import List, Dict, Union, Set
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration for multiple categories
class Config:
    MAX_LENGTH: int = 100
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE: int = 16
    CATEGORIES: Dict[str, Dict] = {
        "finance": {
            "MODEL_PATH": os.getenv("FINANCE", ""),
            "TOKENIZER_NAME": os.getenv("FINANCE", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/finance.json')) if os.path.exists('static/site_id_filter/finance.json') else [],
        },
        "real_estate": {
            "MODEL_PATH": os.getenv("REAL_ESTATE", ""),
            "TOKENIZER_NAME": os.getenv("REAL_ESTATE", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/real_estate.json')) if os.path.exists('static/site_id_filter/real_estate.json') else [],
        },
        "ewallet": {
            "MODEL_PATH": os.getenv("EWALLET", ""),
            "TOKENIZER_NAME": os.getenv("EWALLET", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/ewallet.json')) if os.path.exists('static/site_id_filter/ewallet.json') else [],
        },
        "healthcare_insurance": {
            "MODEL_PATH": os.getenv("HEALTHCARE_INSURANCE", ""),
            "TOKENIZER_NAME": os.getenv("HEALTHCARE_INSURANCE", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/healthcare_insurance.json')) if os.path.exists('static/site_id_filter/healthcare_insurance.json') else [],
        },
        "ecommerce": {
            "MODEL_PATH": os.getenv("ECOMMERCE", ""),
            "TOKENIZER_NAME": os.getenv("ECOMMERCE", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/ecommerce.json')) if os.path.exists('static/site_id_filter/ecommerce.json') else [],
        },
        "education": {
            "MODEL_PATH": os.getenv("EDUCATION", ""),
            "TOKENIZER_NAME": os.getenv("EDUCATION", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/education.json')) if os.path.exists('static/site_id_filter/education.json') else [],
        },
        "logistic_delivery": {
            "MODEL_PATH": os.getenv("LOGISTIC_DELIVERY", ""),
            "TOKENIZER_NAME": os.getenv("LOGISTIC_DELIVERY", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/logistic_delivery.json')) if os.path.exists('static/site_id_filter/logistic_delivery.json') else [],
        },
        "energy_fuels": {
            "MODEL_PATH": os.getenv("ENERGY_FUELS", ""),
            "TOKENIZER_NAME": os.getenv("ENERGY_FUELS", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/energy_fuels.json')) if os.path.exists('static/site_id_filter/energy_fuels.json') else [],
        },
        "fnb": {
            "MODEL_PATH": os.getenv("FNB", ""),
            "TOKENIZER_NAME": os.getenv("FNB", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/fnb.json')) if os.path.exists('static/site_id_filter/fnb.json') else [],
        },
        "investment": {
            "MODEL_PATH": os.getenv("INVESTMENT", ""),
            "TOKENIZER_NAME": os.getenv("INVESTMENT", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/investment.json')) if os.path.exists('static/site_id_filter/investment.json') else [],
        },
        "fmcg": {
            "MODEL_PATH": os.getenv("FMCG", ""),
            "TOKENIZER_NAME": os.getenv("FMCG", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/fmcg.json')) if os.path.exists('static/site_id_filter/fmcg.json') else [],
        },
        "retail": {
            "MODEL_PATH": os.getenv("RETAIL", ""),
            "TOKENIZER_NAME": os.getenv("RETAIL", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/retail.json')) if os.path.exists('static/site_id_filter/retail.json') else [],
        },
        "technology_motorbike_food": {
            "MODEL_PATH": os.getenv("TECHNOLOGY_MOTORBIKE_FOOD", ""),
            "TOKENIZER_NAME": os.getenv("TECHNOLOGY_MOTORBIKE_FOOD", ""),
            "STOPWORDS_PATH": os.getenv("STOPWORDS_PATH", ""),
            "FILTER_SITE_IDS": json.load(open('static/site_id_filter/technology_motorbike_food.json')) if os.path.exists('static/site_id_filter/technology_motorbike_food.json') else [],
        }

    }
    TRUSTED_SITES: List[str] = ['google.com', 'play.google.com', 'apps.apple.com']

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0

# Initialize VnCoreNLP
vncorenlp = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

class TextPreprocessor:
    """Handles text preprocessing tasks like stopword removal, emoji removal, and tokenization."""
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """Remove emojis from text using regex."""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    @staticmethod
    def remove_stopwords(text: str, stopwords: Set[str]) -> str:
        """Remove stopwords from text."""
        return ' '.join(word for word in text.split() if word not in stopwords)

    @staticmethod
    def tokenize_vietnamese(text: str) -> str:
        """Tokenize Vietnamese text using VnCoreNLP."""
        return " ".join([" ".join(sentence) for sentence in vncorenlp.tokenize(text)])

    @classmethod
    def preprocess(cls, text: str, stopwords: Set[str], tokenized: bool = True, lowercased: bool = True) -> str:
        """Preprocess text with stopword removal, emoji removal, and optional tokenization."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = cls.remove_stopwords(text, stopwords)
        text = cls.remove_emojis(text)
        text = text.lower() if lowercased else text
        if tokenized:
            text = cls.tokenize_vietnamese(text)
        return text

    @classmethod
    def preprocess_with_language_detection(cls, text: str, stopwords: Set[str], tokenized: bool = True, lowercased: bool = True) -> tuple[str, str]:
        """Preprocess text and detect language."""
        if not isinstance(text, str) or not text.strip():
            return "", "unknown"
        
        try:
            language = detect(text)
        except:
            language = "unknown"

        if language == 'vi':
            text = cls.preprocess(text, stopwords, tokenized=tokenized, lowercased=lowercased)
        else:
            text = cls.remove_emojis(text)
            text = text.lower() if lowercased else text
        return text, language

class SpamClassifierModel:
    """Manages the spam classification model and predictions for a specific category."""
    
    def __init__(self, category: str):
        """Initialize model and tokenizer for the given category."""
        if category not in Config.CATEGORIES:
            raise ValueError(f"Category {category} not found in configuration.")
        
        self.category = category
        self.config = Config.CATEGORIES[category]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["TOKENIZER_NAME"], use_fast=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config["MODEL_PATH"]).to(Config.DEVICE)
            self.model.eval()
            with open(self.config["STOPWORDS_PATH"], "r", encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f)
            logger.info(f"Model, tokenizer, and stopwords for category {category} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model, tokenizer, or stopwords for category {category}: {e}")
            raise

    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """Predict spam probability for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]

        processed_texts, languages = [], []
        for text in tqdm(texts, desc="Preprocessing texts", disable=True):
            processed_text, lang = TextPreprocessor.preprocess_with_language_detection(text, self.stopwords)
            processed_texts.append(processed_text)
            languages.append(lang)

        results = []
        for text, lang in zip(processed_texts, languages):
            if lang != 'vi':
                results.append({
                    "label": "spam",
                    "probability": 1.0,
                    "all_probabilities": {"not_spam": 0.0, "spam": 1.0},
                    "language": lang
                })

        predict_texts = [text for text, lang in zip(processed_texts, languages) if lang == 'vi']
        predict_indices = [i for i, lang in enumerate(languages) if lang == 'vi']

        if not predict_texts:
            return results

        dataset = TextDataset(predict_texts)
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        labels = ["not_spam", "spam"]

        predict_results = []
        with torch.no_grad():
            for batch_texts in dataloader:
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=Config.MAX_LENGTH,
                    return_tensors="pt"
                ).to(Config.DEVICE)

                outputs = self.model(**encodings)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                for pred, prob in zip(predictions, probabilities):
                    predict_results.append({
                        "label": labels[pred.item()],
                        "probability": prob[pred.item()].item(),
                        "all_probabilities": {
                            labels[0]: prob[0].item(),
                            labels[1]: prob[1].item()
                        },
                        "language": "vi"
                    })

        for idx, result in zip(predict_indices, predict_results):
            results.insert(idx, result)

        return results

class TextDataset(Dataset):
    """Dataset class for text data."""
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

class SpamClassifier:
    """Main class for spam classification from JSON data with multi-category support."""
    
    def __init__(self, category: str):
        self.model = SpamClassifierModel(category)
        self.category = category
        self.required_columns = ['Title', 'Content', 'Description', 'Type', 'Topic', 'SiteName', 'SiteId', 'Sentiment']

    def classify_spam(self, json_data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
        """
        Classify spam from JSON data.

        Args:
            json_data: dict or list of dicts with required fields

        Returns:
            dict or list of dicts with original data plus 'spam' and 'lang' fields
        """
        try:
            is_single = isinstance(json_data, dict)
            if is_single:
                json_data = [json_data]

            df = self._prepare_dataframe(json_data)
            df = self._apply_rules(df)
            df = self._predict_spam(df)
            df = self._finalize_language(df)

            result = df.drop(columns=['Combined_Text']).to_dict('records')
            return result[0] if is_single else result

        except Exception as e:
            logger.error(f"Error processing data for category {self.category}: {e}")
            raise

    def _prepare_dataframe(self, json_data: List[Dict]) -> pd.DataFrame:
        """Prepare DataFrame from JSON data."""
        df = pd.DataFrame(json_data)
        df = df.rename(columns={
            'id': 'Id',
            'title': 'Title',
            'content': 'Content',
            'description': 'Description',
            'type': 'Type',
            'topic': 'Topic',
            'site_name': 'SiteName',
            'site_id': 'SiteId',
            'sentiment': 'Sentiment'
        })
        
        if not all(col in df.columns for col in self.required_columns):
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        df['Combined_Text'] = df[['Title', 'Content', 'Description']].fillna('').agg(' '.join, axis=1)
        df = df[df['Combined_Text'].str.strip() != '']
        df['spam'] = True  # Default to spam
        df['lang'] = 'vi'  # Default language
        return df

    def _apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rule-based spam filtering."""
        # Rule 1: Type is tiktokComment -> not_spam
        tiktok_mask = df['Type'].str.lower() == 'tiktokcomment'
        df.loc[tiktok_mask, 'spam'] = False
        df.loc[tiktok_mask, 'lang'] = df.loc[tiktok_mask, 'Combined_Text'].apply(
            lambda x: detect_language(x) if isinstance(x, str) and x.strip() else 'unknown'
        )

        # Rule 2: SiteId in filter_site_id -> not_spam
        site_mask = df['SiteId'].isin(Config.CATEGORIES[self.category]["FILTER_SITE_IDS"])
        df.loc[site_mask, 'spam'] = False
        df.loc[site_mask, 'lang'] = df.loc[site_mask, 'Combined_Text'].apply(
            lambda x: detect_language(x) if isinstance(x, str) and x.strip() else 'unknown'
        )

        # Rule 3: Type is newsTopic -> detect language only
        news_mask = df['Type'].str.lower() == 'newstopic'
        df.loc[news_mask, 'lang'] = df.loc[news_mask, 'Combined_Text'].apply(
            lambda x: detect_language(x) if isinstance(x, str) and x.strip() else 'unknown'
        )

        # Rule 4: SiteName is trusted -> not_spam
        site_mask = df['SiteName'].str.lower().isin(Config.TRUSTED_SITES)
        df.loc[site_mask, 'spam'] = False
        df.loc[site_mask, 'lang'] = 'vi'

        # Rule 5: Sentiment Negative or Positive -> not_spam
        sentiment_mask = df['Sentiment'].str.lower().isin(['negative', 'positive'])
        df.loc[sentiment_mask, 'spam'] = False
        df.loc[sentiment_mask, 'lang'] = df.loc[sentiment_mask, 'Combined_Text'].apply(
            lambda x: detect_language(x) if isinstance(x, str) and x.strip() else 'unknown'
        )

        return df

    def _predict_spam(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict spam for remaining texts using the model."""
        predict_mask = (df['spam'] == True)
        texts_to_predict = df.loc[predict_mask, 'Combined_Text'].tolist()

        if texts_to_predict:
            predictions = self.model.predict(texts_to_predict)
            predict_indices = df[predict_mask].index
            for idx, pred in zip(predict_indices, predictions):
                df.at[idx, 'spam'] = pred['label'] == 'spam'
                df.at[idx, 'lang'] = 'vi' if pred['language'] == 'vi' else 'na'
        return df

    def _finalize_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize language labels."""
        df.loc[df['lang'] == 'vi', 'lang'] = 'vietnamese'
        df.loc[df['lang'] != 'vietnamese', 'lang'] = 'non_vietnamese'
        return df

def detect_language(text: str) -> str:
    """Detect language of the input text."""
    try:
        return detect(text)
    except:
        return "unknown"