from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import time
from spam import SpamClassifier
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Spam Classification API", version="1.0.0")

# Define Pydantic models for input validation
class Item(BaseModel):
    id: str
    topic_id: Optional[str]
    topic: str
    title: str
    content: str
    description: str
    sentiment: str
    site_name: str
    site_id: str
    label: str
    type: str

class Prediction(BaseModel):
    label: str
    prob: float

class ResultItem(BaseModel):
    id: str
    lang: str
    predict: List[Prediction]

class ResponseData(BaseModel):
    processing_time: float
    request_time: float
    results: List[Dict]

class ResponseModel(BaseModel):
    message: str
    data: ResponseData
    result: int

class RequestModel(BaseModel):
    category: str
    data: List[Item]

class ResultItem(BaseModel):
    topic_id: Optional[str]
    content: str
    description: str
    id: str
    label: str
    sentiment: str
    site_id: str
    site_name: str
    title: str
    topic: str
    type: str
    spam: bool
    lang: str

# Initialize classifiers for all categories at startup
CATEGORIES = [
    "finance", "real_estate", "ewallet", "healthcare_insurance", "ecommerce",
    "education", "logistic_delivery", "energy_fuels", "fnb", "investment",
    "fmcg", "retail", "technology_motorbike_food"
]
CLASSIFIERS = {category: SpamClassifier(category) for category in CATEGORIES}

def generate_cache_key(request_data: str) -> str:
    """Generate a cache key from request data using SHA-256."""
    return hashlib.sha256(request_data.encode()).hexdigest()

from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify_spam(request_data: str, category: str) -> Dict:
    """Cached classification function."""
    try:
        start_time = time.time()
        
        # Parse input data
        input_data = json.loads(request_data)
        print(input_data)
        # Get classifier for the category
        if category not in CLASSIFIERS:
            raise HTTPException(status_code=400, detail="Invalid category")
        classifier = CLASSIFIERS[category]
            
        # Process classification
        processing_start = time.time()
        results = classifier.classify_spam(json_data=input_data)
        processing_time = time.time() - processing_start
        request_time = time.time() - start_time
        
        formatted_results = [
            ResultItem(
                content=item.get("Content", ""),
                description=item.get("Description", ""),
                id=item.get("Id", ""),
                topic_id=item.get("topic_id", ""),
                label=item.get("Label", ""),
                sentiment=item.get("Sentiment", ""),
                site_id=item.get("SiteId", ""),
                site_name=item.get("SiteName", ""),
                title=item.get("Title", ""),
                topic=item.get("Topic", ""),
                type=item.get("Type", ""),
                spam=item.get("spam", False),
                lang=item.get("lang", "")
            ).dict() for item in results
        ]

        return {
            "message": "Prediction successful",
            "data": {
                "processing_time": processing_time,
                "request_time": request_time,
                "results": formatted_results
            },
            "result": 1
        }
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/classify")
async def classify_spam(request_model: RequestModel):
    """
    Classify spam content with caching.
    
    Args:
        request_model: RequestModel containing category and data items
    
    Returns:
        ResponseModel with classification results
    
    Raises:
        HTTPException: If category is invalid or processing fails
    """
    try:
        # Convert request to string for caching
        request_data = json.dumps([item.dict() for item in request_model.data], sort_keys=True)
        cache_key = generate_cache_key(request_data)
        
        # Call cached function
        response = cached_classify_spam(request_data, request_model.category)
        logger.info(f"Processed request for category: {request_model.category}")
        return response
        
    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")