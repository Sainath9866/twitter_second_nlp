# Install necessary dependencies


# Import required libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware



# Load sentiment analysis model from Hugging Face
print("Loading model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Define FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for request body
class TextData(BaseModel):
    text: str

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
    
@app.post("/analyze/")
async def analyze_sentiment(data: TextData):
    try:
        # Analyze the sentiment of the input text
        result = sentiment_pipeline(data.text)[0]
        sentiment_label = result['label']
        score = result['score']

        # Convert Hugging Face labels to POS, NEG, or NEUTRAL
        if sentiment_label in ["1 star", "2 stars"]:
            sentiment = "NEG"
        elif sentiment_label == "3 stars":
            sentiment = "NEUTRAL"
        else:
            sentiment = "POS"

        return {
            "text": data.text,
            "sentiment": sentiment,
            "confidence": round(score, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
