from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize the sentiment-analysis model
sentiment = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

# Define input and output data structures
class TextInput(BaseModel):
    text: str

app = FastAPI()

@app.post("/analyze/")
def analyze_sentiment(input: TextInput):
    try:
        emotion = sentiment(input.text, max_length=40, truncation=True)
        return {"label": emotion[0]['label']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
