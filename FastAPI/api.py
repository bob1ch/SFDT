from fastapi import FastAPI
from classificator.model import load_model

app = FastAPI()
model = None

@app.get("/")
async def test():
    return "i'm fine"

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/test_model")
def model_test():
    print(model.predict())
    return model.predict()[0]
