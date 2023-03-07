import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
from bnote_schema import Noteparam


app = FastAPI()

clf = open("note_classifier.pkl", "rb")
classifier=pickle.load(clf)

@app.get('/')
def index():
    return {'message': 'Hello!'}

@app.post('/predict')
def note_pred(data:Noteparam):
    data=data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    
    pred = classifier.predict([[variance, skewness, curtosis, entropy]])
    
    if(pred[0]>0.6):
        prediction = "Fake note"
    else:
        prediction = "Bank note"
        
    return{
        'prediction' : prediction
    }
    
if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    