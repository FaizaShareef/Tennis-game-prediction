from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fractions import Fraction
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder


class InputData(BaseModel):
    HomeTeamLP: int
    AwayTeamLP: int
    HTFormPts:int
    ATFormPts:int
    HTEloRatings:float
    ATEloRatings:float
    numbers:int
    Avg_HST:float
    Avg_AST:float
    

with open('xgb_model2.pickle', 'rb') as f:
    model1 = pickle.load(f)

app = FastAPI()

@app.post('/scoccer_predictions')
def get_prediction(data: InputData):
    received = pd.DataFrame(jsonable_encoder(data), index=[0])
    cols_new = ["ATFormPts","AwayTeamLP","HTFormPts","ATFormPts","HTEloRatings","ATEloRatings","numbers","Avg_HST","Avg_AST"]

    received = received[cols_new]
    pred_name = model1.predict(received)[0]
    Prob = model1.predict_proba(received) * 100
    probability_percent = {
			"Away_per": round(Prob.tolist()[0][0], 2),
			"Draw_per": round(Prob.tolist()[0][1], 2),
            "Home_per": round(Prob.tolist()[0][2], 2),

	}
    decimal_odds = {
		"Away_odds": round(100/Prob.tolist()[0][0], 2),
		"Draw_odds": round(100/Prob.tolist()[0][1], 2),
        "Home_odds": round(100/Prob.tolist()[0][2], 2),


    }

    return {'prediction': pred_name,
			'probability_percent': probability_percent,
			"predicted_decimal_odds": decimal_odds
			
			}