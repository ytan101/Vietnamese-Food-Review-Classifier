import subprocess
import chardet

from flask_restful import Api, Resource, reqparse
from src.main import SentimentClassifier


class SentimentApiHandler(Resource):
    def get(self):
        return {"resultStatus": "SUCCESS", "message": "Connected to Flask backend"}

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("message", type=str)
        args = parser.parse_args()
        text = args["message"]

        sentiment_classifier = SentimentClassifier(
            mode="infer",
            model_path="../data/models/model_2023-01-15_01-15-00.pt",
            text=text,
        )
        prediction = sentiment_classifier.infer()
        return {"resultStatus": "SUCCESS", "message": prediction}
