import os

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

        # TODO: Find a proper way to call both main.py and flask run without modifying directory
        # Issue: When running python -m src.main.py at ./task_2, cwd is at ./task_2/, however
        #        when using Flask run, cwd is at ./task_2/src. Hence we need to remove /src 
        #        to allow both Flask run and python -m src.main.py to work.
        current_directory = os.getcwd()
        current_directory = current_directory[:-4]

        sentiment_classifier = SentimentClassifier(
            mode="infer",
            current_directory=current_directory,
            model_path=f"{current_directory}\data\models\model_2023-01-16_11-11-54.pt",
            text=text,
        )
        prediction = sentiment_classifier.infer()
        return {"resultStatus": "SUCCESS", "message": prediction}
