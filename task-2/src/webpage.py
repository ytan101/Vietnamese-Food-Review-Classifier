from flask import Flask
import subprocess

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello! This is the main page <h1>Hello<h1>"

@app.route("/<name>")
def user(name):
    return f"Made by {name}"

@app.route("/infer")
def inference():
    subprocess.run(["python", "main.py", "--mode", "infer"])
    return 'Model Inference!'

if __name__ == "__main__":
    app.run()