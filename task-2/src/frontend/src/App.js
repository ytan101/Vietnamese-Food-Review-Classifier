import "./App.css";
import React, { useEffect, useState } from "react";
import axios from "axios";

import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Box from "@mui/material/Box";

function App() {
  const [message, setMessage] = useState({});

  const [sentiment, setSentiment] = useState({});
  const [isSentimentLoaded, setIsSentimentLoaded] = useState(false);
  const [postText, setPostText] = useState("");

  const config = {
    headers: {
      "Content-Type": "application/json;charset=UTF-8",
    },
  };

  useEffect(() => {
    axios
      .get("http://localhost:5000/home")
      .then((response) => {
        console.log("SUCCESS", response);
        setMessage(response);
      })
      .catch((error) => {
        console.log(error);
      });
  }, []);

  const handleSubmit = async () => {
    try {
      setIsSentimentLoaded(false);
      const response = await axios.post(
        "http://localhost:5000/home",
        {
          message: postText,
        },
        config
      );
      setSentiment(response);
      setIsSentimentLoaded(true);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="header-text">Food Review Sentiments</h1>
      </header>
      <body className="App-body">
        <Box
          sx={{
            "& > :not(style)": { m: 2 },
          }}
          className="border"
        >
          <p>How did you feel about your food today?</p>
          <div>
            <TextField
              style={{
                width: "500px",
                height: "10%",
                marginBottom: "1px",
                backgroundColor: "rgb(255, 208, 177, 0.3)",
                textColor: "white",
              }}
              id="filled-multiline-static"
              fullWidth
              label="Type your review here"
              multiline
              rows={4}
              variant="filled"
              type="text"
              value={postText}
              onChange={(e) => setPostText(e.target.value)}
            />
          </div>
          <div>
            <Button
              style={{ backgroundColor: "black", color: "white" }}
              variant="contained"
              onClick={handleSubmit}
            >
              Get Sentiment
            </Button>
            {isSentimentLoaded ? (
              <p>
                You feel{" "}
                <span
                  className={
                    sentiment.data.message === "positive"
                      ? "positive-message"
                      : sentiment.data.message === "negative"
                      ? "negative-message"
                      : sentiment.data.message === "neutral"
                      ? "neutral-message"
                      : ""
                  }
                >
                  {sentiment.data.message}
                </span>{" "}
                about your food
              </p>
            ) : (
              <p>calculating sentiments...</p>
            )}
          </div>
        </Box>
        <div>
          <br></br>
        </div>
        <div>
          {message.status === 200 ? (
            <p>{message.data.message}</p>
          ) : (
            <p>Connecting to Flask backend...</p>
          )}
        </div>
      </body>
    </div>
  );
}

export default App;
