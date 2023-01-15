import logo from './logo.svg';
import './App.css';
import React, { useEffect, useState } from 'react';
import axios from 'axios'

function App() {
  const [message, setMessage] = useState({})

  const [sentiment, setSentiment] = useState({})
  const [isSentimentLoaded, setIsSentimentLoaded] = useState(false)
  const [postText, setPostText] = useState('')

  const config = {
    headers: {
      'Content-Type': 'application/json;charset=UTF-8',
    },
  };

  useEffect(() => {
    axios.get('http://localhost:5000/home').then(response => {
      console.log("SUCCESS", response)
      setMessage(response)
    }).catch(error => {
      console.log(error)
    })

  }, [])

  const handleSubmit = async () => {
    try {
      setIsSentimentLoaded(false);
      const response = await axios.post('http://localhost:5000/home', {
        message: postText
      }, config);
      setSentiment(response);
      setIsSentimentLoaded(true);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>React + Flask Tutorial</p>
        <div>{message.status === 200 ?
          <h3>{message.data.message}</h3>
          :
          <h3>Loading webpage...</h3>}</div>
        <div>
          <input
            type="text"
            value={postText}
            onChange={(e) => setPostText(e.target.value)}
          />
          <button onClick={handleSubmit}>Submit</button>
          {isSentimentLoaded ? (
            <p>{sentiment.data.message}</p>
          ) : (
            <p>Loading...</p>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;