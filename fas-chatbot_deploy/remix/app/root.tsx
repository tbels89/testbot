import { useEffect, useState } from 'react';
import { Links, Meta, Outlet } from "@remix-run/react";
import { chatbot } from './index.py';

export default function App() {
  const [userInput, setUserInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);

  const handleUserInput = (e) => {
    setUserInput(e.target.value);
  };

  const handleSubmit = async () => {
    if (userInput.trim() === '') return;

    const userMessage = { text: userInput, role: 'user' };
    setChatHistory([...chatHistory, userMessage]);
    setUserInput('');

    // Call your chatbot logic to get the response
    const chatbotResponse = await chatbot(userInput);

    const botMessage = { text: chatbotResponse, role: 'bot' };
    setChatHistory([...chatHistory, botMessage]);
  };

  return (
    <div>
      {/* Your chat interface here */}
      <div className="chat-history">
        {chatHistory.map((message, index) => (
          <div key={index} className={message.role}>
            {message.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={userInput}
        onChange={handleUserInput}
        placeholder="Type a message..."
      />
      <button onClick={handleSubmit}>Send</button>
    </div>
  );
}
