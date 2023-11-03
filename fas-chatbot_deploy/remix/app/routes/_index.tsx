import { useState } from 'react';
import { chatbot } from '../index.py'; // Import your chatbot logic
import type { MetaFunction } from "@vercel/remix";

export const meta: MetaFunction = () => {
  return [
    { title: "Chatbot Remix App" },
    { name: "description", content: "Chat with the Remix Chatbot!" },
  ];
};

export default function Index() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);

  const handleUserInput = async () => {
    if (input) {
      setMessages([...messages, { text: input, role: 'user' }]);
      setInput('');

      // Call your chatbot function with user input and chat history
      const response = await chatbot({ input: input, history: messages });

      const chatbotResponse = response.response;

      // Determine the role of the chatbot's response
      const role = chatbotResponse.startsWith('SYS FAILURE') ? 'error' : 'bot';

      setMessages([...messages, { text: chatbotResponse, role: role }]);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: "1.8" }}>
      <h1>Chat with the Remix Chatbot</h1>
      <div>
        {messages.map((message, index) => (
          <div key={index} className={message.role}>
            {message.text}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your message..."
      />
      <button onClick={handleUserInput}>Send</button>
    </div>
  );
}
