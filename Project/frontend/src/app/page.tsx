'use client';

import { useState, useEffect, useRef } from 'react';

interface Message {
  id: number;
  sender: 'bot' | 'user';
  text: string;
  time: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const idCounter = useRef(1);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const initialMessage: Message = {
      id: idCounter.current++,
      sender: 'bot',
      text: "Hi, I'm your AI assistant. How do you feel today?",
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setMessages([initialMessage]);
  }, []);

  const handleSend = () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: idCounter.current++,
      sender: 'user',
      text: input,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');

    setTimeout(() => {
      const botMessage: Message = {
        id: idCounter.current++,
        sender: 'bot',
        text: 'Ok',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages(prev => [...prev, botMessage]);
    }, 500);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 via-green-50 to-gray-100">
      {/* Navigation Bar */}
      <nav className="bg-white/80 backdrop-blur-sm border-b border-gray-200 flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">SC</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">SentiCore</h1>
                <p className="text-xs text-gray-500"></p>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden px-4 sm:px-6 lg:px-8 py-6">
        <div className="h-full max-w-7xl mx-auto">
          <div className="h-full bg-white rounded-3xl shadow-xl overflow-hidden border border-gray-100">
            <div className="h-full grid lg:grid-cols-4">

              {/* Chat Area */}
              <div className="lg:col-span-3 h-full flex flex-col min-h-0">

                {/* Messages Container (this scrolls) */}
                <div className="flex-1 overflow-y-auto px-6 py-6 min-h-0">
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                      message.sender === 'bot' ? (
                        <div key={message.id} className="flex items-start space-x-3 opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]" style={{ animationDelay: `${index * 0.02}s` }}>
                          <div className="w-8 h-8 bg-gradient-to-br from-green-400 to-green-500 rounded-full flex-shrink-0"></div>
                          <div className="flex flex-col space-y-1 max-w-lg">
                            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                              <p className="text-sm text-gray-800">
                                {message.text}
                              </p>
                            </div>
                            <span className="text-xs text-gray-400 ml-2">{message.time}</span>
                          </div>
                        </div>
                      ) : (
                        <div key={message.id} className="flex items-start space-x-3 justify-end opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]" style={{ animationDelay: `${index * 0.02}s` }}>
                          <div className="flex flex-col space-y-1 max-w-lg items-end">
                            <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
                              <p className="text-sm text-white">
                                {message.text}
                              </p>
                            </div>
                            <span className="text-xs text-gray-400 mr-2">{message.time}</span>
                          </div>
                          <div className="w-8 h-8 bg-gradient-to-br from-gray-400 to-gray-500 rounded-full flex-shrink-0"></div>
                        </div>
                      )
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </div>

                {/* Input Area (stays at bottom) */}
                <div className="flex-shrink-0 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
                  <div className="flex items-center space-x-3 bg-white rounded-2xl border border-gray-200 px-4 py-3 shadow-sm hover:shadow-md transition-shadow">
                    {/* <button className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors" aria-label="attach">
                      <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                      </svg>
                    </button> */}
                    <input 
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Type your message here..."
                      className="flex-1 outline-none text-sm bg-transparent placeholder:text-gray-400"
                      aria-label="message-input"
                    />
                    <button 
                      onClick={handleSend}
                      className="px-4 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-xl font-medium hover:from-green-600 hover:to-green-700 transition-all hover:shadow-lg active:scale-95"
                      aria-label="send"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    </button>
                  </div>
                </div>

              </div>

              {/* Microphone Sidebar */}
              <div className="hidden lg:flex bg-gradient-to-b from-green-50/50 to-gray-50/30 border-l border-gray-100 p-6 items-center justify-center">
                <button className="w-24 h-24 bg-gradient-to-br from-green-400 to-green-600 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95 flex items-center justify-center" aria-label="record">
                  <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                </button>
              </div>

            </div>
          </div>
        </div>
      </main>

      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        
        @keyframes fadeSlideIn {
          from {
            opacity: 0;
            transform: translateY(12px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        ::-webkit-scrollbar {
          width: 8px;
        }
        
        ::-webkit-scrollbar-track {
          background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.3);
          border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: rgba(156, 163, 175, 0.5);
        }
      `}</style>
    </div>
  );
}
