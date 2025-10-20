"use client";

import { useState, useEffect, useRef } from "react";
import AudioPlayer, { RHAP_UI } from "react-h5-audio-player";
import "react-h5-audio-player/lib/styles.css";

interface Message {
  id: number;
  sender: "bot" | "user";
  text?: string;
  audioUrl?: string;
  time: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const idCounter = useRef(1);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const initialMessage: Message = {
      id: idCounter.current++,
      sender: "bot",
      text: "Hi, I'm your AI assistant. How do you feel today?",
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setMessages([initialMessage]);
  }, []);

  const handleSend = () => {
    if (!input.trim()) return;
    const userMessage: Message = {
      id: idCounter.current++,
      sender: "user",
      text: input,
      time: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    setTimeout(() => {
      const botMessage: Message = {
        id: idCounter.current++,
        sender: "bot",
        text: "Ok",
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setMessages((prev) => [...prev, botMessage]);
    }, 500);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSend();
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        const audioUrl = URL.createObjectURL(audioBlob);

        const audioMessage: Message = {
          id: idCounter.current++,
          sender: "user",
          audioUrl,
          time: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        };
        setMessages((prev) => [...prev, audioMessage]);

        setTimeout(() => {
          const botMessage: Message = {
            id: idCounter.current++,
            sender: "bot",
            text: "Ok",
            time: new Date().toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            }),
          };
          setMessages((prev) => [...prev, botMessage]);
        }, 500);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please grant permission.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleMicClick = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 via-[var(--color-secondary-blue)]/10 to-gray-100">
      {/* Navigation Bar */}
      <nav className="bg-white/80 backdrop-blur-sm border-b border-gray-200 flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-[var(--color-secondary-blue)] to-[var(--color-secondary-purple)] rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">SC</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-800">SentiCore</h1>
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
                <div className="flex-1 overflow-y-auto px-6 py-6 min-h-0">
                  <div className="space-y-4">
                    {messages.map((message, index) =>
                      message.sender === "bot" ? (
                        <div
                          key={message.id}
                          className="flex items-start space-x-3 opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]"
                          style={{ animationDelay: `${index * 0.02}s` }}
                        >
                          <div className="w-8 h-8 bg-secondary-purple rounded-full flex-shrink-0"></div>
                          <div className="flex flex-col space-y-1 max-w-lg">
                            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                              <p className="text-sm text-gray-800">
                                {message.text}
                              </p>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div
                          key={message.id}
                          className="flex items-start space-x-3 justify-end opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]"
                          style={{ animationDelay: `${index * 0.02}s` }}
                        >
                          <div className="flex flex-col space-y-1 max-w-lg items-end">
                            <div className="bg-gradient-to-br from-[var(--color-secondary-blue)] to-[var(--color-secondary-purple)] rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
                              {message.text ? (
                                <p className="text-sm text-white">
                                  {message.text}
                                </p>
                              ) : message.audioUrl ? (
                                <div className="audio-player-wrapper">
                                  <AudioPlayer
                                    src={message.audioUrl!}
                                    layout="horizontal-reverse"
                                    showJumpControls={false}
                                    customProgressBarSection={[
                                      RHAP_UI.PROGRESS_BAR,
                                    ]}
                                    customControlsSection={[
                                      RHAP_UI.MAIN_CONTROLS,
                                      RHAP_UI.DURATION,
                                    ]}
                                    customAdditionalControls={[]}
                                    customVolumeControls={[]}
                                    autoPlayAfterSrcChange={false}
                                    className="voice-message-player"
                                  />
                                </div>
                              ) : null}
                            </div>
                            <span className="text-xs text-gray-400 mr-2">
                              {message.time}
                            </span>
                          </div>
                          <div className="w-8 h-8 bg-secondary-blue rounded-full flex-shrink-0"></div>
                        </div>
                      )
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </div>

                {/* Input */}
                <div className="flex-shrink-0 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
                  <div className="flex items-center space-x-3 bg-white rounded-2xl border border-gray-200 px-4 py-3 shadow-sm hover:shadow-md transition-shadow">
                    <input
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Type your message here..."
                      className="flex-1 outline-none text-sm bg-transparent placeholder:text-gray-400"
                    />
                    <button
                      onClick={handleSend}
                      className="px-4 py-2 bg-gradient-to-r from-[var(--color-secondary-blue)] to-[var(--color-secondary-purple)] text-white rounded-xl font-medium hover:shadow-lg active:scale-95 transition-all"
                    >
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>

              {/* Mic Sidebar */}
              <div className="hidden lg:flex bg-gradient-to-b from-[var(--color-secondary-blue)]/10 to-gray-50/30 border-l border-gray-100 p-6 items-center justify-center">
                <button
                  onClick={handleMicClick}
                  className={`w-24 h-24 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95 flex items-center justify-center ${
                    isRecording
                      ? "bg-gradient-to-br from-red-400 to-red-500 animate-pulse"
                      : "bg-gradient-to-br from-[var(--color-secondary-blue)] to-[var(--color-secondary-purple)]"
                  }`}
                >
                  {isRecording ? (
                    <svg
                      className="w-12 h-12 text-white"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                  ) : (
                    <svg
                      className="w-12 h-12 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                      />
                    </svg>
                  )}
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

        /* Custom styling for audio player in chat bubbles */
        .audio-player-wrapper {
          width: 200px;
          max-width: 100%;
        }

        .audio-player-wrapper .rhap_container {
          background: transparent;
          box-shadow: none;
          padding: 0;
        }

        .audio-player-wrapper .rhap_progress-section {
          width: 200px;
          margin-bottom: 4px;
        }

        .audio-player-wrapper .rhap_progress-bar {
          background-color: rgba(255, 255, 255, 0.3);
          height: 4px;
        }

        .audio-player-wrapper .rhap_progress-filled,
        .audio-player-wrapper .rhap_progress-indicator {
          background-color: white;
        }

        .audio-player-wrapper .rhap_controls-section {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .audio-player-wrapper .rhap_time {
          margin-left: 6px; /* small space before duration */
          color: white;
          font-size: 12px;
        }

        .audio-player-wrapper .rhap_main-controls button {
          color: white;
          width: 32px;
          height: 32px;
        }

        .audio-player-wrapper .rhap_main-controls-button svg {
          width: 32px;
          height: 32px;
        }
      `}</style>
    </div>
  );
}
