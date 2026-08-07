import React, { useState, useRef, useEffect } from 'react';
import { Bot, User, Send } from 'lucide-react';
import Surface from './Surface';
import Button from './Button';

const ChatBot = ({ questions, initialData, onComplete }) => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! I am your HealthLens AI guided assessment assistant. I'll walk you through a quick series of questions to populate your clinical parameters.",
      sender: 'bot',
    },
    { text: questions[0].question, sender: 'bot', key: questions[0].key },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState(initialData || {});
  const [isComplete, setIsComplete] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (isComplete || !inputValue.trim()) return;

    const currentQ = questions[currentStep];
    const newAnswers = { ...answers, [currentQ.key]: inputValue.trim() };
    setAnswers(newAnswers);

    const newMessages = [...messages, { text: inputValue, sender: 'user' }];
    setInputValue('');

    const nextStep = currentStep + 1;
    if (nextStep < questions.length) {
      setTimeout(() => {
        setMessages([
          ...newMessages,
          { text: questions[nextStep].question, sender: 'bot', key: questions[nextStep].key },
        ]);
        setCurrentStep(nextStep);
      }, 400);
    } else {
      setIsComplete(true);
      setTimeout(() => {
        setMessages([
          ...newMessages,
          {
            text: 'Thank you! All guided parameters have been captured. Review your values in the form above and click Submit Assessment.',
            sender: 'bot',
          },
        ]);
        onComplete(newAnswers);
      }, 400);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isComplete) {
      handleSend();
    }
  };

  return (
    <Surface variant="flat" className="flex flex-col h-[520px] p-0 overflow-hidden shadow-xl animate-fade-in-up">
      {/* Header */}
      <div className="bg-slate-900 px-6 py-4 flex items-center justify-between border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="bg-amber-500/20 p-2 rounded-xl text-amber-400 border border-amber-500/30">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-slate-100 font-bold text-sm">Guided Clinical Assistant</h3>
            <p className="text-slate-400 text-xs font-mono">Step {currentStep + 1} of {questions.length}</p>
          </div>
        </div>
        <span className="px-2.5 py-1 text-[10px] font-bold rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">
          AI INTERACTION
        </span>
      </div>

      {/* Messages Feed */}
      <div className="flex-1 p-6 overflow-y-auto flex flex-col gap-4 bg-slate-950/60">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex items-end gap-3 ${msg.sender === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}
          >
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                msg.sender === 'user'
                  ? 'bg-teal-600 text-white'
                  : 'bg-slate-800 text-amber-400 border border-slate-700'
              }`}
            >
              {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
            </div>
            <div
              className={`px-4 py-3 rounded-2xl max-w-[82%] text-xs md:text-sm leading-relaxed ${
                msg.sender === 'user'
                  ? 'bg-teal-600 text-white rounded-br-none shadow-md'
                  : 'bg-slate-900 text-slate-200 border border-slate-800 shadow-sm rounded-bl-none space-y-2'
              }`}
            >
              <p>{msg.text}</p>
              {!isComplete && msg.sender === 'bot' && idx === messages.length - 1 && questions[currentStep]?.options && (
                <div className="flex flex-wrap gap-2 mt-3 text-xs">
                  {questions[currentStep].options.map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => setInputValue(opt.value)}
                      className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 border border-teal-500/30 text-teal-300 rounded-lg font-medium transition-colors"
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="p-4 bg-slate-900 border-t border-slate-800 flex gap-3">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isComplete ? 'Guided assessment complete — review values above.' : 'Type your answer...'}
          disabled={isComplete}
          aria-label="Guided chatbot answer input"
          className="flex-1 px-4 py-2.5 bg-slate-950 text-slate-100 placeholder-slate-500 border border-slate-800 rounded-xl text-xs md:text-sm focus:border-teal-400 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <Button
          onClick={handleSend}
          disabled={!inputValue.trim() || isComplete}
          variant="primary"
          size="sm"
          icon={Send}
          aria-label="Send response"
        >
          Send
        </Button>
      </div>
    </Surface>
  );
};

export default ChatBot;
