import React, { useState, useRef, useEffect } from 'react';
import { Bot, User, Send } from 'lucide-react';

const ChatBot = ({ questions, initialData, onComplete }) => {
    const [messages, setMessages] = useState([
        { text: "Hello! I'm your healthcare assistant. I'll guide you through a few quick questions to complete your assessment. Let's start with the first one.", sender: 'bot' },
        { text: questions[0].question, sender: 'bot', key: questions[0].key }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [currentStep, setCurrentStep] = useState(0);
    const [answers, setAnswers] = useState(initialData || {});
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = () => {
        if (!inputValue.trim()) return;

        const currentQ = questions[currentStep];
        
        // Save user answer
        const newAnswers = { ...answers, [currentQ.key]: inputValue.trim() };
        setAnswers(newAnswers);

        // Add user message
        const newMessages = [...messages, { text: inputValue, sender: 'user' }];
        setInputValue('');
        
        // Determine next step
        const nextStep = currentStep + 1;
        if (nextStep < questions.length) {
            // Next question
            setTimeout(() => {
                setMessages([
                    ...newMessages,
                    { text: questions[nextStep].question, sender: 'bot', key: questions[nextStep].key }
                ]);
                setCurrentStep(nextStep);
            }, 500);
        } else {
            // Assessment complete
            setTimeout(() => {
                setMessages([
                    ...newMessages,
                    { text: "Thank you! I have all the information I need. Running prediction now...", sender: 'bot' }
                ]);
                onComplete(newAnswers);
            }, 500);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleSend();
        }
    };

    return (
        <div className="flex flex-col bg-slate-50 border border-slate-200 rounded-2xl overflow-hidden h-[500px] shadow-sm animate-fade-in-up">
            {/* Chat header */}
            <div className="bg-blue-600 px-6 py-4 flex items-center gap-4">
                <div className="bg-white/20 p-2 rounded-xl text-white">
                    <Bot size={24} />
                </div>
                <div>
                    <h3 className="text-white font-bold text-lg">AI Health Assistant</h3>
                    <p className="text-blue-100 text-sm">Guided Assessment</p>
                </div>
            </div>

            {/* Chat messages */}
            <div className="flex-1 p-6 overflow-y-auto flex flex-col gap-4">
                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex items-end gap-3 ${msg.sender === 'user' ? 'self-end flex-row-reverse' : 'self-start'}`}>
                        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${msg.sender === 'user' ? 'bg-slate-800 text-white' : 'bg-blue-100 text-blue-600 border border-blue-200'}`}>
                            {msg.sender === 'user' ? <User size={16} /> : <Bot size={16} />}
                        </div>
                        <div className={`px-5 py-3 rounded-2xl max-w-[80%] ${
                            msg.sender === 'user' 
                                ? 'bg-slate-800 text-white rounded-br-sm shadow-md' 
                                : 'bg-white text-slate-700 border border-slate-200 shadow-sm rounded-bl-sm space-y-2'
                        }`}>
                            <p>{msg.text}</p>
                            {/* If it's the current bot question and has options, show shortcut chips */}
                            {msg.sender === 'bot' && idx === messages.length - 1 && questions[currentStep]?.options && (
                                <div className="flex flex-wrap gap-2 mt-3 text-sm">
                                    {questions[currentStep].options.map(opt => (
                                        <button 
                                            key={opt.value}
                                            onClick={() => setInputValue(opt.value)}
                                            className="px-3 py-1 bg-blue-50 hover:bg-blue-100 border border-blue-200 text-blue-700 rounded-full font-medium transition-colors"
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

            {/* Input area */}
            <div className="p-4 bg-white border-t border-slate-200 flex gap-3">
                <input 
                    type="text" 
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your answer here..."
                    disabled={currentStep >= questions.length}
                    className="flex-1 px-4 py-3 bg-slate-100 border-none rounded-xl focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all outline-none text-slate-700 placeholder-slate-400"
                />
                <button 
                    onClick={handleSend}
                    disabled={!inputValue.trim() || currentStep >= questions.length}
                    className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white w-12 h-12 flex items-center justify-center rounded-xl shadow-md transition-all active:scale-95 flex-shrink-0"
                >
                    <Send size={20} className="ml-0.5" />
                </button>
            </div>
        </div>
    );
};

export default ChatBot;
