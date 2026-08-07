import React, { useState, useEffect, useRef } from 'react';
import Surface from './Surface';
import Button from './Button';
import InputField from './InputField';
import StatusBadge from './StatusBadge';
import { getBotQuestions, processBotAnswers } from '../services/api';
import { Sparkles, Bot, Send, CheckCircle, RefreshCw } from 'lucide-react';

const ChatBot = ({ condition = 'diabetes', onComplete }) => {
  const [questions, setQuestions] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [messages, setMessages] = useState([]);
  const [inputVal, setInputVal] = useState('');
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [completed, setCompleted] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const qList = await getBotQuestions(condition);
        setQuestions(qList || []);
        if (qList && qList.length > 0) {
          setMessages([
            {
              sender: 'bot',
              text: `Welcome to the guided ${condition} screening assistant. Let's step through your biometrics one by one.`,
            },
            {
              sender: 'bot',
              text: qList[0].text,
            },
          ]);
        }
      } catch (err) {
        console.error('Failed to load bot questions:', err);
        setMessages([
          {
            sender: 'bot',
            text: 'Unable to initialize guided questions. Please switch to manual form entry.',
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchQuestions();
  }, [condition]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputVal.trim() || completed) return;

    const currentQ = questions[currentIndex];
    const userText = inputVal.trim();
    const numVal = parseFloat(userText);

    if (isNaN(numVal)) {
      setMessages((prev) => [
        ...prev,
        { sender: 'user', text: userText },
        { sender: 'bot', text: `Please enter a valid numeric value for ${currentQ.label || currentQ.key}.` },
      ]);
      setInputVal('');
      return;
    }

    const updatedAnswers = { ...answers, [currentQ.key]: numVal };
    setAnswers(updatedAnswers);

    const newMessages = [
      ...messages,
      { sender: 'user', text: `${userText} ${currentQ.unit || ''}`.trim() },
    ];

    const nextIdx = currentIndex + 1;
    if (nextIdx < questions.length) {
      setCurrentIndex(nextIdx);
      newMessages.push({ sender: 'bot', text: questions[nextIdx].text });
      setMessages(newMessages);
      setInputVal('');
    } else {
      setCompleted(true);
      newMessages.push({
        sender: 'bot',
        text: 'All biometric questions completed! Click below to process your assessment.',
      });
      setMessages(newMessages);
      setInputVal('');
    }
  };

  const handleFinish = async () => {
    setProcessing(true);
    try {
      const result = await processBotAnswers(condition, answers);
      if (onComplete) {
        onComplete(result);
      }
    } catch (err) {
      console.error('Error processing bot answers:', err);
    } finally {
      setProcessing(false);
    }
  };

  const handleReset = () => {
    setCurrentIndex(0);
    setAnswers({});
    setCompleted(false);
    setInputVal('');
    if (questions.length > 0) {
      setMessages([
        {
          sender: 'bot',
          text: `Guided ${condition} screening reset. Let's begin again.`,
        },
        {
          sender: 'bot',
          text: questions[0].text,
        },
      ]);
    }
  };

  return (
    <Surface variant="flat" className="my-4 space-y-4">
      {/* Header with Amber Guided AI Identity */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 border border-amber-500/30 flex items-center justify-center flex-shrink-0 shadow-inner">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-100">Guided Screening Assistant</h3>
            <p className="text-[11px] text-slate-400">Step-by-step interactive questionnaire</p>
          </div>
        </div>
        <StatusBadge label="Amber Guided AI" status="attention" size="sm" />
      </div>

      {/* Messages Viewport */}
      <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800/80 h-72 overflow-y-auto space-y-3 font-sans">
        {loading ? (
          <div className="text-xs text-slate-400 italic text-center py-8">Loading bot questions...</div>
        ) : (
          messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-2xl text-xs leading-relaxed ${
                  msg.sender === 'user'
                    ? 'bg-teal-600 text-slate-950 font-semibold rounded-tr-none shadow-sm'
                    : 'bg-slate-900 text-slate-200 border border-slate-800 rounded-tl-none'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input or Complete Actions */}
      {!completed ? (
        <form onSubmit={handleSend} className="flex gap-2">
          <div className="flex-1">
            <InputField
              value={inputVal}
              onChange={(e) => setInputVal(e.target.value)}
              placeholder={
                questions[currentIndex]
                  ? `Enter ${questions[currentIndex].label || questions[currentIndex].key} (${questions[currentIndex].unit || 'number'})...`
                  : 'Type answer...'
              }
              unit={questions[currentIndex]?.unit}
              disabled={loading}
              className="mb-0"
            />
          </div>
          <Button
            type="submit"
            disabled={!inputVal.trim() || loading}
            variant="ai"
            icon={Send}
            className="flex-shrink-0 font-bold"
          >
            Submit
          </Button>
        </form>
      ) : (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-3 bg-slate-900 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center gap-2 text-xs text-teal-400 font-semibold">
            <CheckCircle className="w-4 h-4" />
            <span>Questionnaire Complete ({Object.keys(answers).length} fields captured)</span>
          </div>
          <div className="flex gap-2 w-full sm:w-auto">
            <Button onClick={handleReset} variant="ghost" size="sm" icon={RefreshCw}>
              Restart
            </Button>
            <Button
              onClick={handleFinish}
              loading={processing}
              loadingLabel="Processing..."
              variant="primary"
              size="sm"
              icon={Sparkles}
            >
              Run Risk Analysis
            </Button>
          </div>
        </div>
      )}
    </Surface>
  );
};

export default ChatBot;
