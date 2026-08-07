import React, { useState } from 'react';
import GlassCard from '../components/GlassCard';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import Button from '../components/Button';
import ResultCard from '../components/ResultCard';
import UploadReport from '../components/UploadReport';
import ChatBot from '../components/ChatBot';
import { predictHeartDisease, saveHistory, recordAssessment } from '../services/api';
import { generateSuggestions } from '../utils/suggestionEngine';
import { Heart, FileText, Keyboard, MessageSquare } from 'lucide-react';

const HeartDiseasePrediction = () => {
    const [formData, setFormData] = useState({
        patientName: '',
        age: '',
        gender: '',
        height_cm: '',
        weight_kg: '',
        systolic_bp: '',
        diastolic_bp: '',
        cholesterol: '',
        glucose: '',
        smoke: '0',
        alco: '0',
        active: '1'
    });

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [activeTab, setActiveTab] = useState('manual');
    const [suggestions, setSuggestions] = useState([]);
    const [ocrExtracted, setOcrExtracted] = useState(false);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData((prev) => ({ ...prev, [name]: value }));
    };

    const handleReset = () => {
        setFormData({
            patientName: '',
            age: '',
            gender: '',
            height_cm: '',
            weight_kg: '',
            systolic_bp: '',
            diastolic_bp: '',
            cholesterol: '',
            glucose: '',
            smoke: '0',
            alco: '0',
            active: '1'
        });
        setResult(null);
        setError('');
    };

    const triggerPrediction = async (dataToPredict) => {
        setLoading(true);
        setError('');

        try {
            const payload = {
                ...dataToPredict,
                age: Number(dataToPredict.age),
                height_cm: Number(dataToPredict.height_cm),
                weight_kg: Number(dataToPredict.weight_kg),
                systolic_bp: Number(dataToPredict.systolic_bp),
                diastolic_bp: Number(dataToPredict.diastolic_bp),
                cholesterol: Number(dataToPredict.cholesterol),
                glucose: Number(dataToPredict.glucose),
                smoke: dataToPredict.smoke === '1',
                alco: dataToPredict.alco === '1',
                active: dataToPredict.active === '1',
            };

            const predictionResponse = await predictHeartDisease(payload);
            setResult(predictionResponse);
            
            const sugs = generateSuggestions('heart', payload, predictionResponse);
            setSuggestions(sugs);
            
            let userId = localStorage.getItem("userId");
            if (!userId) {
              userId = "user_" + Date.now();
              localStorage.setItem("userId", userId);
            }

            // Save to session history (legacy bridge)
            saveHistory({
                userId: userId,
                userName: dataToPredict.patientName || 'Anonymous',
                type: 'heart',
                inputs: dataToPredict,
                prediction: predictionResponse.prediction,
                probability: predictionResponse.probability
            }).catch(e => console.error("Failed to save history:", e));

            // Persist to PostgreSQL via v1 API
            recordAssessment({
                condition: 'HEART',
                inputPayload: dataToPredict,
                modelVersion: 'heart-v1.0',
                probability: predictionResponse.probability,
                riskBand: predictionResponse.prediction === 1 ? 'HIGH' : 'LOW',
                observations: [
                    { name: 'systolic_bp', value: Number(dataToPredict.systolic_bp || 0), unit: 'mmHg' },
                    { name: 'diastolic_bp', value: Number(dataToPredict.diastolic_bp || 0), unit: 'mmHg' },
                    { name: 'cholesterol', value: Number(dataToPredict.cholesterol || 0), unit: 'mg/dL' },
                    { name: 'glucose', value: Number(dataToPredict.glucose || 0), unit: 'mg/dL' },
                ],
            }).catch(e => console.log('Assessment record (unauthenticated or guest session):', e.message));
        } catch (_err) {
            setError('Failed to connect to the prediction server. Please try again later.');
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        await triggerPrediction(formData);
    };

    const handleExtract = (extracted) => {
        // Never auto-trigger prediction after OCR. Always let the user review
        // the extracted values in the manual form before submitting.
        setFormData(prev => ({ ...prev, ...extracted }));
        setActiveTab('manual');
        setResult(null);
        setError('');
        setOcrExtracted(true);
    };

    const handleChatComplete = async (answers) => {
        setFormData(prev => ({ ...prev, ...answers }));
        await triggerPrediction(answers);
    };

    const genderOptions = [
        { value: 'male', label: 'Male' },
        { value: 'female', label: 'Female' }
    ];

    const yesNoOptions = [
        { value: '1', label: 'Yes' },
        { value: '0', label: 'No' }
    ];

    return (
        <div className="max-w-4xl mx-auto animate-fade-in-up pb-12">
            <div className="text-center mb-10">
                <div className="bg-red-500/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 text-red-600 border border-red-200">
                    <Heart size={32} />
                </div>
                <h1 className="text-3xl md:text-5xl font-black mb-4 text-slate-800">Cardiac Health Assessment</h1>
                <p className="text-slate-600 max-w-2xl mx-auto">
                    Enter cardiovascular parameters for personalized cardiac risk evaluation based on advanced machine learning algorithms.
                </p>
            </div>

            <div className="flex justify-center mb-8">
                <div className="bg-slate-100 p-1.5 rounded-2xl inline-flex text-sm shadow-inner shadow-slate-200/50">
                    <button onClick={() => setActiveTab('manual')} className={`flex items-center gap-2 px-6 py-2.5 rounded-xl transition-all ${activeTab === 'manual' ? 'bg-white text-red-600 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700 font-medium'}`}>
                        <Keyboard size={18} /> Manual Entry
                    </button>
                    <button onClick={() => setActiveTab('upload')} className={`flex items-center gap-2 px-6 py-2.5 rounded-xl transition-all ${activeTab === 'upload' ? 'bg-white text-red-600 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700 font-medium'}`}>
                        <FileText size={18} /> Upload Report
                    </button>
                    <button onClick={() => setActiveTab('chat')} className={`flex items-center gap-2 px-6 py-2.5 rounded-xl transition-all ${activeTab === 'chat' ? 'bg-white text-red-600 shadow-sm font-bold' : 'text-slate-500 hover:text-slate-700 font-medium'}`}>
                        <MessageSquare size={18} /> Chat Assistant
                    </button>
                </div>
            </div>

            {activeTab === 'upload' && (
                <GlassCard className="mb-8 border-t-4 border-t-red-500/50">
                    <UploadReport onExtract={handleExtract} />
                </GlassCard>
            )}

            {activeTab === 'chat' && (
                <div className="mb-8 border-t-4 border-t-red-500/50 rounded-2xl overflow-hidden">
                    <ChatBot 
                        initialData={formData}
                        onComplete={handleChatComplete}
                        questions={[
                            { key: 'age', question: "What is the patient's age in years?" },
                            { key: 'gender', question: "What is the patient's gender?", options: genderOptions },
                            { key: 'height_cm', question: "What is their height (in cm)?" },
                            { key: 'weight_kg', question: "What is their weight (in kg)?" },
                            { key: 'systolic_bp', question: "What is their Systolic Blood Pressure?" },
                            { key: 'diastolic_bp', question: "What is their Diastolic Blood Pressure?" },
                            { key: 'cholesterol', question: "What is their Cholesterol Level (mg/dL)?" },
                            { key: 'glucose', question: "What is their Fasting Blood Glucose (mg/dL)?" },
                            { key: 'smoke', question: "Are they a smoker?", options: yesNoOptions },
                            { key: 'alco', question: "Do they consume alcohol regularly?", options: yesNoOptions },
                            { key: 'active', question: "Are they physically active?", options: yesNoOptions }
                        ]}
                    />
                </div>
            )}

            {activeTab === 'manual' && (
            <GlassCard className="mb-8 border-t-4 border-t-red-500/50">
                {ocrExtracted && (
                    <div className="flex items-start gap-3 bg-orange-50 border border-orange-200 rounded-xl p-4 mb-6">
                        <svg className="w-5 h-5 text-orange-600 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                        <div>
                            <p className="text-sm font-bold text-orange-900">Fields extracted from your document</p>
                            <p className="text-xs text-orange-800 mt-0.5">Please review all values carefully — especially Height and Weight — before submitting. Correct any errors before running the assessment.</p>
                        </div>
                        <button onClick={() => setOcrExtracted(false)} className="ml-auto text-orange-400 hover:text-orange-700" aria-label="Dismiss">&times;</button>
                    </div>
                )}
                <form onSubmit={handleSubmit}>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Biometric Data</h2>

                    <div className="grid md:grid-cols-2 gap-6 mb-8">
                        <InputField label="Patient Name (Optional)" name="patientName" value={formData.patientName} onChange={handleChange} placeholder="Jane Doe" />
                        <InputField label="Age (Years)" name="age" type="number" value={formData.age} onChange={handleChange} placeholder="45" min="1" max="120" required />
                        <SelectField label="Gender" name="gender" value={formData.gender} onChange={handleChange} options={genderOptions} required />
                        <div className="grid grid-cols-2 gap-4">
                            <InputField label="Height (cm)" name="height_cm" type="number" value={formData.height_cm} onChange={handleChange} placeholder="170" min="120" max="220" required />
                            <InputField label="Weight (kg)" name="weight_kg" type="number" step="0.1" value={formData.weight_kg} onChange={handleChange} placeholder="70" min="30" max="200" required />
                        </div>
                    </div>

                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Vital Statistics</h2>

                    <div className="grid md:grid-cols-2 gap-6 mb-8">
                        <InputField label="Systolic BP (mmHg)" name="systolic_bp" type="number" value={formData.systolic_bp} onChange={handleChange} placeholder="120" min="80" max="200" required />
                        <InputField label="Diastolic BP (mmHg)" name="diastolic_bp" type="number" value={formData.diastolic_bp} onChange={handleChange} placeholder="80" min="50" max="120" required />
                        <InputField label="Cholesterol Level (mg/dL)" name="cholesterol" type="number" value={formData.cholesterol} onChange={handleChange} placeholder="200" min="100" max="400" required />
                        <InputField label="Blood Glucose (mg/dL)" name="glucose" type="number" value={formData.glucose} onChange={handleChange} placeholder="100" min="50" max="300" required />
                    </div>

                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Lifestyle Factors</h2>

                    <div className="grid md:grid-cols-3 gap-6 mb-8">
                        <SelectField label="Smoker" name="smoke" value={formData.smoke} onChange={handleChange} options={yesNoOptions} required />
                        <SelectField label="Alcohol Use" name="alco" value={formData.alco} onChange={handleChange} options={yesNoOptions} required />
                        <SelectField label="Physically Active" name="active" value={formData.active} onChange={handleChange} options={yesNoOptions} required />
                    </div>

                    {error && (
                        <div className="bg-danger/20 border border-danger p-4 rounded-lg mb-6 text-danger text-center">
                            {error}
                        </div>
                    )}

                    <div className="flex gap-4 pt-4">
                        <Button type="button" variant="secondary" onClick={handleReset} disabled={loading}>
                            Reset Form
                        </Button>
                        <Button type="submit" disabled={loading} className="bg-red-600 hover:bg-red-500 shadow-red-500/20">
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Processing...
                                </span>
                            ) : 'Initiate Cardiac Scan'}
                        </Button>
                    </div>
                </form>
            </GlassCard>
            )}

            {result && (
                <div id="heart-result" className="scroll-mt-24">
                    <ResultCard
                        prediction={result.prediction}
                        probability={result.probability}
                        riskLevel={result.risk_level}
                        riskBand={result.risk_band || (result.prediction === 1 ? 'HIGH' : 'LOW')}
                        explanation={result.explanation}
                        suggestions={suggestions}
                        extras={[
                            { label: 'Patient', value: formData.patientName || 'Anonymous' },
                            { label: 'Blood Pressure', value: `${formData.systolic_bp}/${formData.diastolic_bp} mmHg` },
                            { label: 'Cholesterol', value: `${formData.cholesterol} mg/dL` },
                            { label: 'Calculated BMI', value: result.bmi_val ? `${result.bmi_val}` : 'N/A' }
                        ]}
                    />
                </div>
            )}
        </div>
    );
};

export default HeartDiseasePrediction;
