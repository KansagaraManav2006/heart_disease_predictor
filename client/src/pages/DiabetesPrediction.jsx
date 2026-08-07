import React, { useState } from 'react';
import Surface from '../components/Surface';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import Button from '../components/Button';
import ResultCard from '../components/ResultCard';
import UploadReport from '../components/UploadReport';
import ChatBot from '../components/ChatBot';
import PageHeader from '../components/PageHeader';
import SegmentedTabs from '../components/SegmentedTabs';
import StatusBadge from '../components/StatusBadge';
import ErrorState from '../components/ErrorState';
import { predictDiabetes, saveHistory, recordAssessment } from '../services/api';
import { generateSuggestions } from '../utils/suggestionEngine';
import { Activity, FileText, Keyboard, MessageSquare, RotateCcw, Play } from 'lucide-react';

const DiabetesPrediction = () => {
  const [formData, setFormData] = useState({
    patientName: '',
    age: '',
    gender: '',
    bmi: '',
    smokingHistory: '',
    hypertension: '',
    heartDisease: '',
    hba1c: '',
    glucose: '',
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
      bmi: '',
      smokingHistory: '',
      hypertension: '',
      heartDisease: '',
      hba1c: '',
      glucose: '',
    });
    setResult(null);
    setError('');
    setOcrExtracted(false);
  };

  const triggerPrediction = async (dataToPredict) => {
    setLoading(true);
    setError('');

    try {
      const predictionResponse = await predictDiabetes({
        ...dataToPredict,
        age: Number(dataToPredict.age),
        bmi: Number(dataToPredict.bmi),
        hba1c: Number(dataToPredict.hba1c),
        glucose: Number(dataToPredict.glucose),
      });

      setResult(predictionResponse);
      const sugs = generateSuggestions('diabetes', dataToPredict, predictionResponse);
      setSuggestions(sugs);

      let userId = localStorage.getItem('userId');
      if (!userId) {
        userId = 'user_' + Date.now();
        localStorage.setItem('userId', userId);
      }

      saveHistory({
        userId,
        userName: dataToPredict.patientName || 'Anonymous',
        type: 'diabetes',
        inputs: dataToPredict,
        prediction: predictionResponse.prediction,
        probability: predictionResponse.probability,
      }).catch((e) => console.error('Failed to save history:', e));

      recordAssessment({
        condition: 'DIABETES',
        inputPayload: dataToPredict,
        modelVersion: 'diabetes-v1.0',
        probability: predictionResponse.probability,
        riskBand: predictionResponse.prediction === 1 ? 'HIGH' : 'LOW',
        observations: [
          { name: 'glucose', value: Number(dataToPredict.glucose || 0), unit: 'mg/dL' },
          { name: 'hba1c', value: Number(dataToPredict.hba1c || 0), unit: '%' },
          { name: 'bmi', value: Number(dataToPredict.bmi || 0), unit: 'kg/m²' },
        ],
      }).catch((e) => console.log('Assessment record:', e.message));
    } catch (_err) {
      setError('Failed to connect to the prediction engine. Check network or backend server status.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await triggerPrediction(formData);
  };

  const handleExtract = (extracted) => {
    setFormData((prev) => ({ ...prev, ...extracted }));
    setActiveTab('manual');
    setResult(null);
    setError('');
    setOcrExtracted(true);
  };

  const handleChatComplete = async (answers) => {
    setFormData((prev) => ({ ...prev, ...answers }));
    await triggerPrediction(answers);
  };

  const genderOptions = [
    { value: 'female', label: 'Female' },
    { value: 'male', label: 'Male' },
    { value: 'other', label: 'Other' },
  ];

  const smokingOptions = [
    { value: 'never', label: 'Never Smoked' },
    { value: 'former', label: 'Former Smoker' },
    { value: 'current', label: 'Current Smoker' },
    { value: 'ever', label: 'Ever Smoked' },
    { value: 'not current', label: 'Not Current Smoker' },
  ];

  const yesNoOptions = [
    { value: '1', label: 'Yes' },
    { value: '0', label: 'No' },
  ];

  const tabOptions = [
    { id: 'manual', label: 'Manual Entry', icon: Keyboard, badge: ocrExtracted ? 'Review Required' : null },
    { id: 'upload', label: 'Upload Lab Report', icon: FileText },
    { id: 'chat', label: 'Guided Assistant', icon: MessageSquare },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Diabetes Risk Assessment"
        subtitle="Calibrated glycemic risk stratification based on clinical biometrics and metabolic biomarkers."
        badge={{ label: 'Glycemic Identity: Cyan', status: 'processing' }}
      />

      {/* Input Mode Selector */}
      <div className="flex justify-center">
        <SegmentedTabs
          tabs={tabOptions}
          activeTab={activeTab}
          onChange={setActiveTab}
        />
      </div>

      {activeTab === 'upload' && <UploadReport onExtract={handleExtract} />}

      {activeTab === 'chat' && (
        <ChatBot
          initialData={formData}
          onComplete={handleChatComplete}
          questions={[
            { key: 'age', question: "What is the patient's age in years?" },
            { key: 'gender', question: "What is the patient's gender?", options: genderOptions },
            { key: 'bmi', question: 'What is their Body Mass Index (BMI)?' },
            { key: 'glucose', question: 'What is their fasting blood glucose level (mg/dL)?' },
            { key: 'hba1c', question: 'What is their HbA1c percentage (%)?' },
            { key: 'smokingHistory', question: 'What is their smoking history?', options: smokingOptions },
            { key: 'hypertension', question: 'Do they have a history of hypertension?', options: yesNoOptions },
            { key: 'heartDisease', question: 'Do they have a history of heart disease?', options: yesNoOptions },
          ]}
        />
      )}

      {activeTab === 'manual' && (
        <Surface variant="flat" accent="teal">
          {ocrExtracted && (
            <div className="flex items-start justify-between gap-3 bg-amber-500/10 border border-amber-500/30 text-amber-300 p-4 rounded-xl mb-6 text-xs">
              <div className="flex items-start gap-2">
                <StatusBadge label="Review required" status="attention" size="sm" />
                <p className="mt-0.5">
                  Values were extracted via document OCR. Please verify all fields carefully before submitting.
                </p>
              </div>
              <button onClick={() => setOcrExtracted(false)} className="text-amber-400 hover:text-amber-200">
                &times;
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-8">
            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800 flex items-center justify-between">
                <span>1. Patient Demographics</span>
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <InputField
                  label="Patient Name (Optional)"
                  name="patientName"
                  value={formData.patientName}
                  onChange={handleChange}
                  placeholder="John Doe"
                />

                <InputField
                  label="Age"
                  name="age"
                  type="number"
                  unit="Years"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="45"
                  min="1"
                  max="120"
                  required
                />

                <SelectField
                  label="Gender"
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  options={genderOptions}
                  required
                />

                <SelectField
                  label="Smoking History"
                  name="smokingHistory"
                  value={formData.smokingHistory}
                  onChange={handleChange}
                  options={smokingOptions}
                  required
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800">
                2. Metabolic Biomarkers
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <InputField
                  label="Body Mass Index (BMI)"
                  name="bmi"
                  type="number"
                  step="0.1"
                  unit="kg/m²"
                  value={formData.bmi}
                  onChange={handleChange}
                  placeholder="25.5"
                  min="10"
                  max="60"
                  required
                />

                <InputField
                  label="HbA1c Level"
                  name="hba1c"
                  type="number"
                  step="0.1"
                  unit="%"
                  value={formData.hba1c}
                  onChange={handleChange}
                  placeholder="5.5"
                  min="3"
                  max="15"
                  required
                />

                <InputField
                  label="Blood Glucose Level"
                  name="glucose"
                  type="number"
                  unit="mg/dL"
                  value={formData.glucose}
                  onChange={handleChange}
                  placeholder="100"
                  min="50"
                  max="300"
                  required
                />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800">
                3. Clinical History
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <SelectField
                  label="Hypertension History"
                  name="hypertension"
                  value={formData.hypertension}
                  onChange={handleChange}
                  options={yesNoOptions}
                  required
                />

                <SelectField
                  label="Heart Disease History"
                  name="heartDisease"
                  value={formData.heartDisease}
                  onChange={handleChange}
                  options={yesNoOptions}
                  required
                />
              </div>
            </div>

            {error && <ErrorState title="Calculation Failed" message={error} />}

            <div className="flex flex-col sm:flex-row items-center justify-end gap-4 pt-4 border-t border-slate-800">
              <Button
                type="button"
                variant="secondary"
                onClick={handleReset}
                disabled={loading}
                icon={RotateCcw}
                className="w-full sm:w-auto"
              >
                Reset Form
              </Button>
              <Button
                type="submit"
                disabled={loading}
                loading={loading}
                loadingLabel="Processing Assessment..."
                variant="primary"
                icon={Play}
                className="w-full sm:w-auto font-bold px-8"
              >
                Run Assessment
              </Button>
            </div>
          </form>
        </Surface>
      )}

      {result && (
        <div id="diabetes-result" className="scroll-mt-24">
          <ResultCard
            prediction={result.prediction}
            probability={result.probability}
            riskBand={result.risk_band || (result.prediction === 1 ? 'HIGH' : 'LOW')}
            explanation={result.explanation}
            suggestions={suggestions}
            extras={[
              { label: 'Patient', value: formData.patientName || 'Anonymous' },
              { label: 'BMI', value: `${formData.bmi} kg/m²` },
              { label: 'HbA1c', value: `${formData.hba1c}%` },
              { label: 'Glucose', value: `${formData.glucose} mg/dL` },
            ]}
          />
        </div>
      )}
    </div>
  );
};

export default DiabetesPrediction;
