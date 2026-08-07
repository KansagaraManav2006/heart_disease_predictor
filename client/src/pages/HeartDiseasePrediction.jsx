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
import { predictHeartDisease, saveHistory, recordAssessment } from '../services/api';
import { generateSuggestions } from '../utils/suggestionEngine';
import { Heart, FileText, Keyboard, MessageSquare, RotateCcw, Play } from 'lucide-react';

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
    active: '1',
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
      active: '1',
    });
    setResult(null);
    setError('');
    setOcrExtracted(false);
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

      let userId = localStorage.getItem('userId');
      if (!userId) {
        userId = 'user_' + Date.now();
        localStorage.setItem('userId', userId);
      }

      saveHistory({
        userId,
        userName: dataToPredict.patientName || 'Anonymous',
        type: 'heart',
        inputs: dataToPredict,
        prediction: predictionResponse.prediction,
        probability: predictionResponse.probability,
      }).catch((e) => console.error('Failed to save history:', e));

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
      }).catch((e) => console.log('Assessment record:', e.message));
    } catch (_err) {
      setError('Failed to connect to the prediction engine. Check backend server connection.');
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
    { value: 'male', label: 'Male' },
    { value: 'female', label: 'Female' },
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
        title="Cardiac Risk Assessment"
        subtitle="Calibrated cardiovascular risk evaluation based on vital statistics, lipids, and lifestyle factors."
        badge={{ label: 'Condition Identity: Violet', status: 'secondary' }}
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
            { key: 'height_cm', question: 'What is their height in centimeters?' },
            { key: 'weight_kg', question: 'What is their weight in kilograms?' },
            { key: 'systolic_bp', question: 'What is their Systolic Blood Pressure (mmHg)?' },
            { key: 'diastolic_bp', question: 'What is their Diastolic Blood Pressure (mmHg)?' },
            { key: 'cholesterol', question: 'What is their Serum Cholesterol level (mg/dL)?' },
            { key: 'glucose', question: 'What is their Fasting Blood Glucose (mg/dL)?' },
            { key: 'smoke', question: 'Do they currently smoke?', options: yesNoOptions },
            { key: 'alco', question: 'Do they consume alcohol regularly?', options: yesNoOptions },
            { key: 'active', question: 'Are they physically active?', options: yesNoOptions },
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
                  Values were extracted via document OCR. Please verify height, weight, and blood pressure before submitting.
                </p>
              </div>
              <button onClick={() => setOcrExtracted(false)} className="text-amber-400 hover:text-amber-200">
                &times;
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-8">
            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800">
                1. Biometric Demographics
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <InputField
                  label="Patient Name (Optional)"
                  name="patientName"
                  value={formData.patientName}
                  onChange={handleChange}
                  placeholder="Jane Doe"
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

                <div className="grid grid-cols-2 gap-4">
                  <InputField
                    label="Height"
                    name="height_cm"
                    type="number"
                    unit="cm"
                    value={formData.height_cm}
                    onChange={handleChange}
                    placeholder="170"
                    min="120"
                    max="220"
                    required
                  />
                  <InputField
                    label="Weight"
                    name="weight_kg"
                    type="number"
                    step="0.1"
                    unit="kg"
                    value={formData.weight_kg}
                    onChange={handleChange}
                    placeholder="70"
                    min="30"
                    max="200"
                    required
                  />
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 pb-2 border-b border-slate-800">
                2. Cardiovascular Biomarkers
              </h2>
              <div className="grid md:grid-cols-2 gap-6">
                <InputField
                  label="Systolic Blood Pressure"
                  name="systolic_bp"
                  type="number"
                  unit="mmHg"
                  value={formData.systolic_bp}
                  onChange={handleChange}
                  placeholder="120"
                  min="80"
                  max="200"
                  required
                />

                <InputField
                  label="Diastolic Blood Pressure"
                  name="diastolic_bp"
                  type="number"
                  unit="mmHg"
                  value={formData.diastolic_bp}
                  onChange={handleChange}
                  placeholder="80"
                  min="50"
                  max="120"
                  required
                />

                <InputField
                  label="Serum Cholesterol Level"
                  name="cholesterol"
                  type="number"
                  unit="mg/dL"
                  value={formData.cholesterol}
                  onChange={handleChange}
                  placeholder="200"
                  min="100"
                  max="400"
                  required
                />

                <InputField
                  label="Fasting Blood Glucose"
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
                3. Lifestyle Risk Factors
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                <SelectField
                  label="Current Smoker"
                  name="smoke"
                  value={formData.smoke}
                  onChange={handleChange}
                  options={yesNoOptions}
                  required
                />

                <SelectField
                  label="Alcohol Intake"
                  name="alco"
                  value={formData.alco}
                  onChange={handleChange}
                  options={yesNoOptions}
                  required
                />

                <SelectField
                  label="Physically Active"
                  name="active"
                  value={formData.active}
                  onChange={handleChange}
                  options={yesNoOptions}
                  required
                />
              </div>
            </div>

            {error && <ErrorState title="Calculation Error" message={error} />}

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
        <div id="heart-result" className="scroll-mt-24">
          <ResultCard
            prediction={result.prediction}
            probability={result.probability}
            riskBand={result.risk_band || (result.prediction === 1 ? 'HIGH' : 'LOW')}
            explanation={result.explanation}
            suggestions={suggestions}
            extras={[
              { label: 'Patient', value: formData.patientName || 'Anonymous' },
              { label: 'Blood Pressure', value: `${formData.systolic_bp}/${formData.diastolic_bp} mmHg` },
              { label: 'Cholesterol', value: `${formData.cholesterol} mg/dL` },
              { label: 'Calculated BMI', value: result.bmi_val ? `${result.bmi_val}` : 'N/A' },
            ]}
          />
        </div>
      )}
    </div>
  );
};

export default HeartDiseasePrediction;
