import React, { useState } from 'react';
import GlassCard from '../components/GlassCard';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import Button from '../components/Button';
import ResultCard from '../components/ResultCard';
import { predictDiabetes } from '../services/api';
import { Activity } from 'lucide-react';

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
        glucose: ''
    });

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

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
            glucose: ''
        });
        setResult(null);
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const predictionResponse = await predictDiabetes({
                ...formData,
                age: Number(formData.age),
                bmi: Number(formData.bmi),
                hba1c: Number(formData.hba1c),
                glucose: Number(formData.glucose)
            });

            setResult(predictionResponse);
        } catch (err) {
            setError('Failed to connect to the prediction server. Please try again later.');
        } finally {
            setLoading(false);
        }
    };

    const genderOptions = [
        { value: 'female', label: 'Female' },
        { value: 'male', label: 'Male' },
        { value: 'other', label: 'Other' }
    ];

    const smokingOptions = [
        { value: 'never', label: 'Never Smoked' },
        { value: 'former', label: 'Former Smoker' },
        { value: 'current', label: 'Current Smoker' },
        { value: 'ever', label: 'Ever Smoked' },
        { value: 'not current', label: 'Not Current Smoker' }
    ];

    const yesNoOptions = [
        { value: '1', label: 'Yes' },
        { value: '0', label: 'No' }
    ];

    return (
        <div className="max-w-4xl mx-auto animate-fade-in-up pb-12">
            <div className="text-center mb-10">
                <div className="bg-blue-500/10 w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4 text-blue-600 border border-blue-200">
                    <Activity size={32} />
                </div>
                <h1 className="text-3xl md:text-5xl font-black mb-4 text-slate-800">Diabetes Risk Assessment</h1>
                <p className="text-slate-600 max-w-2xl mx-auto">
                    Enter patient biometrics and medical history below for a comprehensive metabolic risk evaluation.
                </p>
            </div>

            <GlassCard className="mb-8">
                <form onSubmit={handleSubmit}>
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Patient Details</h2>

                    <div className="grid md:grid-cols-2 gap-6 mb-8">
                        <InputField
                            label="Patient Name (Optional)"
                            name="patientName"
                            value={formData.patientName}
                            onChange={handleChange}
                            placeholder="John Doe"
                        />

                        <InputField
                            label="Age (Years)"
                            name="age"
                            type="number"
                            value={formData.age}
                            onChange={handleChange}
                            placeholder="45"
                            min="1" max="120"
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

                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Medical Metrics</h2>

                    <div className="grid md:grid-cols-2 gap-6 mb-8">
                        <InputField
                            label="BMI (Body Mass Index)"
                            name="bmi"
                            type="number"
                            step="0.1"
                            value={formData.bmi}
                            onChange={handleChange}
                            placeholder="25.5"
                            min="10" max="60"
                            required
                        />

                        <InputField
                            label="HbA1c Level (%)"
                            name="hba1c"
                            type="number"
                            step="0.1"
                            value={formData.hba1c}
                            onChange={handleChange}
                            placeholder="5.5"
                            min="3" max="15"
                            required
                        />

                        <InputField
                            label="Blood Glucose Level (mg/dL)"
                            name="glucose"
                            type="number"
                            value={formData.glucose}
                            onChange={handleChange}
                            placeholder="100"
                            min="50" max="300"
                            required
                        />
                    </div>

                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-borderLight pb-2">Medical History</h2>

                    <div className="grid md:grid-cols-2 gap-6 mb-8">
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

                    {error && (
                        <div className="bg-danger/20 border border-danger p-4 rounded-lg mb-6 text-danger text-center">
                            {error}
                        </div>
                    )}

                    <div className="flex gap-4 pt-4">
                        <Button type="button" variant="secondary" onClick={handleReset} disabled={loading}>
                            Reset Form
                        </Button>
                        <Button type="submit" disabled={loading}>
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Processing...
                                </span>
                            ) : 'Initiate Scan'}
                        </Button>
                    </div>
                </form>
            </GlassCard>

            {result && (
                <div id="diabetes-result" className="scroll-mt-24">
                    <ResultCard
                        prediction={result.prediction}
                        probability={result.probability}
                        riskLevel={result.risk_level}
                        extras={[
                            { label: 'Patient', value: formData.patientName || 'Anonymous' },
                            { label: 'BMI', value: formData.bmi },
                            { label: 'HbA1c', value: `${formData.hba1c}%` }
                        ]}
                    />
                </div>
            )}
        </div>
    );
};

export default DiabetesPrediction;
