import React, { useState } from 'react';
import GlassCard from '../components/GlassCard';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import Button from '../components/Button';
import ResultCard from '../components/ResultCard';
import { predictHeartDisease } from '../services/api';
import { Heart } from 'lucide-react';

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

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const payload = {
                ...formData,
                age: Number(formData.age),
                height_cm: Number(formData.height_cm),
                weight_kg: Number(formData.weight_kg),
                systolic_bp: Number(formData.systolic_bp),
                diastolic_bp: Number(formData.diastolic_bp),
                cholesterol: Number(formData.cholesterol),
                glucose: Number(formData.glucose),
                smoke: formData.smoke === '1',
                alco: formData.alco === '1',
                active: formData.active === '1',
            };

            const predictionResponse = await predictHeartDisease(payload);
            setResult(predictionResponse);
        } catch (err) {
            setError('Failed to connect to the prediction server. Please try again later.');
        } finally {
            setLoading(false);
        }
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

            <GlassCard className="mb-8 border-t-4 border-t-red-500/50">
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

            {result && (
                <div id="heart-result" className="scroll-mt-24">
                    <ResultCard
                        prediction={result.prediction}
                        probability={result.probability}
                        riskLevel={result.risk_level}
                        extras={[
                            { label: 'Patient', value: formData.patientName || 'Anonymous' },
                            { label: 'BMI', value: result.bmi_val ? `${result.bmi_val}` : 'N/A' },
                            { label: 'Blood Pressure', value: `${formData.systolic_bp}/${formData.diastolic_bp}` }
                        ]}
                    />
                </div>
            )}
        </div>
    );
};

export default HeartDiseasePrediction;
