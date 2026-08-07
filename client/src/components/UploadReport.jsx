import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import Button from './Button';
import { uploadReport } from '../services/api';

const UploadReport = ({ onExtract }) => {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setError('');
            setSuccessMessage('');
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) {
            setFile(droppedFile);
            setError('');
            setSuccessMessage('');
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a file first.');
            return;
        }

        setLoading(true);
        setError('');
        
        try {
            const data = await uploadReport(file);
            if (data && data.extracted_data && Object.keys(data.extracted_data).length > 0) {
                const count = Object.keys(data.extracted_data).length;
                const confidence = data.confidence || 'unknown';
                const warningText = data.warnings?.length ? ` Note: ${data.warnings.join(' ')}` : '';
                setSuccessMessage(
                    `Extracted ${count} field(s) with ${confidence} confidence. Please review all values before submitting.${warningText}`
                );
                onExtract(data.extracted_data);
            } else {
                setError('No medical parameters could be extracted from this file. Please enter values manually.');
            }
        } catch (err) {
            setError(err.message || 'Failed to process the document. Server error or OCR dependency issue.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full">
            <div 
                className="border-2 border-dashed border-slate-300 hover:border-blue-400 bg-slate-50 hover:bg-blue-50/50 transition-colors duration-200 rounded-2xl p-8 mb-6 text-center cursor-pointer relative"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onClick={() => fileInputRef.current?.click()}
            >
                <input 
                    type="file" 
                    className="hidden" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept="image/*,application/pdf"
                />
                
                {file ? (
                    <div className="flex flex-col items-center">
                        <FileText className="w-12 h-12 text-blue-500 mb-3" />
                        <p className="text-slate-700 font-semibold text-lg">{file.name}</p>
                        <p className="text-slate-500 text-sm mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                        <p className="text-blue-500 text-sm mt-3 font-medium cursor-pointer hover:underline">Click to change file</p>
                    </div>
                ) : (
                    <div className="flex flex-col items-center">
                        <UploadCloud className="w-16 h-16 text-slate-400 mb-4" />
                        <h3 className="text-lg font-bold text-slate-700 mb-2">Upload Lab Report</h3>
                        <p className="text-slate-500 mb-4 max-w-sm">Drag and drop your PDF or Image report here, or click to browse files.</p>
                        <span className="bg-white border rounded-full px-4 py-1 text-sm font-semibold text-slate-600 shadow-sm">Select File</span>
                    </div>
                )}
            </div>

            {error && (
                <div className="flex items-center gap-3 bg-red-50 text-red-700 p-4 rounded-xl mb-6 border border-red-200">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    <p className="text-sm font-medium">{error}</p>
                </div>
            )}

            {successMessage && (
                <div className="flex items-center gap-3 bg-green-50 text-green-700 p-4 rounded-xl mb-6 border border-green-200">
                    <CheckCircle className="w-5 h-5 flex-shrink-0" />
                    <p className="text-sm font-medium">{successMessage}</p>
                </div>
            )}
            
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-blue-50 p-4 rounded-xl border border-blue-100 mb-8">
                <div className="flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div>
                        <h4 className="text-sm font-bold text-blue-900">How Auto-Extract Works</h4>
                        <p className="text-xs text-blue-800 mt-1">Our AI scans the document for standard markers like Glucose, HbA1c, and BP. Extracted values will populate the Manual form where you can verify them before running the prediction.</p>
                    </div>
                </div>
                <Button onClick={handleUpload} disabled={!file || loading} className="w-full sm:w-auto flex-shrink-0 min-w-[140px]">
                    {loading ? 'Scanning...' : 'Extract Data'}
                </Button>
            </div>
        </div>
    );
};

export default UploadReport;
