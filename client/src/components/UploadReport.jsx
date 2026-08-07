import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import Button from './Button';
import Surface from './Surface';
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
      setError('Please select a PDF or image file first.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const data = await uploadReport(file);
      if (data && data.extracted_data && Object.keys(data.extracted_data).length > 0) {
        const count = Object.keys(data.extracted_data).length;
        const confidence = data.confidence || 'unknown';
        setSuccessMessage(
          `Successfully extracted ${count} biomarker field(s) with ${confidence} confidence. Review extracted values below.`
        );
        onExtract(data.extracted_data);
      } else {
        setError('No medical parameters could be extracted from this document. Please enter values manually.');
      }
    } catch (err) {
      setError(err.message || 'Failed to process document. Check backend service and OCR dependencies.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Surface variant="flat" className="w-full my-4">
      <div
        className="border-2 border-dashed border-slate-700 hover:border-teal-400 bg-slate-900/60 hover:bg-slate-900 transition-colors duration-200 rounded-2xl p-8 mb-6 text-center cursor-pointer relative"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload lab report file dropzone"
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
        }}
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
            <FileText className="w-12 h-12 text-teal-400 mb-3" />
            <p className="text-slate-100 font-semibold text-base">{file.name}</p>
            <p className="text-slate-400 text-xs mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            <p className="text-teal-400 text-xs mt-3 font-semibold hover:underline">Click to change file</p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <UploadCloud className="w-14 h-14 text-slate-500 mb-3" />
            <h3 className="text-base font-bold text-slate-200 mb-1">Upload Lab Report (PDF / Image)</h3>
            <p className="text-slate-400 text-xs mb-4 max-w-sm">
              Drag and drop your clinical lab document here, or click to browse files.
            </p>
            <span className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-xs font-semibold text-slate-300">
              Select Document
            </span>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-coral-500/10 border border-coral-500/30 text-coral-300 p-4 rounded-xl mb-6 text-xs font-medium">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 text-coral-400" />
          <p>{error}</p>
        </div>
      )}

      {successMessage && (
        <div className="flex items-center gap-3 bg-teal-500/10 border border-teal-500/30 text-teal-300 p-4 rounded-xl mb-6 text-xs font-medium">
          <CheckCircle className="w-4 h-4 flex-shrink-0 text-teal-400" />
          <p>{successMessage}</p>
        </div>
      )}

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-slate-900 p-4 rounded-xl border border-slate-800">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-teal-400 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider">Auto-Extraction Pipeline</h4>
            <p className="text-xs text-slate-400 mt-1">
              Extracted values are populated into the assessment form with an amber "Review required" indicator for verification.
            </p>
          </div>
        </div>
        <Button
          onClick={handleUpload}
          disabled={!file || loading}
          loading={loading}
          loadingLabel="Scanning OCR..."
          variant="primary"
          size="md"
          className="w-full sm:w-auto flex-shrink-0"
        >
          Extract Lab Data
        </Button>
      </div>
    </Surface>
  );
};

export default UploadReport;
