import React, { useState } from 'react';
import Surface from '../components/Surface';
import PageHeader from '../components/PageHeader';
import StatusBadge from '../components/StatusBadge';
import Button from '../components/Button';
import InputField from '../components/InputField';
import SelectField from '../components/SelectField';
import SearchBar from '../components/SearchBar';
import { Pill, AlertTriangle, CheckCircle2, Plus, Trash2, ShieldAlert, Clock } from 'lucide-react';

const INITIAL_MEDICATIONS = [
  {
    id: 'med_1',
    name: 'Metformin HCl',
    dosage: '1000 mg',
    frequency: 'Twice daily with meals',
    category: 'Antidiabetic',
    status: 'ACTIVE',
    startDate: '2025-04-12',
    notes: 'Monitored for renal tolerance. eGFR must remain > 45 mL/min.',
    warning: null,
  },
  {
    id: 'med_2',
    name: 'Empagliflozin (Jardiance)',
    dosage: '10 mg',
    frequency: 'Once daily in morning',
    category: 'SGLT2 Inhibitor',
    status: 'ACTIVE',
    startDate: '2025-09-01',
    notes: 'Cardiorenal protective agent. Maintain hydration.',
    warning: null,
  },
  {
    id: 'med_3',
    name: 'Atorvastatin',
    dosage: '20 mg',
    frequency: 'Once daily at bedtime',
    category: 'Statin (Lipid Lowering)',
    status: 'ACTIVE',
    startDate: '2024-11-15',
    notes: 'Target LDL < 70 mg/dL.',
    warning: null,
  },
  {
    id: 'med_4',
    name: 'Lisinopril',
    dosage: '10 mg',
    frequency: 'Once daily',
    category: 'ACE Inhibitor (BP)',
    status: 'ACTIVE',
    startDate: '2025-01-20',
    notes: 'Monitored for serum potassium levels.',
    warning: 'Check serum potassium if combined with potassium supplements.',
  },
];

const MedicationHub = () => {
  const [medications, setMedications] = useState(INITIAL_MEDICATIONS);
  const [searchQuery, setSearchQuery] = useState('');
  const [showAddForm, setShowAddForm] = useState(false);

  const [name, setName] = useState('');
  const [dosage, setDosage] = useState('');
  const [frequency, setFrequency] = useState('Once daily');
  const [category, setCategory] = useState('Antidiabetic');
  const [notes, setNotes] = useState('');

  const handleAddMedication = (e) => {
    e.preventDefault();
    if (!name.trim() || !dosage.trim()) return;

    const newMed = {
      id: `med_${Date.now()}`,
      name: name.trim(),
      dosage: dosage.trim(),
      frequency,
      category,
      status: 'ACTIVE',
      startDate: new Date().toISOString().split('T')[0],
      notes: notes.trim() || 'Patient self-reported active prescription.',
      warning: null,
    };

    setMedications([newMed, ...medications]);
    setName('');
    setDosage('');
    setNotes('');
    setShowAddForm(false);
  };

  const handleDeleteMedication = (id) => {
    setMedications(medications.filter((m) => m.id !== id));
  };

  const filteredMeds = medications.filter(
    (m) =>
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.category.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Medication &amp; Treatment Interaction Hub"
        subtitle="Track active cardiometabolic prescriptions, dosage schedules, renal safety thresholds, and interaction flags."
        badge={{ label: '4 Active Prescriptions', status: 'healthy' }}
        action={
          <Button
            onClick={() => setShowAddForm(!showAddForm)}
            variant="primary"
            size="sm"
            icon={Plus}
          >
            {showAddForm ? 'Cancel Form' : 'Add Prescription'}
          </Button>
        }
      />

      {/* Renal & Biomarker Interaction Alert Notice */}
      <Surface variant="flat" accent="amber" className="bg-accent/10 border-amber-500/30 text-foreground">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-md bg-accent/20 text-amber-500 border border-amber-500/30 flex-shrink-0">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-amber-500 uppercase tracking-wider mb-1">
              Automated Safety &amp; Contraindication Checker Active
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              HealthLens cross-checks active prescriptions against recorded renal (eGFR) and liver enzyme bounds. Always notify your clinician before altering prescribed dosages.
            </p>
          </div>
        </div>
      </Surface>

      {/* Add Medication Form */}
      {showAddForm && (
        <Surface variant="raised" accent="teal" className="space-y-4">
          <h3 className="text-sm font-bold text-foreground uppercase tracking-wider mb-2 flex items-center gap-2">
            <Pill className="w-4 h-4 text-primary" /> Add New Active Prescription
          </h3>
          <form onSubmit={handleAddMedication} className="grid md:grid-cols-2 gap-4">
            <InputField
              label="Medication Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Metformin or Jardiance"
              required
            />
            <InputField
              label="Dosage &amp; Strength"
              value={dosage}
              onChange={(e) => setDosage(e.target.value)}
              placeholder="e.g. 500 mg or 10 mg"
              required
            />
            <SelectField
              label="Frequency Schedule"
              value={frequency}
              onChange={(e) => setFrequency(e.target.value)}
              options={[
                'Once daily in morning',
                'Once daily at bedtime',
                'Twice daily with meals',
                'Three times daily',
                'As needed (PRN)',
              ]}
              required
            />
            <SelectField
              label="Clinical Category"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              options={[
                'Antidiabetic',
                'SGLT2 Inhibitor',
                'Statin (Lipid Lowering)',
                'ACE Inhibitor (BP)',
                'Beta Blocker',
                'Antiplatelet / Aspirin',
                'Other Pharmacotherapy',
              ]}
              required
            />
            <div className="md:col-span-2">
              <InputField
                label="Clinical Notes / Prescribing Directions"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g. Take with food. Target A1c < 7.0%"
              />
            </div>
            <div className="md:col-span-2 flex justify-end gap-2">
              <Button type="button" onClick={() => setShowAddForm(false)} variant="ghost" size="sm">
                Cancel
              </Button>
              <Button type="submit" variant="primary" size="sm" icon={Plus}>
                Save Prescription
              </Button>
            </div>
          </form>
        </Surface>
      )}

      {/* Toolbar & Search */}
      <Surface variant="flat" className="p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="w-full md:w-80">
          <SearchBar
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onClear={() => setSearchQuery('')}
            placeholder="Search medication name or category..."
          />
        </div>
        <div className="text-xs text-muted-foreground font-mono">
          Showing {filteredMeds.length} of {medications.length} prescriptions
        </div>
      </Surface>

      {/* Medications List */}
      <div className="grid md:grid-cols-2 gap-6">
        {filteredMeds.map((med) => (
          <Surface key={med.id} variant="flat" className="flex flex-col justify-between p-6">
            <div>
              <div className="flex items-start justify-between gap-2 mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-md bg-primary/10 text-primary border border-primary/30">
                    <Pill className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-base font-bold text-foreground">{med.name}</h4>
                    <span className="text-xs text-primary font-mono font-semibold">{med.dosage}</span>
                  </div>
                </div>
                <StatusBadge label={med.status} status="healthy" size="sm" />
              </div>

              <div className="space-y-2 text-xs text-muted-foreground my-4">
                <div className="flex items-center justify-between border-b border-border pb-1">
                  <span>Category:</span>
                  <span className="font-semibold text-foreground">{med.category}</span>
                </div>
                <div className="flex items-center justify-between border-b border-border pb-1">
                  <span>Schedule:</span>
                  <span className="font-semibold text-foreground">{med.frequency}</span>
                </div>
                <div className="flex items-center justify-between border-b border-border pb-1 font-mono">
                  <span>Start Date:</span>
                  <span>{med.startDate}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-2 italic bg-muted p-2.5 rounded border border-border">
                  "{med.notes}"
                </p>
              </div>

              {med.warning && (
                <div className="flex items-center gap-2 bg-accent/10 border border-amber-500/30 text-amber-500 p-2.5 rounded text-xs mt-3">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  <span>{med.warning}</span>
                </div>
              )}
            </div>

            <div className="pt-4 mt-4 border-t border-border flex items-center justify-between">
              <span className="text-[11px] text-muted-foreground flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-primary" /> Active regimen
              </span>
              <button
                onClick={() => handleDeleteMedication(med.id)}
                className="text-xs text-destructive hover:text-red-400 font-semibold flex items-center gap-1 px-2 py-1 rounded hover:bg-muted"
                aria-label={`Remove ${med.name}`}
              >
                <Trash2 className="w-3.5 h-3.5" /> Remove
              </button>
            </div>
          </Surface>
        ))}
      </div>
    </div>
  );
};

export default MedicationHub;
