import { FormEvent, useState } from 'react';

type Props = {
  onSubmit: (inFile: File | null, outFile: File | null) => Promise<void>;
  loading: boolean;
};

export default function UploadForm({ onSubmit, loading }: Props) {
  const [inFile, setInFile] = useState<File | null>(null);
  const [outFile, setOutFile] = useState<File | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await onSubmit(inFile, outFile);
  };

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <h2>Upload CSVs</h2>
      <p className="muted">Sube uno o ambos archivos para iniciar el pipeline ETL y análisis.</p>

      <label className="field">
        <span>Grupo1_in.csv (opcional)</span>
        <input type="file" accept=".csv" onChange={(e) => setInFile(e.target.files?.[0] ?? null)} />
      </label>

      <label className="field">
        <span>Grupo1_out.csv (opcional)</span>
        <input type="file" accept=".csv" onChange={(e) => setOutFile(e.target.files?.[0] ?? null)} />
      </label>

      <button type="submit" disabled={loading}>
        {loading ? 'Procesando...' : 'Crear Dataset'}
      </button>
    </form>
  );
}
