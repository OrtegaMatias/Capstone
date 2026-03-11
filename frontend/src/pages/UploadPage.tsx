import { useMutation } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';

import { uploadDataset } from '../api/client';
import type { UploadResponse } from '../api/types';
import UploadForm from '../components/UploadForm';
import ValidationPanel from '../components/ValidationPanel';
import { useDatasetId } from '../hooks/useDatasetId';

export default function UploadPage() {
  const navigate = useNavigate();
  const { setDatasetId } = useDatasetId();

  const mutation = useMutation<UploadResponse, Error, { inFile: File | null; outFile: File | null }>({
    mutationFn: ({ inFile, outFile }) => uploadDataset(inFile, outFile),
    onSuccess: (payload) => {
      setDatasetId(payload.dataset_id);
    },
  });

  const handleSubmit = async (inFile: File | null, outFile: File | null) => {
    const payload = await mutation.mutateAsync({ inFile, outFile });
    if (payload.dataset_id) {
      navigate('/dashboard');
    }
  };

  return (
    <main className="page">
      <header className="hero">
        <h1>Capstone ETL + EDA Workbench</h1>
        <p>Upload, valida, analiza y documenta hallazgos en un pipeline profesional extensible.</p>
      </header>

      <section className="grid-2">
        <UploadForm onSubmit={handleSubmit} loading={mutation.isPending} />
        <ValidationPanel payload={mutation.data ?? null} />
      </section>

      {mutation.isError ? <p className="error">{mutation.error.message}</p> : null}
    </main>
  );
}
