import { useEffect, useState } from 'react';

type Props = {
  initialContent: string;
  updatedAt?: string | null;
  onSave: (content: string) => Promise<void>;
  loading: boolean;
};

export default function NotesPanel({ initialContent, updatedAt, onSave, loading }: Props) {
  const [text, setText] = useState(initialContent);

  useEffect(() => {
    setText(initialContent);
  }, [initialContent]);

  return (
    <div className="panel">
      <h3>Notas de Informe</h3>
      <p className="muted">Espacio para registrar hallazgos, hipótesis y decisiones del análisis.</p>
      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={14} />
      <div className="notes-footer">
        <small className="muted">Última actualización: {updatedAt ?? 'sin guardar'}</small>
        <button disabled={loading} onClick={() => onSave(text)}>
          {loading ? 'Guardando...' : 'Guardar notas'}
        </button>
      </div>
    </div>
  );
}
