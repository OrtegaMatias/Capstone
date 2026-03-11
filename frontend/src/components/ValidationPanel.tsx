import type { UploadResponse } from '../api/types';
import WarningChips from './WarningChips';

type Props = {
  payload: UploadResponse | null;
};

export default function ValidationPanel({ payload }: Props) {
  if (!payload) {
    return (
      <section className="panel">
        <h3>Validaciones</h3>
        <p className="muted">Después de subir archivos verás schema detectado, preview y warnings.</p>
      </section>
    );
  }

  return (
    <section className="panel">
      <h3>Dataset {payload.dataset_id}</h3>
      <p className="muted">
        has_in: {String(payload.has_in)} | has_out: {String(payload.has_out)} | has_target: {String(payload.has_target)}
      </p>
      <h4>Schema detectado</h4>
      <pre className="code-block">{JSON.stringify(payload.schema, null, 2)}</pre>
      <h4>Warnings</h4>
      <WarningChips warnings={payload.warnings} />
    </section>
  );
}
