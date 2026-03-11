import type { WarningItem } from '../api/types';

type Props = {
  warnings: WarningItem[];
};

export default function WarningChips({ warnings }: Props) {
  if (!warnings.length) {
    return <p className="muted">Sin alertas relevantes.</p>;
  }

  return (
    <div className="chips">
      {warnings.map((warning, index) => (
        <div className={`chip ${warning.severity}`} key={`${warning.code}-${index}`}>
          <strong>{warning.severity === 'warning' ? 'Advertencia' : 'Nota'}</strong>
          <span>{warning.message}</span>
          {warning.suggestion ? <small>{warning.suggestion}</small> : null}
        </div>
      ))}
    </div>
  );
}
