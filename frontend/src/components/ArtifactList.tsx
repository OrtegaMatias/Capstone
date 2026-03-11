import type { WeekArtifact } from '../api/types';

type Props = {
  artifacts: WeekArtifact[];
};

export default function ArtifactList({ artifacts }: Props) {
  return (
    <div className="panel">
      <div className="section-header">
        <h3>Artefactos</h3>
        <p className="muted">Salidas repo-locales generadas por semana.</p>
      </div>
      <div className="artifact-list">
        {artifacts.map((artifact) => (
          <article key={`${artifact.kind}-${artifact.path}`} className="artifact-card">
            <div>
              <strong>{artifact.label}</strong>
              <p className="muted">{artifact.path}</p>
            </div>
            <span className={`status-pill ${artifact.available ? 'ready' : 'pending'}`}>
              {artifact.available ? 'Disponible' : 'Pendiente'}
            </span>
          </article>
        ))}
      </div>
    </div>
  );
}
