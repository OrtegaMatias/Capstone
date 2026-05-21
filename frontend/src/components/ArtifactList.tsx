import type { WeekArtifact } from '../api/types';

type Props = {
  artifacts: WeekArtifact[];
  weekId?: string;
};

function shouldHideArtifact(artifact: WeekArtifact, weekId?: string): boolean {
  if (weekId !== 'week-3') return false;
  return (
    artifact.path.includes('official_model.') ||
    artifact.path.includes('cplex_segregations.csv') ||
    artifact.path.includes('cplex_ml_signals.csv')
  );
}

export default function ArtifactList({ artifacts, weekId }: Props) {
  const visibleArtifacts = artifacts.filter((artifact) => !shouldHideArtifact(artifact, weekId));

  return (
    <div className="panel">
      <div className="section-header">
        <h3>Artefactos</h3>
        <p className="muted">Salidas repo-locales generadas por semana.</p>
      </div>
      <div className="artifact-list">
        {visibleArtifacts.map((artifact) => (
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
