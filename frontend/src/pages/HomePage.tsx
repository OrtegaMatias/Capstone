import { useQuery } from '@tanstack/react-query';

import { fetchFrameworkSummary } from '../api/client';
import RoadmapCard from '../components/RoadmapCard';

export default function HomePage() {
  const frameworkQuery = useQuery({
    queryKey: ['framework-summary'],
    queryFn: fetchFrameworkSummary,
  });

  if (frameworkQuery.isLoading) {
    return <main className="page"><p className="muted">Cargando roadmap del framework...</p></main>;
  }

  if (frameworkQuery.isError || !frameworkQuery.data) {
    return (
      <main className="page">
        <p className="error">No se pudo cargar el framework semanal. Revisa el backend y el manifiesto.</p>
      </main>
    );
  }

  const framework = frameworkQuery.data;

  return (
    <main className="page">
      <header className="hero hero-slab">
        <div>
          <span className="eyebrow">Framework académico repo-local</span>
          <h1>{framework.framework_name}</h1>
          <p>{framework.summary}</p>
        </div>
        <div className="hero-note panel">
          <strong>Workspace</strong>
          <p className="muted">{framework.workspace_root}</p>
          <small className="muted">Generado: {new Date(framework.generated_at).toLocaleString()}</small>
        </div>
      </header>

      <section className="roadmap-grid">
        {framework.weeks.map((week) => (
          <RoadmapCard key={week.week_id} week={week} />
        ))}
      </section>
    </main>
  );
}
