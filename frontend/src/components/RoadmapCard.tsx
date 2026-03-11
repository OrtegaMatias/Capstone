import { Link } from 'react-router-dom';

import type { WeekSummary } from '../api/types';

type Props = {
  week: WeekSummary;
};

export default function RoadmapCard({ week }: Props) {
  return (
    <article className="roadmap-card">
      <div className="roadmap-card-top">
        <span className="roadmap-kicker">Semana {week.week_number}</span>
        <span className={`status-pill ${week.status === 'active' ? 'ready' : 'pending'}`}>
          {week.status === 'active' ? 'Activa' : 'Base'}
        </span>
      </div>
      <h3>{week.stage_name}</h3>
      <p>{week.summary}</p>
      <div className="roadmap-meta">
        <span>{week.analysis_available.length} modulos</span>
        <span>{week.report_available ? 'reporte listo' : 'reporte inicial'}</span>
      </div>
      <Link to={`/weeks/${week.week_id}`} className="card-link">
        Abrir semana
      </Link>
    </article>
  );
}
