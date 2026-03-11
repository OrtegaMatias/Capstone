import type { WeekReportSummary } from '../api/types';

type Props = {
  report: WeekReportSummary | undefined;
  loading: boolean;
  refreshing: boolean;
  onRefresh: () => Promise<void>;
};

function triggerDownload(filename: string, content: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function ReportPanel({ report, loading, refreshing, onRefresh }: Props) {
  return (
    <div className="panel">
      <div className="section-header">
        <div>
          <h3>Reporte semanal</h3>
          <p className="muted">Preview del entregable automatico Markdown/HTML.</p>
        </div>
        <div className="hero-actions">
          <button type="button" onClick={() => onRefresh()} disabled={refreshing}>
            {refreshing ? 'Actualizando...' : 'Regenerar'}
          </button>
          <button
            type="button"
            onClick={() => report && triggerDownload(`${report.week_id}.md`, report.markdown_content, 'text/markdown')}
            disabled={!report}
          >
            Descargar MD
          </button>
          <button
            type="button"
            onClick={() => report && triggerDownload(`${report.week_id}.html`, report.html_content, 'text/html')}
            disabled={!report}
          >
            Descargar HTML
          </button>
        </div>
      </div>
      {loading && !report ? <p className="muted">Cargando reporte...</p> : null}
      {report ? (
        <>
          <small className="muted">Ultima actualizacion: {report.updated_at ?? 'sin datos'}</small>
          <pre className="code-block report-preview">{report.markdown_content}</pre>
        </>
      ) : null}
    </div>
  );
}
