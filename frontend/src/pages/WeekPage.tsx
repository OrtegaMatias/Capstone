import { useEffect, useRef, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';

import {
  fetchWeekArtifactText,
  fetchWeekConfig,
  fetchWeekClustering,
  fetchWeekEDA,
  fetchWeekMlCached,
  fetchWeekNotes,
  fetchWeekOptimizationModel,
  fetchWeekPreview,
  fetchWeekReport,
  getApiBaseUrl,
  refreshWeekReport,
  saveWeekNotes,
} from '../api/client';
import type { MlEvaluationSummary } from '../api/types';
import ArtifactList from '../components/ArtifactList';
import AnovaTable from '../components/AnovaTable';
import BoxplotPanel from '../components/BoxplotPanel';
import DataTable from '../components/DataTable';
import MetricCards from '../components/MetricCards';
import MlClassificationPanel from '../components/MlClassificationPanel';
import MlOverviewPanel from '../components/MlOverviewPanel';
import MultipleRegressionPanel from '../components/MultipleRegressionPanel';
import NotesPanel from '../components/NotesPanel';
import OptimizationModelPanel from '../components/OptimizationModelPanel';
import ReportPanel from '../components/ReportPanel';
import Week1AcademicEdaPanel from '../components/Week1AcademicEdaPanel';
import WarningChips from '../components/WarningChips';
import { downloadPageAsPdf } from '../utils/pdfExport';

export default function WeekPage() {
  const { weekId } = useParams();
  const isWeek1 = weekId === 'week-1';
  const isWeek4 = weekId === 'week-4';
  const pageRef = useRef<HTMLElement>(null);
  const [pdfStatus, setPdfStatus] = useState<string | null>(null);
  const [mlData, setMlData] = useState<MlEvaluationSummary | null>(null);
  const [mlRunning, setMlRunning] = useState(false);
  const [mlProgress, setMlProgress] = useState<{ current: number; total: number; message: string } | null>(null);
  const [mlError, setMlError] = useState<string | null>(null);

  const handleDownloadPdf = async () => {
    if (!pageRef.current) return;
    setPdfStatus('Iniciando...');
    try {
      await downloadPageAsPdf(
        pageRef.current,
        `semana-${week?.week_number ?? weekId}-reporte-completo.pdf`,
        setPdfStatus
      );
    } catch (err) {
      console.error('PDF export failed', err);
      setPdfStatus('Error al generar PDF');
    } finally {
      setTimeout(() => setPdfStatus(null), 2500);
    }
  };

  const weekQuery = useQuery({
    queryKey: ['week-config', weekId],
    queryFn: () => fetchWeekConfig(weekId as string),
    enabled: !!weekId,
  });

  const inheritedWeek3Query = useQuery({
    queryKey: ['week-config', 'week-3', 'inherited-for-week-4'],
    queryFn: () => fetchWeekConfig('week-3'),
    enabled: isWeek4,
  });

  const inheritedModelCodeQuery = useQuery({
    queryKey: ['week-4-inherited-cplex-code'],
    queryFn: async () => {
      const [mod, dat] = await Promise.all([
        fetchWeekArtifactText('week-3', 'official_model.mod'),
        fetchWeekArtifactText('week-3', 'official_model.dat'),
      ]);
      return { mod, dat };
    },
    enabled: isWeek4,
  });

  const previewQuery = useQuery({
    queryKey: ['week-preview', weekId],
    queryFn: () => fetchWeekPreview(weekId as string, 20),
    enabled: !!weekId && !isWeek1 && !!weekQuery.data?.analysis_available.includes('preview'),
  });

  const edaQuery = useQuery({
    queryKey: ['week-eda', weekId],
    queryFn: () => fetchWeekEDA(weekId as string),
    enabled: !!weekId && (isWeek1 || !!weekQuery.data?.analysis_available.includes('eda')),
  });

  const mlCachedQuery = useQuery({
    queryKey: ['week-ml-cached', weekId],
    queryFn: () => fetchWeekMlCached(weekId as string),
    enabled: !!weekId && !!weekQuery.data?.analysis_available.includes('ml_overview'),
  });

  const optimizationQuery = useQuery({
    queryKey: ['week-optimization', weekId],
    queryFn: () => fetchWeekOptimizationModel(weekId as string),
    enabled: !!weekId && !!weekQuery.data?.analysis_available.includes('optimization_model'),
  });

  useEffect(() => {
    if (mlCachedQuery.data) {
      setMlData(mlCachedQuery.data);
    }
  }, [mlCachedQuery.data]);

  const handleRunMl = () => {
    if (mlRunning || !weekId) return;
    setMlRunning(true);
    setMlProgress(null);
    setMlError(null);

    const baseUrl = getApiBaseUrl();
    const url = `${baseUrl}/weeks/${weekId}/ml/run`;

    fetch(url, { method: 'POST' }).then((response) => {
      if (!response.ok || !response.body) {
        setMlError('Error al conectar con el servidor');
        setMlRunning(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const processChunk = ({ done, value }: ReadableStreamReadResult<Uint8Array>): Promise<void> | void => {
        if (done) {
          setMlRunning(false);
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split('\n');
          let eventType = 'message';
          let data = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) eventType = line.slice(7);
            else if (line.startsWith('data: ')) data = line.slice(6);
          }
          if (eventType === 'done' && data) {
            try {
              const payload = JSON.parse(data) as MlEvaluationSummary;
              setMlData(payload);
              setMlProgress(null);
            } catch { /* ignore parse errors */ }
            setMlRunning(false);
            return;
          } else if (eventType === 'error' && data) {
            try {
              const err = JSON.parse(data) as { detail: string };
              setMlError(err.detail);
            } catch { /* ignore parse errors */ }
            setMlRunning(false);
            return;
          } else if (data) {
            try {
              const progress = JSON.parse(data) as { current: number; total: number; message: string };
              setMlProgress(progress);
            } catch { /* ignore parse errors */ }
          }
        }
        return reader.read().then(processChunk);
      };

      reader.read().then(processChunk);
    }).catch(() => {
      setMlError('Error de conexión');
      setMlRunning(false);
    });
  };

  const clusteringQuery = useQuery({
    queryKey: ['week-clustering', weekId],
    queryFn: () => fetchWeekClustering(weekId as string),
    enabled: isWeek1,
  });

  const notesQuery = useQuery({
    queryKey: ['week-notes', weekId],
    queryFn: () => fetchWeekNotes(weekId as string),
    enabled: !!weekId,
  });

  const reportQuery = useQuery({
    queryKey: ['week-report', weekId],
    queryFn: () => fetchWeekReport(weekId as string),
    enabled: !!weekId,
  });

  const saveNotesMutation = useMutation({
    mutationFn: (content: string) => saveWeekNotes(weekId as string, content),
    onSuccess: () => {
      notesQuery.refetch();
    },
  });

  const refreshReportMutation = useMutation({
    mutationFn: () => refreshWeekReport(weekId as string),
    onSuccess: () => {
      reportQuery.refetch();
    },
  });

  const week = weekQuery.data;
  const inheritedModelArtifacts = (inheritedWeek3Query.data?.artifacts ?? []).filter(
    (artifact) => artifact.path.includes('official_model.mod') || artifact.path.includes('official_model.dat')
  );



  if (!weekId) {
    return <Navigate to="/" replace />;
  }

  if (weekQuery.isError) {
    return <main className="page"><p className="error">No se pudo cargar la semana solicitada.</p></main>;
  }

  if (weekQuery.isLoading || !week) {
    return <main className="page"><p className="muted">Cargando semana...</p></main>;
  }

  return (
    <main className="page" ref={pageRef}>
      <header className="hero hero-slab">
        <div>
          <Link to="/" className="back-link">
            Volver al roadmap
          </Link>
          <div className="week-heading">
            <span className="eyebrow">Semana {week.week_number}</span>
            <h1>{week.stage_name}</h1>
          </div>
          <p>{week.description}</p>
          {week.academic_context ? (
            <p className="muted">
              Objetivo: {week.academic_context.objective} | Unidad: {week.academic_context.unit_of_observation}
            </p>
          ) : null}
        </div>
        <div className="hero-note panel">
          <span className={`status-pill ${week.status === 'active' ? 'ready' : 'pending'}`}>
            {week.status === 'active' ? 'Activa' : 'Base'}
          </span>
          <small className="muted">Seed IN: {week.seed_paths.in_file ?? 'n/a'}</small>
          <small className="muted">Seed OUT: {week.seed_paths.out_file ?? 'n/a'}</small>
          <button
            type="button"
            className="outline-btn"
            style={{ marginTop: '0.5rem', fontSize: '0.85rem', padding: '0.4rem 0.8rem', width: '100%' }}
            onClick={() => void handleDownloadPdf()}
            disabled={!!pdfStatus}
          >
            {pdfStatus ? `⏳ ${pdfStatus}` : '📄 Descargar PDF Completo'}
          </button>
        </div>
      </header>

      <section className="week-brief-grid">
        <div className="panel">
          <h3>Checklist</h3>
          <ul className="plain-list">
            {week.checklist.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <h3>Inputs esperados</h3>
          <ul className="plain-list">
            {week.expected_inputs.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="panel">
          <h3>Entregables</h3>
          <ul className="plain-list">
            {week.deliverables.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <ArtifactList artifacts={week.artifacts} weekId={week.week_id} />

      {isWeek4 ? (
        inheritedWeek3Query.isLoading ? (
          <section className="panel">
            <div className="section-header">
              <h3>Modelo Base Heredado</h3>
              <p className="muted">Cargando el `.mod` y el `.dat` definidos en la semana 3.</p>
            </div>
          </section>
        ) : inheritedWeek3Query.isError ? (
          <section className="panel">
            <div className="section-header">
              <h3>Modelo Base Heredado</h3>
              <p className="error">No se pudo cargar la referencia al modelo CPLEX de la semana 3.</p>
            </div>
          </section>
        ) : inheritedModelArtifacts.length > 0 ? (
          <section className="panel">
            <div className="section-header">
              <h3>Modelo Base Heredado</h3>
              <p className="muted">
                La semana 4 toma como base el modelo oficial ya generado en la semana 3. Aqui se muestran el
                archivo `.mod` y su `.dat` asociado.
              </p>
            </div>
            <div className="artifact-list">
              {inheritedModelArtifacts.map((artifact) => (
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
            {inheritedModelCodeQuery.isLoading ? (
              <div className="grid-2" style={{ marginTop: '1rem' }}>
                <section className="panel">
                  <h4 style={{ marginBottom: '0.75rem' }}>official_model.mod</h4>
                  <p className="muted">Cargando contenido del modelo...</p>
                </section>
                <section className="panel">
                  <h4 style={{ marginBottom: '0.75rem' }}>official_model.dat</h4>
                  <p className="muted">Cargando datos del modelo...</p>
                </section>
              </div>
            ) : inheritedModelCodeQuery.isError ? (
              <div className="panel" style={{ marginTop: '1rem' }}>
                <p className="error">No se pudo cargar el contenido del `.mod` y `.dat` heredados.</p>
              </div>
            ) : inheritedModelCodeQuery.data ? (
              <div className="grid-2" style={{ marginTop: '1rem' }}>
                <section className="panel">
                  <h4 style={{ marginBottom: '0.75rem' }}>{inheritedModelCodeQuery.data.mod.filename}</h4>
                  <pre className="code-block" style={{ maxHeight: '30rem', margin: 0 }}>
                    {inheritedModelCodeQuery.data.mod.content}
                  </pre>
                </section>
                <section className="panel">
                  <h4 style={{ marginBottom: '0.75rem' }}>{inheritedModelCodeQuery.data.dat.filename}</h4>
                  <pre className="code-block" style={{ maxHeight: '30rem', margin: 0 }}>
                    {inheritedModelCodeQuery.data.dat.content}
                  </pre>
                </section>
              </div>
            ) : null}
          </section>
        ) : (
          <section className="panel">
            <div className="section-header">
              <h3>Modelo Base Heredado</h3>
              <p className="muted">La semana 3 no expone todavia el `.mod` y el `.dat` para heredarlos en semana 4.</p>
            </div>
          </section>
        )
      ) : null}

      {isWeek1 ? (
        edaQuery.isLoading ? (
          <section className="panel">
            <h3>EDA académico</h3>
            <p className="muted">Cargando análisis académico de la semana 1...</p>
          </section>
        ) : edaQuery.isError ? (
          <section className="panel">
            <h3>EDA académico</h3>
            <p className="error">No se pudo cargar el EDA académico de la semana 1.</p>
          </section>
        ) : edaQuery.data ? (
          <Week1AcademicEdaPanel
            payload={edaQuery.data}
            clustering={clusteringQuery.data}
            clusteringLoading={clusteringQuery.isLoading || clusteringQuery.isPending}
          />
        ) : null
      ) : null}

      {week.analysis_available.includes('preview') && previewQuery.data && !isWeek1 ? (
        <section className="stack">
          <DataTable columns={previewQuery.data.columns} rows={previewQuery.data.rows} />
        </section>
      ) : null}

      {week.analysis_available.includes('ml_overview') ? (
        mlData ? (
          <section className="stack">
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '-8px' }}>
              <button
                type="button"
                className="outline-btn"
                style={{ fontSize: '0.8rem', padding: '6px 12px', background: '#dfe8ec', color: 'var(--ink)' }}
                onClick={handleRunMl}
                disabled={mlRunning}
              >
                {mlRunning ? 'Recalculando...' : 'Recalcular'}
              </button>
            </div>
            {mlRunning && mlProgress ? (
              <div className="panel">
                <div className="ml-progress-bar">
                  <div
                    className="ml-progress-fill"
                    style={{ width: `${Math.round((mlProgress.current / mlProgress.total) * 100)}%` }}
                  />
                </div>
                <p className="muted" style={{ margin: '8px 0 0', fontSize: '0.85rem' }}>
                  {Math.round((mlProgress.current / mlProgress.total) * 100)}% ({mlProgress.current}/{mlProgress.total}) — {mlProgress.message}
                </p>
              </div>
            ) : null}
            {mlData.priority_classification ? (
              <MlClassificationPanel classification={mlData.priority_classification} split={mlData.split} />
            ) : null}
            <details className="panel">
              <summary><h3 style={{ display: 'inline' }}>Analisis de regresion (comparacion)</h3></summary>
              <MlOverviewPanel payload={mlData} />
              <div className="panel">
                <h3>Resumen supervisado</h3>
                <MetricCards metrics={mlData.supervised_overview.target_stats} />
              </div>
              <WarningChips warnings={mlData.supervised_overview.warnings} />
              <AnovaTable rows={mlData.anova.rows} />
              <BoxplotPanel series={mlData.anova.boxplot_data} />
              <MultipleRegressionPanel payload={mlData.multiple_regression} />
            </details>
          </section>
        ) : (
          <section className="panel">
            <h3>Análisis ML</h3>
            {mlRunning && mlProgress ? (
              <>
                <div className="ml-progress-bar">
                  <div
                    className="ml-progress-fill"
                    style={{ width: `${Math.round((mlProgress.current / mlProgress.total) * 100)}%` }}
                  />
                </div>
                <p className="muted" style={{ margin: '8px 0 0', fontSize: '0.85rem' }}>
                  {Math.round((mlProgress.current / mlProgress.total) * 100)}% ({mlProgress.current}/{mlProgress.total}) — {mlProgress.message}
                </p>
              </>
            ) : mlRunning ? (
              <p className="muted">Iniciando análisis...</p>
            ) : (
              <>
                {mlError ? <p className="error" style={{ marginBottom: '8px' }}>{mlError}</p> : null}
                <button type="button" onClick={handleRunMl} disabled={mlRunning}>
                  Ejecutar análisis ML
                </button>
                <p className="muted" style={{ margin: '8px 0 0', fontSize: '0.85rem' }}>
                  Entrena 5 modelos × 3 estrategias sobre el dataset temporal.
                </p>
              </>
            )}
          </section>
        )
      ) : null}

      {week.analysis_available.includes('optimization_model') ? (
        optimizationQuery.isLoading ? (
          <section className="panel">
            <h3>Modelo de optimización</h3>
            <p className="muted">Construyendo la formulación y los parámetros de semana 3...</p>
          </section>
        ) : optimizationQuery.isError ? (
          <section className="panel">
            <h3>Modelo de optimización</h3>
            <p className="error">No se pudo cargar el modelo de optimización de la semana 3.</p>
          </section>
        ) : optimizationQuery.data ? (
          <OptimizationModelPanel payload={optimizationQuery.data} />
        ) : null
      ) : null}

      {week.status === 'scaffolded' ? (
        <section className="panel">
          <h3>Base guiada</h3>
          <p className="muted">
            Esta semana ya tiene estructura, reporte persistente y artefactos esperados, pero la logica matematica u
            optimizacion aun no esta implementada.
          </p>
        </section>
      ) : null}

      {weekId === 'week-1' ? (
        <>
          <section id="report">
            <ReportPanel
              report={reportQuery.data}
              loading={reportQuery.isLoading}
              refreshing={refreshReportMutation.isPending}
              onRefresh={() => refreshReportMutation.mutateAsync().then(() => undefined)}
            />
          </section>
          <NotesPanel
            initialContent={notesQuery.data?.content ?? ''}
            updatedAt={notesQuery.data?.updated_at}
            onSave={(content) => saveNotesMutation.mutateAsync(content).then(() => undefined)}
            loading={saveNotesMutation.isPending}
          />
        </>
      ) : (
        <div className="grid-2">
          <NotesPanel
            initialContent={notesQuery.data?.content ?? ''}
            updatedAt={notesQuery.data?.updated_at}
            onSave={(content) => saveNotesMutation.mutateAsync(content).then(() => undefined)}
            loading={saveNotesMutation.isPending}
          />
          <ReportPanel
            report={reportQuery.data}
            loading={reportQuery.isLoading}
            refreshing={refreshReportMutation.isPending}
            onRefresh={() => refreshReportMutation.mutateAsync().then(() => undefined)}
          />
        </div>
      )}

      {(weekQuery.isError || previewQuery.isError || edaQuery.isError || clusteringQuery.isError || mlCachedQuery.isError || optimizationQuery.isError || notesQuery.isError || reportQuery.isError) ? (
        <p className="error">Hubo un error cargando la semana. Revisa backend, manifest y seeds.</p>
      ) : null}
    </main>
  );
}
