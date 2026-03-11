import { useEffect, useRef, useState } from 'react';
import { Link, Navigate, useParams } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';

import {
  fetchWeekConfig,
  fetchWeekClustering,
  fetchWeekEDA,
  fetchWeekMlOverview,
  fetchWeekNotes,
  fetchWeekPreview,
  fetchWeekReport,
  refreshWeekReport,
  saveWeekNotes,
} from '../api/client';
import ArtifactList from '../components/ArtifactList';
import AnovaTable from '../components/AnovaTable';
import BoxplotPanel from '../components/BoxplotPanel';
import DataTable from '../components/DataTable';
import MetricCards from '../components/MetricCards';
import MlOverviewPanel from '../components/MlOverviewPanel';
import MultipleRegressionPanel from '../components/MultipleRegressionPanel';
import NotesPanel from '../components/NotesPanel';
import ReportPanel from '../components/ReportPanel';
import Week1AcademicEdaPanel from '../components/Week1AcademicEdaPanel';
import WarningChips from '../components/WarningChips';
import { downloadPageAsPdf } from '../utils/pdfExport';

export default function WeekPage() {
  const { weekId } = useParams();
  const isWeek1 = weekId === 'week-1';
  const pageRef = useRef<HTMLElement>(null);
  const [pdfStatus, setPdfStatus] = useState<string | null>(null);

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

  const mlQuery = useQuery({
    queryKey: ['week-ml', weekId],
    queryFn: () => fetchWeekMlOverview(weekId as string),
    enabled: !!weekId && !!weekQuery.data?.analysis_available.includes('ml_overview'),
  });

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

      <ArtifactList artifacts={week.artifacts} />

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

      {week.analysis_available.includes('ml_overview') && mlQuery.data ? (
        <section className="stack">
          <MlOverviewPanel payload={mlQuery.data} />
          <div className="panel">
            <h3>Resumen supervisado</h3>
            <MetricCards metrics={mlQuery.data.supervised_overview.target_stats} />
          </div>
          <WarningChips warnings={mlQuery.data.supervised_overview.warnings} />
          <AnovaTable rows={mlQuery.data.anova.rows} />
          <BoxplotPanel series={mlQuery.data.anova.boxplot_data} />
          <MultipleRegressionPanel payload={mlQuery.data.multiple_regression} />
        </section>
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

      {(weekQuery.isError || previewQuery.isError || edaQuery.isError || clusteringQuery.isError || mlQuery.isError || notesQuery.isError || reportQuery.isError) ? (
        <p className="error">Hubo un error cargando la semana. Revisa backend, manifest y seeds.</p>
      ) : null}
    </main>
  );
}
