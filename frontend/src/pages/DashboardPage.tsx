import { useMemo, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useMutation, useQuery } from '@tanstack/react-query';

import {
  fetchAnova,
  fetchEDA,
  fetchMultipleRegression,
  fetchNotes,
  fetchPreview,
  fetchSupervisedOverview,
  fetchVariability,
  saveNotes,
} from '../api/client';
import type { VariabilityRow, WarningItem } from '../api/types';
import AnovaTable from '../components/AnovaTable';
import BoxplotPanel from '../components/BoxplotPanel';
import CategoricalDistributionChart from '../components/CategoricalDistributionChart';
import DataTable from '../components/DataTable';
import MetricCards from '../components/MetricCards';
import MultipleRegressionPanel from '../components/MultipleRegressionPanel';
import NotesPanel from '../components/NotesPanel';
import NumericHistogram from '../components/NumericHistogram';
import PivotTablePanel from '../components/PivotTablePanel';
import VariabilityRanking from '../components/VariabilityRanking';
import WarningChips from '../components/WarningChips';
import { useDatasetId } from '../hooks/useDatasetId';

type TabKey = 'preview' | 'eda' | 'variability' | 'supervised' | 'pivot' | 'notes';

type VariabilityMetric = 'entropy' | 'gini_impurity' | 'coefficient_variation' | 'custom_index';

function dedupeWarnings(warnings: WarningItem[]): WarningItem[] {
  const seen = new Set<string>();
  return warnings.filter((warning) => {
    const key = [warning.code, warning.severity, warning.message, warning.suggestion ?? ''].join('|');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function summarizeVariabilityWarnings(rows: VariabilityRow[], metric: VariabilityMetric): WarningItem[] {
  const allWarnings = rows.flatMap((row) => row.warnings);
  if (metric !== 'custom_index') {
    return dedupeWarnings(allWarnings.filter((warning) => warning.code !== 'custom_non_informative'));
  }
  const customNonInformative = allWarnings.filter((warning) => warning.code === 'custom_non_informative');
  const nonCustom = allWarnings.filter((warning) => warning.code !== 'custom_non_informative');

  const deduped = dedupeWarnings(nonCustom);
  if (customNonInformative.length === 0) return deduped;

  const base = customNonInformative[0];
  deduped.push({
    code: base.code,
    severity: base.severity,
    column: null,
    message:
      customNonInformative.length > 1
        ? `${base.message} Detectado en ${customNonInformative.length} columnas categóricas.`
        : base.message,
    suggestion: base.suggestion,
  });
  return deduped;
}

export default function DashboardPage() {
  const { datasetId, setDatasetId } = useDatasetId();
  const [tab, setTab] = useState<TabKey>('preview');
  const [pivotSource, setPivotSource] = useState<'in' | 'out'>('in');
  const [metric, setMetric] = useState<VariabilityMetric>('entropy');
  const [customMode, setCustomMode] = useState<'freq_only' | 'ordinal_map'>('freq_only');
  const [ordinalStrategy, setOrdinalStrategy] = useState<'frequency' | 'alphabetical'>('frequency');

  if (!datasetId) {
    return <Navigate to="/" replace />;
  }

  const previewQuery = useQuery({
    queryKey: ['preview', datasetId],
    queryFn: () => fetchPreview(datasetId, 20),
  });

  const edaQuery = useQuery({
    queryKey: ['eda', datasetId],
    queryFn: () => fetchEDA(datasetId),
  });

  const variabilityQuery = useQuery({
    queryKey: ['variability', datasetId, customMode, ordinalStrategy],
    queryFn: () => fetchVariability(datasetId, customMode, ordinalStrategy),
  });

  const supervisedQuery = useQuery({
    queryKey: ['supervised-overview', datasetId],
    queryFn: () => fetchSupervisedOverview(datasetId),
  });

  const anovaQuery = useQuery({
    queryKey: ['anova', datasetId],
    queryFn: () => fetchAnova(datasetId),
  });

  const multipleRegressionQuery = useQuery({
    queryKey: ['multiple-regression', datasetId],
    queryFn: () => fetchMultipleRegression(datasetId),
  });

  const notesQuery = useQuery({
    queryKey: ['notes', datasetId],
    queryFn: () => fetchNotes(datasetId),
  });

  const saveNotesMutation = useMutation({
    mutationFn: (content: string) => saveNotes(datasetId, content),
    onSuccess: () => {
      notesQuery.refetch();
    },
  });

  const categoricalCharts = useMemo(() => {
    if (!edaQuery.data) return [];
    return Object.entries(edaQuery.data.top_values).filter(([, values]) => values.length > 0).slice(0, 6);
  }, [edaQuery.data]);

  const histogramCharts = useMemo(() => {
    if (!edaQuery.data) return [];
    return Object.entries(edaQuery.data.numeric_histograms).filter(([, bins]) => bins.length > 0).slice(0, 4);
  }, [edaQuery.data]);

  const variabilityRows: VariabilityRow[] = variabilityQuery.data?.rows ?? [];
  const variabilityWarnings = useMemo(
    () => summarizeVariabilityWarnings(variabilityRows, metric),
    [variabilityRows, metric],
  );
  const hasTarget = previewQuery.data?.columns.includes('DaysInDeposit') ?? false;

  return (
    <main className="page">
      <header className="hero compact">
        <div>
          <h1>Dashboard Dataset</h1>
          <p>
            dataset_id: {datasetId} | target DaysInDeposit: <strong>{hasTarget ? 'disponible' : 'no disponible'}</strong>
          </p>
        </div>
        <div className="hero-actions">
          <Link to="/">Subir nuevos CSV</Link>
          <button onClick={() => setDatasetId(null)}>Limpiar dataset</button>
        </div>
      </header>

      <nav className="tabs">
        {[
          ['preview', 'Preview'],
          ['eda', 'EDA'],
          ['variability', 'Variability'],
          ['supervised', 'Supervised'],
          ['pivot', 'Pivot'],
          ['notes', 'Notas'],
        ].map(([key, label]) => (
          <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key as TabKey)}>
            {label}
          </button>
        ))}
      </nav>

      {tab === 'preview' && previewQuery.data ? (
        <section>
          <DataTable columns={previewQuery.data.columns} rows={previewQuery.data.rows} />
        </section>
      ) : null}

      {tab === 'eda' && edaQuery.data ? (
        <section className="stack">
          <MetricCards metrics={edaQuery.data.global_metrics} />
          <WarningChips warnings={edaQuery.data.warnings} />

          <div className="grid-2">
            {categoricalCharts.map(([column, values]) => (
              <CategoricalDistributionChart key={column} title={`Top valores - ${column}`} data={values} />
            ))}
          </div>

          <div className="grid-2">
            {histogramCharts.map(([column, bins]) => (
              <NumericHistogram key={column} title={`Histograma - ${column}`} bins={bins} />
            ))}
          </div>
        </section>
      ) : null}

      {tab === 'variability' ? (
        <section className="stack">
          <div className="panel controls">
            <div className="control-group">
              <label>Métrica ranking</label>
              <select value={metric} onChange={(e) => setMetric(e.target.value as VariabilityMetric)}>
                <option value="entropy">Entropy</option>
                <option value="gini_impurity">Gini</option>
                <option value="coefficient_variation">CV</option>
                <option value="custom_index">Custom Index</option>
              </select>
            </div>
            <div className="control-group">
              <label>Custom mode</label>
              <select value={customMode} onChange={(e) => setCustomMode(e.target.value as 'freq_only' | 'ordinal_map')}>
                <option value="freq_only">freq_only</option>
                <option value="ordinal_map">ordinal_map</option>
              </select>
            </div>
            <div className="control-group">
              <label>Ordinal strategy</label>
              <select value={ordinalStrategy} onChange={(e) => setOrdinalStrategy(e.target.value as 'frequency' | 'alphabetical')}>
                <option value="frequency">frequency</option>
                <option value="alphabetical">alphabetical</option>
              </select>
            </div>
          </div>

          <VariabilityRanking rows={variabilityRows} metric={metric} />
          <WarningChips warnings={variabilityWarnings} />
        </section>
      ) : null}

      {tab === 'supervised' ? (
        <section className="stack">
          {supervisedQuery.isLoading ? (
            <div className="panel">
              <h3>Supervised</h3>
              <p className="muted">Cargando análisis supervisado...</p>
            </div>
          ) : supervisedQuery.isError ? (
            <div className="panel">
              <h3>Supervised no disponible</h3>
              <p className="error">No se pudo cargar el análisis supervisado para este dataset.</p>
              <p className="muted">Reintenta o vuelve a subir ambos archivos IN/OUT.</p>
            </div>
          ) : supervisedQuery.data?.target_present ? (
            <>
              <MetricCards metrics={supervisedQuery.data.target_stats as Record<string, number>} />
              <WarningChips warnings={[...(supervisedQuery.data.warnings ?? []), ...(anovaQuery.data?.warnings ?? [])]} />
              <AnovaTable rows={anovaQuery.data?.rows ?? []} />
              <BoxplotPanel series={anovaQuery.data?.boxplot_data ?? []} />
              {multipleRegressionQuery.isLoading ? (
                <div className="panel">
                  <h3>Regresión Múltiple (OUT -&gt; DaysInDeposit)</h3>
                  <p className="muted">Calculando modelo multivariado...</p>
                </div>
              ) : null}
              {multipleRegressionQuery.data ? <MultipleRegressionPanel payload={multipleRegressionQuery.data} /> : null}
            </>
          ) : (
            <div className="panel">
              <h3>Supervised no disponible</h3>
              <p className="muted">Este dataset no contiene DaysInDeposit.</p>
              <p className="muted">
                Para habilitarlo, vuelve a Upload y sube <code>Grupo1_in.csv</code> + <code>Grupo1_out.csv</code> en la misma carga.
              </p>
              <p className="muted">
                Dataset actual columnas: {previewQuery.data?.columns.join(', ') ?? 'cargando...'}
              </p>
              <WarningChips warnings={supervisedQuery.data?.warnings ?? []} />
            </div>
          )}
        </section>
      ) : null}

      {tab === 'pivot' ? (
        <section className="stack">
          <h3>Pivot Lab</h3>
          <p className="muted">Flujo simple: elige fuente, fila, columna, valor y agregación. Se recalcula automáticamente.</p>
          <div className="panel">
            <div className="segmented">
              <button className={pivotSource === 'in' ? 'active' : ''} onClick={() => setPivotSource('in')}>
                Fuente IN
              </button>
              <button className={pivotSource === 'out' ? 'active' : ''} onClick={() => setPivotSource('out')}>
                Fuente OUT
              </button>
            </div>
          </div>
          <PivotTablePanel datasetId={datasetId} source={pivotSource} />
        </section>
      ) : null}

      {tab === 'notes' ? (
        <NotesPanel
          initialContent={notesQuery.data?.content ?? ''}
          updatedAt={notesQuery.data?.updated_at}
          onSave={(content) => saveNotesMutation.mutateAsync(content).then(() => undefined)}
          loading={saveNotesMutation.isPending}
        />
      ) : null}

      {(previewQuery.isLoading || edaQuery.isLoading || variabilityQuery.isLoading) && <p className="muted">Cargando...</p>}
      {(
        previewQuery.isError ||
        edaQuery.isError ||
        variabilityQuery.isError ||
        supervisedQuery.isError ||
        anovaQuery.isError ||
        multipleRegressionQuery.isError
      ) && (
        <p className="error">Hubo un error consultando la API. Revisa backend y CORS.</p>
      )}
    </main>
  );
}
