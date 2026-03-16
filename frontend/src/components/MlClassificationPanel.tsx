import { useRef, useState } from 'react';
import Plot from 'react-plotly.js';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type {
  ClassificationModelResult,
  ClassificationPredictionPoint,
  MethodologyStep,
  MlSplitSummary,
  PriorityClassificationResult,
} from '../api/types';
import { exportHtmlElementAsPng, exportSvgInContainerAsPng } from '../utils/chartExport';

type Props = {
  classification: PriorityClassificationResult;
  split?: MlSplitSummary;
};

const BAND_COLORS: Record<string, string> = {
  Rapido: '#1f8f6b',
  Medio: '#d08a22',
  Largo: '#b14c36',
  // Legacy 4-band colors
  Urgente: '#b14c36',
  Corto: '#d08a22',
};

const STEP_ICONS: Record<number, string> = {
  1: '1',
  2: '2',
  3: '3',
  4: '4',
  5: '5',
  6: '6',
  7: '7',
  8: '8',
  9: '9',
  10: '10',
};

type FlowChartRow = {
  actualBand: string;
  total: number;
} & Record<string, string | number>;

type FlowTooltipEntry = {
  color?: string;
  dataKey?: string | number;
  value?: number | string;
  payload?: FlowChartRow;
  name?: string;
};

type FlowTooltipProps = {
  active?: boolean;
  label?: string;
  payload?: FlowTooltipEntry[];
};

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatPercentValue(value: number): string {
  return `${value.toFixed(1)}%`;
}

function formatCount(value: number): string {
  return value.toLocaleString('es-CL');
}

function formatMetric(value: number): string {
  return value.toFixed(4);
}

function formatMaybeMetric(value: number | null | undefined, digits = 3): string {
  if (value == null || Number.isNaN(value)) return 'n/a';
  return value.toFixed(digits);
}

async function exportSectionPng(element: HTMLElement | null, fileBaseName: string, sectionLabel: string): Promise<void> {
  if (!element) return;
  try {
    await exportHtmlElementAsPng(
      element,
      fileBaseName,
      { scale: 4, backgroundColor: '#f7f3ec' },
    );
  } catch (error) {
    console.error(`${sectionLabel} export failed`, error);
    window.alert('No se pudo exportar el PNG HD de esta sección.');
  }
}

function formatRange(minDays: number, maxDays: number): string {
  return maxDays > 999 ? `${minDays}+ dias` : `${minDays}-${maxDays} dias`;
}

function formatDeltaPoints(value: number): string {
  const points = value * 100;
  return `${points >= 0 ? '+' : ''}${points.toFixed(1)} pp`;
}

function buildHistogramWindow(predictions: ClassificationPredictionPoint[]): { start: number; end: number; size: number } | null {
  const values = predictions
    .map((point) => point.actual_days)
    .filter((value) => Number.isFinite(value));

  if (values.length === 0) return null;

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (minValue === maxValue) {
    return { start: minValue - 0.5, end: maxValue + 0.5, size: 1 };
  }

  const size = Math.max((maxValue - minValue) / 18, 0.5);
  return {
    start: minValue,
    end: maxValue + size,
    size,
  };
}

function buildFlowChartData(
  predictions: ClassificationPredictionPoint[],
  bandLabels: string[],
): FlowChartRow[] {
  return bandLabels.map((actualBand) => {
    const rowPredictions = predictions.filter((point) => point.actual_band === actualBand);
    const total = rowPredictions.length;
    const row: FlowChartRow = {
      actualBand,
      total,
    };

    for (const predictedBand of bandLabels) {
      const count = rowPredictions.filter((point) => point.predicted_band === predictedBand).length;
      row[predictedBand] = total > 0 ? (count / total) * 100 : 0;
      row[`${predictedBand}__count`] = count;
    }

    return row;
  });
}

function describeBand(label: string, minDays: number, maxDays: number): string {
  if (label === 'Rapido') return `Salida veloz en ${formatRange(minDays, maxDays)}.`;
  if (label === 'Medio') return `Comportamiento intermedio en ${formatRange(minDays, maxDays)}.`;
  if (label === 'Largo') return `Retencion prolongada de ${formatRange(minDays, maxDays)}.`;
  return `Contenedores en ${formatRange(minDays, maxDays)}.`;
}

function BandBar({ label, ratio, color }: { label: string; ratio: number; color: string }) {
  return (
    <div style={{ display: 'grid', gap: '0.35rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', fontSize: '0.82rem' }}>
        <span style={{ color: 'var(--muted)', fontWeight: 600 }}>{label}</span>
        <strong style={{ color }}>{formatPct(ratio)}</strong>
      </div>
      <div
        style={{
          height: '10px',
          borderRadius: '999px',
          overflow: 'hidden',
          background: 'rgba(29, 43, 54, 0.08)',
          border: '1px solid rgba(29, 43, 54, 0.06)',
        }}
      >
        <div
          style={{
            width: `${Math.max(0, Math.min(100, ratio * 100))}%`,
            height: '100%',
            borderRadius: '999px',
            background: `linear-gradient(90deg, ${color}cc 0%, ${color} 100%)`,
          }}
        />
      </div>
    </div>
  );
}

function BandCards({ classification, split }: Props) {
  const totalTrain = classification.bands.reduce((sum, band) => sum + band.count_train, 0);
  const totalTest = classification.bands.reduce((sum, band) => sum + band.count_test, 0);
  const totalWeeks = (split?.train_weeks.length ?? 0) + (split?.test_weeks.length ?? 0);

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.75rem',
        }}
      >
        <article className="mini-panel" style={{ background: 'linear-gradient(180deg, #fbfdfe 0%, #f4f8fa 100%)' }}>
          <strong>Ventana temporal</strong>
          <p>{totalWeeks > 0 ? `${totalWeeks} semanas` : 'n/a'}</p>
          {split ? (
            <small className="muted">
              Train {split.train_weeks.length} | Test {split.test_weeks.join(', ') || 'n/a'}
            </small>
          ) : null}
        </article>
        <article className="mini-panel" style={{ background: 'linear-gradient(180deg, #fbfdfe 0%, #f4f8fa 100%)' }}>
          <strong>Total train</strong>
          <p>{formatCount(totalTrain)}</p>
          <small className="muted">Distribucion historica usada para entrenar.</small>
        </article>
        <article className="mini-panel" style={{ background: 'linear-gradient(180deg, #fbfdfe 0%, #f4f8fa 100%)' }}>
          <strong>Total test</strong>
          <p>{formatCount(totalTest)}</p>
          <small className="muted">Holdout temporal para validar el clasificador.</small>
        </article>
        <article className="mini-panel" style={{ background: 'linear-gradient(180deg, #fbfdfe 0%, #f4f8fa 100%)' }}>
          <strong>Baseline</strong>
          <p>{formatPct(classification.baseline_accuracy)}</p>
          <small className="muted">Accuracy de la clase mas frecuente.</small>
        </article>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(265px, 1fr))',
          gap: '1rem',
        }}
      >
        {classification.bands.map((band) => {
          const color = BAND_COLORS[band.label] ?? '#95a5a6';
          const trainRatio = totalTrain > 0 ? band.count_train / totalTrain : 0;
          const testRatio = totalTest > 0 ? band.count_test / totalTest : 0;
          const delta = testRatio - trainRatio;

          return (
            <article
              key={band.label}
              style={{
                border: `1px solid ${color}33`,
                borderTop: `5px solid ${color}`,
                borderRadius: '16px',
                padding: '1rem',
                background: `linear-gradient(180deg, ${color}12 0%, rgba(255,255,255,0.96) 42%, #ffffff 100%)`,
                boxShadow: '0 10px 24px rgba(23, 48, 67, 0.08)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.55rem', marginBottom: '0.45rem' }}>
                    <span
                      style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '999px',
                        background: color,
                        boxShadow: `0 0 0 5px ${color}1f`,
                      }}
                    />
                    <strong style={{ color, fontSize: '1rem' }}>{band.label}</strong>
                  </div>
                  <div style={{ fontSize: '1.08rem', fontWeight: 800, color: 'var(--ink)' }}>
                    {formatRange(band.min_days, band.max_days)}
                  </div>
                </div>
                <span
                  style={{
                    padding: '0.35rem 0.6rem',
                    borderRadius: '999px',
                    background: `${color}18`,
                    color,
                    fontSize: '0.74rem',
                    fontWeight: 800,
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {delta >= 0 ? 'sube en test' : 'baja en test'}
                </span>
              </div>

              <p style={{ margin: '0.55rem 0 0', fontSize: '0.88rem', lineHeight: 1.55, color: 'var(--muted)' }}>
                {describeBand(band.label, band.min_days, band.max_days)}
              </p>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: '0.75rem',
                  marginTop: '1rem',
                }}
              >
                <div
                  style={{
                    padding: '0.8rem',
                    borderRadius: '12px',
                    border: '1px solid rgba(29, 43, 54, 0.08)',
                    background: 'rgba(255,255,255,0.88)',
                  }}
                >
                  <small className="muted" style={{ display: 'block', marginBottom: '0.3rem' }}>Train</small>
                  <strong style={{ display: 'block', fontSize: '1.2rem' }}>{formatCount(band.count_train)}</strong>
                  <span style={{ fontSize: '0.84rem', color: color }}>{formatPct(trainRatio)}</span>
                </div>
                <div
                  style={{
                    padding: '0.8rem',
                    borderRadius: '12px',
                    border: '1px solid rgba(29, 43, 54, 0.08)',
                    background: 'rgba(255,255,255,0.88)',
                  }}
                >
                  <small className="muted" style={{ display: 'block', marginBottom: '0.3rem' }}>Test</small>
                  <strong style={{ display: 'block', fontSize: '1.2rem' }}>{formatCount(band.count_test)}</strong>
                  <span style={{ fontSize: '0.84rem', color: color }}>{formatPct(testRatio)}</span>
                </div>
              </div>

              <div style={{ display: 'grid', gap: '0.7rem', marginTop: '1rem' }}>
                <BandBar label="Peso en train" ratio={trainRatio} color={color} />
                <BandBar label="Peso en test" ratio={testRatio} color={color} />
              </div>

              <div
                style={{
                  marginTop: '1rem',
                  padding: '0.85rem 0.9rem',
                  borderRadius: '12px',
                  border: '1px dashed rgba(29, 43, 54, 0.12)',
                  background: 'rgba(250, 252, 253, 0.92)',
                }}
              >
                <small
                  style={{
                    display: 'block',
                    fontSize: '0.73rem',
                    color: 'var(--muted)',
                    letterSpacing: '0.04em',
                    textTransform: 'uppercase',
                    marginBottom: '0.25rem',
                  }}
                >
                  Cambio test vs train
                </small>
                <strong style={{ color, fontSize: '1rem' }}>{formatDeltaPoints(delta)}</strong>
                <p style={{ margin: '0.35rem 0 0', fontSize: '0.82rem', color: 'var(--muted)' }}>
                  Diferencia en participacion de la banda entre ambos cortes.
                </p>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function FlowTooltipContent({ active, payload, label }: FlowTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;

  const row = payload[0]?.payload;
  if (!row) return null;

  const entries = payload
    .filter((entry) => typeof entry.dataKey === 'string' && !String(entry.dataKey).endsWith('__count'))
    .map((entry) => {
      const band = String(entry.dataKey);
      const pct = typeof entry.value === 'number' ? entry.value : Number(entry.value ?? 0);
      const countKey = `${band}__count`;
      const count = typeof row[countKey] === 'number' ? Number(row[countKey]) : 0;
      return {
        band,
        pct,
        count,
        color: entry.color ?? BAND_COLORS[band] ?? '#21495f',
      };
    })
    .sort((left, right) => right.pct - left.pct);

  return (
    <div
      style={{
        backgroundColor: 'var(--panel)',
        border: '1px solid var(--line)',
        borderRadius: '10px',
        padding: '0.75rem 0.9rem',
        boxShadow: '0 10px 20px rgba(23, 48, 67, 0.12)',
      }}
    >
      <strong style={{ display: 'block', marginBottom: '0.45rem' }}>{label}</strong>
      <p style={{ margin: '0 0 0.5rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
        Total holdout: {formatCount(Number(row.total ?? 0))}
      </p>
      <div style={{ display: 'grid', gap: '0.35rem' }}>
        {entries.map((entry) => (
          <div key={entry.band} style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', fontSize: '0.82rem' }}>
            <span style={{ color: entry.color, fontWeight: 700 }}>{entry.band}</span>
            <span>{formatCount(entry.count)} | {formatPercentValue(entry.pct)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function HistogramByBandPanel({
  predictions,
  bandLabels,
  testLabel,
}: {
  predictions: ClassificationPredictionPoint[];
  bandLabels: string[];
  testLabel: string;
}) {
  const [selectedBand, setSelectedBand] = useState(bandLabels[0] ?? '');
  const chartRef = useRef<HTMLDivElement | null>(null);
  const activeBand = bandLabels.includes(selectedBand) ? selectedBand : (bandLabels[0] ?? '');
  const histogramWindow = buildHistogramWindow(predictions);
  const activeColor = BAND_COLORS[activeBand] ?? '#21495f';
  const actualValues = predictions
    .filter((point) => point.actual_band === activeBand)
    .map((point) => point.actual_days);
  const predictedValues = predictions
    .filter((point) => point.predicted_band === activeBand)
    .map((point) => point.actual_days);

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `histograma-real-vs-predicho-${activeBand}-${testLabel}`);
    } catch (error) {
      console.error('classification histogram export failed', error);
      window.alert('No se pudo exportar el PNG HD de esta gráfica.');
    }
  };

  return (
    <div className="panel table-panel">
      <div className="chart-header">
        <div>
          <h3>Histograma real vs predicho por banda</h3>
          <p className="chart-note muted">
            Compara la distribucion de <code>DaysInDeposit</code> real entre filas cuya banda fue real vs predicha.
          </p>
        </div>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()}>
          Descargar PNG HD
        </button>
      </div>

      <div className="tag-list" style={{ marginBottom: '0.9rem' }}>
        {bandLabels.map((bandLabel) => (
          <button
            key={bandLabel}
            type="button"
            className={`tag ${activeBand === bandLabel ? 'active' : ''}`}
            onClick={() => setSelectedBand(bandLabel)}
            style={activeBand === bandLabel ? { background: BAND_COLORS[bandLabel] ?? 'var(--accent)', color: '#fff' } : undefined}
          >
            {bandLabel}
          </button>
        ))}
      </div>

      <div className="mini-grid" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', marginBottom: '0.9rem' }}>
        <article className="mini-panel">
          <strong>Filas banda real</strong>
          <p>{formatCount(actualValues.length)}</p>
        </article>
        <article className="mini-panel">
          <strong>Filas banda predicha</strong>
          <p>{formatCount(predictedValues.length)}</p>
        </article>
      </div>

      {histogramWindow && (actualValues.length > 0 || predictedValues.length > 0) ? (
        <div ref={chartRef} className="chart-export-area">
          <Plot
            data={[
              {
                x: actualValues,
                type: 'histogram',
                name: `Actual = ${activeBand}`,
                opacity: 0.72,
                marker: { color: activeColor },
                xbins: histogramWindow,
                hovertemplate: 'DaysInDeposit: %{x}<br>Conteo: %{y}<extra>Actual</extra>',
              },
              {
                x: predictedValues,
                type: 'histogram',
                name: `Predicho = ${activeBand}`,
                opacity: 0.58,
                marker: { color: '#21495f' },
                xbins: histogramWindow,
                hovertemplate: 'DaysInDeposit: %{x}<br>Conteo: %{y}<extra>Predicho</extra>',
              },
            ]}
            layout={{
              barmode: 'overlay',
              autosize: true,
              height: 340,
              margin: { t: 18, r: 20, b: 56, l: 56 },
              paper_bgcolor: '#f8f6f3',
              plot_bgcolor: '#ffffff',
              legend: { orientation: 'h', y: 1.12 },
              xaxis: { title: 'DaysInDeposit real' },
              yaxis: { title: 'Conteo' },
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      ) : (
        <p className="muted">No hay suficientes filas en esta banda para construir el histograma comparativo.</p>
      )}
    </div>
  );
}

function ActualToPredictedFlowPanel({
  predictions,
  bandLabels,
  testLabel,
}: {
  predictions: ClassificationPredictionPoint[];
  bandLabels: string[];
  testLabel: string;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const data = buildFlowChartData(predictions, bandLabels);

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `flujo-real-predicho-${testLabel}`);
    } catch (error) {
      console.error('classification flow export failed', error);
      window.alert('No se pudo exportar el PNG HD de esta gráfica.');
    }
  };

  return (
    <div className="panel table-panel">
      <div className="chart-header">
        <div>
          <h3>Flujo banda real → banda predicha</h3>
          <p className="chart-note muted">
            Cada barra suma 100% y muestra como se reparte la prediccion dentro de cada banda real.
          </p>
        </div>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()}>
          Descargar PNG HD
        </button>
      </div>
      <div ref={chartRef} className="chart-export-area">
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={data} margin={{ top: 12, right: 12, left: 8, bottom: 18 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--line)" opacity={0.55} />
            <XAxis dataKey="actualBand" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(value: number) => formatPercentValue(value)}
              tick={{ fill: 'var(--muted)' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<FlowTooltipContent />} />
            <Legend />
            {bandLabels.map((bandLabel) => (
              <Bar
                key={bandLabel}
                dataKey={bandLabel}
                name={bandLabel}
                stackId="flow"
                fill={BAND_COLORS[bandLabel] ?? '#21495f'}
                radius={bandLabel === bandLabels[bandLabels.length - 1] ? [4, 4, 0, 0] : [0, 0, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PriorityScoreDistributionPanel({
  predictions,
  bandLabels,
  testLabel,
}: {
  predictions: ClassificationPredictionPoint[];
  bandLabels: string[];
  testLabel: string;
}) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const traces = bandLabels
    .map((bandLabel) => ({
      bandLabel,
      values: predictions
        .filter((point) => point.actual_band === bandLabel && point.priority_score != null)
        .map((point) => Number(point.priority_score)),
    }))
    .filter((entry) => entry.values.length > 0);

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `priority-score-por-banda-${testLabel}`);
    } catch (error) {
      console.error('priority score export failed', error);
      window.alert('No se pudo exportar el PNG HD de esta gráfica.');
    }
  };

  return (
    <div className="panel table-panel">
      <div className="chart-header">
        <div>
          <h3>Distribución de priority score por banda real</h3>
          <p className="chart-note muted">
            Boxplot del score continuo derivado de <code>predict_proba</code>, agrupado por banda real.
          </p>
        </div>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()} disabled={traces.length === 0}>
          Descargar PNG HD
        </button>
      </div>

      {traces.length > 0 ? (
        <div ref={chartRef} className="chart-export-area">
          <Plot
            data={traces.map((trace) => ({
              y: trace.values,
              type: 'box',
              name: trace.bandLabel,
              boxpoints: 'outliers',
              marker: { color: BAND_COLORS[trace.bandLabel] ?? '#21495f' },
              line: { color: BAND_COLORS[trace.bandLabel] ?? '#21495f' },
            }))}
            layout={{
              autosize: true,
              height: 340,
              margin: { t: 18, r: 20, b: 52, l: 50 },
              paper_bgcolor: '#f8f6f3',
              plot_bgcolor: '#ffffff',
              yaxis: { title: 'Priority score' },
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      ) : (
        <p className="muted">El mejor modelo actual no expone <code>priority_score</code>; esta visualizacion se oculta.</p>
      )}
    </div>
  );
}

function AnalyticalVisualizationsSection({ classification, split }: Props) {
  const predictions = classification.best_model_predictions ?? [];
  const bandLabels = classification.bands.map((band) => band.label);
  const testLabel = split?.test_weeks.join('-') || 'holdout';
  const pointsWithScore = predictions.filter((point) => point.priority_score != null);

  if (predictions.length === 0) {
    return (
      <section className="panel">
        <h3>Visualizaciones analíticas</h3>
        <p className="muted">No hay predicciones detalladas del mejor modelo para construir estas visualizaciones.</p>
      </section>
    );
  }

  return (
    <section className="stack">
      <div className="panel" style={{ borderLeft: '4px solid #21495f' }}>
        <div className="section-header">
          <div>
            <h3>Visualizaciones analíticas</h3>
            <p className="muted" style={{ margin: '0.45rem 0 0' }}>
              Lectura visual del mejor clasificador sobre el holdout temporal: distribución real, flujo de bandas y separación del score continuo.
            </p>
          </div>
        </div>
        <div className="mini-grid" style={{ marginTop: '1rem' }}>
          <article className="mini-panel">
            <strong>Mejor modelo</strong>
            <p>{classification.best_model || 'n/a'}</p>
          </article>
          <article className="mini-panel">
            <strong>Filas detalladas</strong>
            <p>{formatCount(predictions.length)}</p>
          </article>
          <article className="mini-panel">
            <strong>Con score</strong>
            <p>{formatCount(pointsWithScore.length)}</p>
          </article>
          <article className="mini-panel">
            <strong>Corr. score vs días</strong>
            <p>{formatMaybeMetric(classification.priority_score_corr, 3)}</p>
          </article>
        </div>
      </div>

      <div className="grid-2">
        <HistogramByBandPanel predictions={predictions} bandLabels={bandLabels} testLabel={testLabel} />
        <ActualToPredictedFlowPanel predictions={predictions} bandLabels={bandLabels} testLabel={testLabel} />
      </div>

      <PriorityScoreDistributionPanel predictions={predictions} bandLabels={bandLabels} testLabel={testLabel} />
    </section>
  );
}

function ClassifierTable({ models, bestModel }: { models: ClassificationModelResult[]; bestModel: string }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Accuracy</th>
            <th>Adj. Accuracy</th>
            <th>MAE (bandas)</th>
            <th>F1 Weighted</th>
            <th>Estado</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const isWinner = model.model_name === bestModel && model.available;
            return (
              <tr
                key={model.model_name}
                style={isWinner ? { backgroundColor: 'rgba(15, 155, 114, 0.12)' } : undefined}
              >
                <td>
                  <div>{model.model_name}</div>
                  {isWinner ? (
                    <small style={{ color: '#0f9b72', fontWeight: 700 }}>Mejor modelo</small>
                  ) : null}
                </td>
                <td>{model.available ? formatPct(model.accuracy) : '-'}</td>
                <td>{model.available ? formatPct(model.adjacent_accuracy) : '-'}</td>
                <td>{model.available ? model.band_mae.toFixed(3) : '-'}</td>
                <td>{model.available ? formatMetric(model.f1_weighted) : '-'}</td>
                <td>
                  {model.available ? 'Disponible' : 'No disponible'}
                  {model.notes.length > 0 ? (
                    <small className="muted" style={{ display: 'block' }}>{model.notes[0]}</small>
                  ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ConfusionMatrix({ model, bandLabels }: { model: ClassificationModelResult; bandLabels: string[] }) {
  const matrix = model.confusion_matrix;
  if (!matrix || matrix.length === 0) return null;

  const maxVal = Math.max(1, ...matrix.flat());

  return (
    <div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Real \ Predicho</th>
              {bandLabels.map((lbl) => (
                <th key={lbl} style={{ color: BAND_COLORS[lbl] ?? undefined }}>{lbl}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={bandLabels[i]}>
                <td style={{ fontWeight: 600, color: BAND_COLORS[bandLabels[i]] ?? undefined }}>
                  {bandLabels[i]}
                </td>
                {row.map((cell, j) => {
                  const isDiagonal = i === j;
                  const alpha = cell / maxVal;
                  return (
                    <td
                      key={j}
                      className="pivot-value-cell"
                      style={{
                        '--heat-alpha': alpha,
                        textAlign: 'center',
                        fontWeight: isDiagonal ? 700 : 400,
                        backgroundColor: isDiagonal
                          ? `rgba(15, 155, 114, ${0.15 + alpha * 0.45})`
                          : `rgba(207, 92, 54, ${alpha * 0.3})`,
                        color: alpha > 0.7 ? '#fff' : 'var(--text)',
                      } as React.CSSProperties}
                    >
                      {cell}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {model.per_class.length > 0 ? (
        <div className="table-wrap" style={{ marginTop: '1rem' }}>
          <table>
            <thead>
              <tr>
                <th>Banda</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Support</th>
              </tr>
            </thead>
            <tbody>
              {model.per_class.map((cls) => (
                <tr key={cls.band}>
                  <td style={{ color: BAND_COLORS[cls.band] ?? undefined, fontWeight: 600 }}>{cls.band}</td>
                  <td>{formatMetric(cls.precision)}</td>
                  <td>{formatMetric(cls.recall)}</td>
                  <td>{formatMetric(cls.f1)}</td>
                  <td>{cls.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function MethodologyStepCard({ step }: { step: MethodologyStep }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: '10px',
        padding: '1.25rem 1.5rem',
        marginBottom: '1rem',
        backgroundColor: 'var(--bg)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
        <div
          style={{
            minWidth: '36px',
            height: '36px',
            borderRadius: '50%',
            backgroundColor: '#21495f',
            color: '#fff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 700,
            fontSize: '0.85rem',
            flexShrink: 0,
          }}
        >
          {STEP_ICONS[step.step] ?? step.step}
        </div>
        <div style={{ flex: 1 }}>
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem', color: 'var(--text)' }}>
            {step.title}
          </h4>

          <div style={{ marginBottom: '0.75rem' }}>
            <div
              style={{
                fontSize: '0.78rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: '#64748b',
                marginBottom: '0.25rem',
              }}
            >
              Por que
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text)' }}>
              {step.rationale}
            </p>
          </div>

          <div style={{ marginBottom: step.evidence ? '0.75rem' : 0 }}>
            <div
              style={{
                fontSize: '0.78rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: '#0f9b72',
                marginBottom: '0.25rem',
              }}
            >
              Decision
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.6, color: 'var(--text)', fontWeight: 500 }}>
              {step.decision}
            </p>
          </div>

          {step.evidence ? (
            <div>
              <div
                style={{
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  color: '#cf5c36',
                  marginBottom: '0.25rem',
                }}
              >
                Evidencia
              </div>
              <p
                style={{
                  margin: 0,
                  fontSize: '0.85rem',
                  lineHeight: 1.6,
                  color: 'var(--muted)',
                  backgroundColor: 'rgba(0,0,0,0.03)',
                  padding: '0.5rem 0.75rem',
                  borderRadius: '6px',
                  borderLeft: '3px solid #cf5c36',
                }}
              >
                {step.evidence}
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

type FeatureGroupDef = {
  title: string;
  color: string;
  bgColor: string;
  description: string;
  example: string;
  features: string[];
};

function classifyFeatures(featureNames: string[]): FeatureGroupDef[] {
  const targetEncSimple = featureNames.filter(
    (f) => f.endsWith('_target_enc') && !f.includes('x') && !f.includes('lag'),
  );
  const targetEncInteraction = featureNames.filter(
    (f) => f.endsWith('_target_enc') && f.includes('x') && !f.includes('lag'),
  );
  const freqEnc = featureNames.filter((f) => f.endsWith('_freq'));
  const countEnc = featureNames.filter((f) => f.endsWith('_count'));
  const lagMedian = featureNames.filter((f) => f.endsWith('_lag_median'));
  const lagStd = featureNames.filter((f) => f.endsWith('_lag_std'));
  const lagTrend = featureNames.filter((f) => f.endsWith('_lag_trend'));
  const lagVolume = featureNames.filter((f) => f.endsWith('_lag_volume'));
  const global = featureNames.filter((f) => f === 'prev_week_volume' || f === 'week_number');

  const groups: FeatureGroupDef[] = [];

  if (targetEncSimple.length > 0)
    groups.push({
      title: 'Target encoding simple',
      color: '#21495f',
      bgColor: 'rgba(33, 73, 95, 0.08)',
      description:
        'Mediana de DaysInDeposit por cada categoria individual, calculada sobre el conjunto de entrenamiento. Convierte una categoria (ej: Owner=7) en un numero que representa "cuanto se quedan tipicamente los contenedores de ese grupo".',
      example:
        'Owner_target_enc = 3.0 significa que la mediana de todos los contenedores de Owner 7 en train fue 3 dias. Size_target_enc = 6.0 para Size=2 vs 26.5 para Size=1.',
      features: targetEncSimple,
    });

  if (targetEncInteraction.length > 0)
    groups.push({
      title: 'Target encoding de interacciones',
      color: '#21495f',
      bgColor: 'rgba(33, 73, 95, 0.08)',
      description:
        'Mediana de DaysInDeposit por cada PAR de categorias. Captura que la combinacion importa: un contenedor DRY de Owner 7 se comporta distinto a un RF de Owner 1, aunque individualmente Owner y Type no lo expliquen.',
      example:
        'OwnerxQuality_target_enc: Owner=7 + CLASE B-C tiene mediana 2.5 dias, pero Owner=1 + INSPECTION tiene mediana 55 dias. Esta interaccion tiene correlacion 0.496 con el target, mas alta que cualquier feature individual.',
      features: targetEncInteraction,
    });

  if (freqEnc.length > 0)
    groups.push({
      title: 'Frequency encoding',
      color: '#6366f1',
      bgColor: 'rgba(99, 102, 241, 0.08)',
      description:
        'Proporcion de cada categoria en train (valor entre 0.0 y 1.0). Los owners mas frecuentes tienden a retirar contenedores mas rapido porque tienen operaciones regulares y booking predecible.',
      example:
        'Owner_freq = 0.49 para Owner 7 (el mas comun, mueve rapido) vs Owner_freq = 0.01 para Owner 4 (poco frecuente, tiempos largos). Correlacion con target: -0.400.',
      features: freqEnc,
    });

  if (countEnc.length > 0)
    groups.push({
      title: 'Count encoding',
      color: '#6366f1',
      bgColor: 'rgba(99, 102, 241, 0.08)',
      description:
        'Cantidad absoluta de registros de esa categoria en train. Similar al frequency encoding pero en escala absoluta, util cuando el volumen total importa.',
      example:
        'Owner_count = 3600 para Owner 7 vs Owner_count = 80 para Owner 4. Owners con alto volumen operan regularmente y retiran mas rapido.',
      features: countEnc,
    });

  if (lagMedian.length > 0)
    groups.push({
      title: 'Lag median (historia acumulada)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Mediana de DaysInDeposit del grupo en TODAS las semanas anteriores al registro actual. Para una fila del holdout temporal, usa solo semanas previas del entrenamiento. No trackea el mismo contenedor: usa el comportamiento agregado del grupo.',
      example:
        'Owner_lag_median para Owner=7: mediana historica de 3 dias calculada con todas las semanas previas disponibles del train. A mayor historia acumulada, la estimacion suele ser mas estable.',
      features: lagMedian,
    });

  if (lagStd.length > 0)
    groups.push({
      title: 'Lag std (volatilidad historica)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Desviacion estandar historica de DaysInDeposit del grupo. Mide que tan predecible es un grupo: un Owner con std baja es confiable, uno con std alta es impredecible.',
      example:
        'Owner_lag_std = 2.5 para Owner 7 (estable, siempre retira en ~3 dias) vs Owner_lag_std = 20.1 para Owner 4 (impredecible, puede ser 2 dias o 46 dias). Grupos con alta volatilidad son mas dificiles de clasificar.',
      features: lagStd,
    });

  if (lagTrend.length > 0)
    groups.push({
      title: 'Lag trend (tendencia reciente)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Diferencia de mediana entre las 2 semanas mas recientes. Captura si un grupo esta acelerando o frenando su retiro. Valor positivo = se estan quedando mas, negativo = se estan yendo mas rapido.',
      example:
        'Owner_lag_trend = -3.0 para Owner 7: la mediana bajo 3 dias entre las 2 semanas previas mas recientes, esta retirando mas rapido. Owner_lag_trend = +15 para Owner 4: salto fuerte, algo cambio en su operacion.',
      features: lagTrend,
    });

  if (lagVolume.length > 0)
    groups.push({
      title: 'Lag volume (actividad historica)',
      color: '#0f9b72',
      bgColor: 'rgba(15, 155, 114, 0.08)',
      description:
        'Cantidad acumulada de contenedores del grupo en semanas anteriores. Proxy de nivel de actividad y confiabilidad de las estadisticas: mas volumen = estimaciones mas estables.',
      example:
        'Owner_lag_volume = 3200 para Owner 7 (muchos datos, estimacion confiable) vs Owner_lag_volume = 15 para Owner 9 (pocos datos, estimacion ruidosa).',
      features: lagVolume,
    });

  if (global.length > 0)
    groups.push({
      title: 'Variables globales',
      color: '#cf5c36',
      bgColor: 'rgba(207, 92, 54, 0.08)',
      description:
        'Features que no dependen de una categoria especifica sino del contexto general del puerto.',
      example:
        'prev_week_volume: total de contenedores la semana anterior (proxy de congestion portuaria — si entran muchos, los tiempos suben). week_number: numero de semana, captura tendencia lineal o estacionalidad.',
      features: global,
    });

  return groups;
}

function FeatureGroupCard({ group }: { group: FeatureGroupDef }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: '8px',
        padding: '1rem',
        marginBottom: '0.75rem',
        borderLeft: `4px solid ${group.color}`,
        backgroundColor: group.bgColor,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.5rem' }}>
        <h5 style={{ margin: 0, fontSize: '0.9rem', color: group.color }}>{group.title}</h5>
        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{group.features.length} variables</span>
      </div>

      <p style={{ margin: '0 0 0.5rem', fontSize: '0.85rem', lineHeight: 1.6, color: 'var(--text)' }}>
        {group.description}
      </p>

      <div
        style={{
          fontSize: '0.82rem',
          lineHeight: 1.6,
          color: 'var(--muted)',
          backgroundColor: 'rgba(255,255,255,0.5)',
          padding: '0.5rem 0.75rem',
          borderRadius: '5px',
          marginBottom: '0.5rem',
          fontStyle: 'italic',
        }}
      >
        Ej: {group.example}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
        {group.features.map((f) => (
          <code
            key={f}
            style={{
              fontSize: '0.72rem',
              padding: '0.15rem 0.4rem',
              borderRadius: '3px',
              backgroundColor: 'rgba(255,255,255,0.6)',
              border: `1px solid ${group.color}33`,
              color: group.color,
            }}
          >
            {f}
          </code>
        ))}
      </div>
    </div>
  );
}

function MethodologySection({ classification }: Props) {
  const methodology = classification.methodology;
  if (!methodology || methodology.length === 0) return null;

  const featureNames = classification.feature_names ?? [];
  const featureGroups = classifyFeatures(featureNames);
  const methodologyExportRef = useRef<HTMLDivElement | null>(null);

  return (
    <div className="panel table-panel">
      <div className="table-header">
        <div>
          <h3>Metodologia y justificacion</h3>
          <span className="muted">
            Detalle paso a paso de cada decision tomada en el pipeline de clasificacion.
          </span>
        </div>
        <button
          type="button"
          className="chart-export-btn"
          onClick={() => void exportSectionPng(methodologyExportRef.current, 'metodologia-clasificacion', 'methodology')}
        >
          Descargar PNG HD
        </button>
      </div>
      <div ref={methodologyExportRef} className="chart-export-area" style={{ padding: '1rem 1.5rem 1.5rem' }}>

        {/* Origin explanation */}
        <div
          style={{
            marginBottom: '1.5rem',
            padding: '1.25rem',
            borderRadius: '8px',
            border: '1px solid var(--border)',
            backgroundColor: 'rgba(33, 73, 95, 0.04)',
            borderLeft: '4px solid #21495f',
          }}
        >
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '1rem' }}>
            De donde salen las {featureNames.length} variables?
          </h4>
          <p style={{ margin: '0 0 0.75rem', fontSize: '0.9rem', lineHeight: 1.7 }}>
            El dataset original solo tiene <strong>4 columnas categoricas</strong> (Owner, Size, Type, Quality),
            el target (DaysInDeposit) y la semana. Un arbol de decision no puede usar categorias directamente.
            Las {featureNames.length} variables se <strong>construyen matematicamente</strong> a partir de esas
            4 columnas — no son datos nuevos, son <strong>transformaciones</strong> que extraen la senal
            que ya existe en los datos pero en una forma que los modelos pueden procesar.
          </p>
          <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: 1.7 }}>
            Ejemplo: para un contenedor de <code>Owner=7, Size=2, Type=DRY</code> en el holdout temporal, se calcula:
            "la mediana historica de Owner 7 fue 3 dias" (<code>Owner_target_enc=3.0</code>),
            "Owner 7 representa el 49% del trafico" (<code>Owner_freq=0.49</code>),
            "en las semanas previas del train la mediana de Owner 7 fue 3 dias" (<code>Owner_lag_median=3.0</code>),
            "la tendencia bajo 3 dias" (<code>Owner_lag_trend=-3.0</code>). Todo calculado sin mirar la
            semana que se predice (sin leakage temporal).
          </p>
        </div>

        {/* Feature inventory by group */}
        {featureGroups.length > 0 ? (
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.95rem' }}>
              Inventario de features ({featureNames.length} variables en {featureGroups.length} grupos)
            </h4>
            {featureGroups.map((group) => (
              <FeatureGroupCard key={group.title} group={group} />
            ))}
          </div>
        ) : null}

        {/* Priority score summary */}
        {classification.priority_score_corr != null ? (
          <div
            style={{
              marginBottom: '1.5rem',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px solid var(--border)',
              backgroundColor: 'rgba(15, 155, 114, 0.04)',
              borderLeft: '4px solid #0f9b72',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#0f9b72' }}>
              Priority Score para el optimizador
            </h4>
            <p style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', lineHeight: 1.6 }}>
              score = P(Rapido) x 1 + P(Medio) x 2 + P(Largo) x 3
            </p>
            <div className="mini-grid">
              <article className="mini-panel">
                <strong>Correlacion vs dias reales</strong>
                <p>{classification.priority_score_corr.toFixed(3)}</p>
              </article>
              {classification.priority_score_stats?.min != null ? (
                <article className="mini-panel">
                  <strong>Rango</strong>
                  <p>[{classification.priority_score_stats.min}, {classification.priority_score_stats.max}]</p>
                </article>
              ) : null}
              {classification.priority_score_stats?.mean != null ? (
                <article className="mini-panel">
                  <strong>Media / Std</strong>
                  <p>{classification.priority_score_stats.mean} / {classification.priority_score_stats.std}</p>
                </article>
              ) : null}
            </div>
          </div>
        ) : null}

        {/* Steps */}
        {methodology.map((step) => (
          <MethodologyStepCard key={step.step} step={step} />
        ))}
      </div>
    </div>
  );
}

export default function MlClassificationPanel({ classification, split }: Props) {
  const bestModel = classification.models.find((m) => m.model_name === classification.best_model && m.available);
  const bandLabels = classification.bands.map((b) => b.label);
  const summaryExportRef = useRef<HTMLDivElement | null>(null);
  const bandsExportRef = useRef<HTMLDivElement | null>(null);
  const modelsExportRef = useRef<HTMLDivElement | null>(null);
  const confusionExportRef = useRef<HTMLDivElement | null>(null);
  const testLabel = split?.test_weeks.join('-') || 'holdout';

  return (
    <section className="stack">
      {/* Hero / Narrative */}
      <div className="panel" style={{ borderLeft: '4px solid #0f9b72' }}>
        <div className="section-header">
          <div ref={summaryExportRef} className="chart-export-area">
            <h3>Clasificacion por bandas de prioridad</h3>
            <p style={{ fontSize: '1.05rem', lineHeight: 1.6, margin: '0.5rem 0 0' }}>
              {classification.narrative}
            </p>
            <p className="muted" style={{ marginTop: '0.5rem' }}>
              Baseline (clase mas frecuente): {formatPct(classification.baseline_accuracy)}
              {classification.priority_score_corr != null ? (
                <> | Corr. priority score vs dias: {classification.priority_score_corr.toFixed(3)}</>
              ) : null}
            </p>
          </div>
          <button
            type="button"
            className="chart-export-btn"
            onClick={() => void exportSectionPng(summaryExportRef.current, `clasificacion-prioridad-${testLabel}`, 'classification summary')}
          >
            Descargar PNG HD
          </button>
        </div>
      </div>

      {/* Band distribution cards */}
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Bandas de prioridad</h3>
            <span className="muted">Distribucion por banda entre entrenamiento historico y holdout temporal.</span>
          </div>
          <button
            type="button"
            className="chart-export-btn"
            onClick={() => void exportSectionPng(bandsExportRef.current, `bandas-prioridad-${testLabel}`, 'priority bands')}
          >
            Descargar PNG HD
          </button>
        </div>
        <div ref={bandsExportRef} className="chart-export-area" style={{ padding: '1rem 1.5rem 1.5rem' }}>
          <BandCards classification={classification} split={split} />
        </div>
      </div>

      <AnalyticalVisualizationsSection classification={classification} split={split} />

      {/* Classifier comparison table */}
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Modelos clasificadores</h3>
            <span className="muted">
              Accuracy = acierto exacto de banda. Adj. Accuracy = acierto o error por 1 banda adyacente.
            </span>
          </div>
          <button
            type="button"
            className="chart-export-btn"
            onClick={() => void exportSectionPng(modelsExportRef.current, `modelos-clasificadores-${testLabel}`, 'classifier models')}
          >
            Descargar PNG HD
          </button>
        </div>
        <div ref={modelsExportRef} className="chart-export-area">
          <ClassifierTable models={classification.models} bestModel={classification.best_model} />
        </div>
      </div>

      {/* Confusion matrix of best model */}
      {bestModel ? (
        <div className="panel table-panel">
          <div className="table-header">
            <div>
              <h3>Matriz de confusion — {bestModel.model_name}</h3>
              <span className="muted">
                Filas = banda real, columnas = banda predicha. Diagonal = aciertos.
              </span>
            </div>
            <button
              type="button"
              className="chart-export-btn"
              onClick={() => void exportSectionPng(confusionExportRef.current, `matriz-confusion-${bestModel.model_name}-${testLabel}`, 'confusion matrix')}
            >
              Descargar PNG HD
            </button>
          </div>
          <div ref={confusionExportRef} className="chart-export-area" style={{ padding: '1rem 1.5rem 1.5rem' }}>
            <ConfusionMatrix model={bestModel} bandLabels={bandLabels} />
          </div>
        </div>
      ) : null}

      {/* Methodology section */}
      <MethodologySection classification={classification} split={split} />
    </section>
  );
}
