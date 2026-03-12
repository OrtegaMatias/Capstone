import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type {
  HeuristicModelResult,
  MlBenchmarkRow,
  MlEvaluationSummary,
  MlModelResult,
  StrategyComparisonEntry,
  TreeNode,
} from '../api/types';
import BoxplotPanel from './BoxplotPanel';
import WarningChips from './WarningChips';

type Props = {
  payload: MlEvaluationSummary;
};

type TabKey = 'regresiones' | 'heuristicas' | 'aprendizajes';

function formatMetric(value?: number | null): string {
  return value == null || Number.isNaN(value) ? '-' : value.toFixed(3);
}

function formatSignedMetric(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '-';
  return `${value > 0 ? '+' : ''}${value.toFixed(3)}`;
}

function tabLabel(tab: TabKey): string {
  if (tab === 'regresiones') return 'Regresiones';
  if (tab === 'heuristicas') return 'Heurísticas';
  return 'Aprendizajes';
}

function modelBadgeText(model: MlModelResult): string {
  return model.strategy_label ? `${model.model_name} · ${model.strategy_label}` : model.model_name;
}

function strategyEntryText(entry?: StrategyComparisonEntry | null): string {
  if (!entry) return 'n/a';
  return entry.strategy_label ? `${entry.model_name} · ${entry.strategy_label}` : entry.model_name;
}

function winnerLabel(winner?: string | null): string {
  if (winner === 'regression') return 'Regresión';
  if (winner === 'heuristic') return 'Heurística';
  if (winner === 'tie') return 'Empate';
  return 'n/a';
}

function isBestRegressionRow(row: MlBenchmarkRow, bestRegression?: StrategyComparisonEntry | null): boolean {
  if (!bestRegression || row.model_name !== bestRegression.model_name) return false;
  if (bestRegression.strategy_name) return row.strategy_name === bestRegression.strategy_name;
  if (bestRegression.strategy_label) return row.strategy_label === bestRegression.strategy_label;
  return true;
}

function isBestRegressionModel(model: MlModelResult, bestRegression?: StrategyComparisonEntry | null): boolean {
  if (!bestRegression || model.model_name !== bestRegression.model_name) return false;
  if (bestRegression.strategy_name) return model.strategy_name === bestRegression.strategy_name;
  if (bestRegression.strategy_label) return model.strategy_label === bestRegression.strategy_label;
  return true;
}

function sortByMetric(value?: number | null): number {
  return value == null || Number.isNaN(value) ? Number.POSITIVE_INFINITY : value;
}

function strategyLabelRank(label: string): number {
  const order = [
    'Raw target',
    'Log1p target',
    'Log1p + outlier normalization',
    'Winsor IQR',
  ];
  const index = order.indexOf(label);
  return index === -1 ? order.length : index;
}

function heuristicBadgeText(model: HeuristicModelResult): string {
  return `${model.model_name} · ${model.family_label}`;
}

function TreeNodeView({ node, depth = 0 }: { node?: TreeNode | null; depth?: number }) {
  if (!node) return null;
  const isLeaf = node.type === 'leaf';
  const bgColor = isLeaf ? '#14866d' : '#21495f';
  const maxDepthForCollapse = 3;
  const [collapsed, setCollapsed] = useState(depth >= maxDepthForCollapse && !isLeaf);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: isLeaf ? 88 : 164 }}>
      <div
        onClick={() => !isLeaf && setCollapsed((current) => !current)}
        style={{
          backgroundColor: bgColor,
          color: 'white',
          padding: '6px 10px',
          borderRadius: isLeaf ? '20px' : '10px',
          fontSize: '11px',
          textAlign: 'center',
          cursor: isLeaf ? 'default' : 'pointer',
          maxWidth: 190,
          lineHeight: 1.4,
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          opacity: collapsed ? 0.7 : 1,
        }}
      >
        {isLeaf ? (
          <>
            <div style={{ fontWeight: 700 }}>≈ {node.value.toFixed(1)} d</div>
            <div style={{ fontSize: '9px', opacity: 0.85 }}>n={node.samples}</div>
          </>
        ) : (
          <>
            <div style={{ fontWeight: 700 }}>{node.feature}</div>
            <div>≤ {node.threshold}</div>
            <div style={{ fontSize: '9px', opacity: 0.85 }}>n={node.samples}{collapsed ? ' [+]' : ''}</div>
          </>
        )}
      </div>

      {!isLeaf && !collapsed && (node.left || node.right) ? (
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 10 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <span className="muted" style={{ fontSize: '0.72rem' }}>Sí</span>
            <TreeNodeView node={node.left} depth={depth + 1} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <span className="muted" style={{ fontSize: '0.72rem' }}>No</span>
            <TreeNodeView node={node.right} depth={depth + 1} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function BenchmarkTable({
  rows,
  bestRegression,
}: {
  rows: MlBenchmarkRow[];
  bestRegression?: StrategyComparisonEntry | null;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Estrategia</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>MedAE</th>
            <th>R²</th>
            <th>Baseline</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const isWinner = isBestRegressionRow(row, bestRegression);
            return (
              <tr
                key={`${row.model_name}-${row.strategy_name}`}
                style={isWinner ? { backgroundColor: 'rgba(15, 155, 114, 0.12)' } : undefined}
              >
                <td>
                  <div>{row.model_name}</div>
                  {isWinner ? (
                    <small style={{ color: '#0f9b72', fontWeight: 700 }}>Ganador global</small>
                  ) : null}
                </td>
                <td>{row.strategy_label}</td>
                <td>{formatMetric(row.metrics.mae)}</td>
                <td>{formatMetric(row.metrics.rmse)}</td>
                <td>{formatMetric(row.metrics.medae)}</td>
                <td>{formatMetric(row.metrics.r2)}</td>
                <td>{formatMetric(row.metrics.baseline_mae)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ModelBenchmarkTable({
  models,
  bestRegression,
}: {
  models: MlModelResult[];
  bestRegression?: StrategyComparisonEntry | null;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Modelo</th>
            <th>Mejor estrategia</th>
            <th>Estado</th>
            <th>MAE</th>
            <th>RMSE</th>
            <th>MedAE</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => {
            const isWinner = isBestRegressionModel(model, bestRegression);
            const isAvailable = model.metrics.mae != null;
            return (
              <tr
                key={`${model.model_name}-${model.strategy_name ?? 'none'}`}
                style={isWinner ? { backgroundColor: 'rgba(15, 155, 114, 0.12)' } : undefined}
              >
                <td>
                  <div>{model.model_name}</div>
                  {isWinner ? (
                    <small style={{ color: '#0f9b72', fontWeight: 700 }}>Ganador global</small>
                  ) : null}
                </td>
                <td>
                  <div>{model.strategy_label ?? 'n/a'}</div>
                  {model.notes?.[0] ? <small className="muted">{model.notes[0]}</small> : null}
                </td>
                <td>{isAvailable ? 'Disponible' : 'No disponible'}</td>
                <td>{formatMetric(model.metrics.mae)}</td>
                <td>{formatMetric(model.metrics.rmse)}</td>
                <td>{formatMetric(model.metrics.medae)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function StrategyBenchmarkTables({
  rows,
  bestRegression,
}: {
  rows: MlBenchmarkRow[];
  bestRegression?: StrategyComparisonEntry | null;
}) {
  const availableRows = rows
    .filter((row) => row.available && row.strategy_name !== 'baseline')
    .sort((left, right) => {
      if (left.strategy_label !== right.strategy_label) return strategyLabelRank(left.strategy_label) - strategyLabelRank(right.strategy_label);
      return sortByMetric(left.metrics.mae) - sortByMetric(right.metrics.mae);
    });
  const unavailableRows = rows.filter((row) => !row.available);
  const groupedRows = availableRows
    .reduce<Array<{ strategyLabel: string; rows: MlBenchmarkRow[] }>>((groups, row) => {
      const existing = groups.find((group) => group.strategyLabel === row.strategy_label);
      if (existing) {
        existing.rows.push(row);
        return groups;
      }
      groups.push({ strategyLabel: row.strategy_label, rows: [row] });
      return groups;
    }, [])
    .sort((left, right) => strategyLabelRank(left.strategyLabel) - strategyLabelRank(right.strategyLabel));

  return (
    <div className="stack">
      {groupedRows.map((group) => (
        <div key={group.strategyLabel} className="panel table-panel">
          <div className="table-header">
            <div>
              <h3>Benchmark detallado: {group.strategyLabel}</h3>
              <span className="muted">Comparación por modelo dentro de una misma transformación del target.</span>
            </div>
          </div>
          <BenchmarkTable rows={group.rows} bestRegression={bestRegression} />
        </div>
      ))}

      {unavailableRows.length > 0 ? (
        <div className="panel table-panel">
          <div className="table-header">
            <div>
              <h3>Modelos sin benchmark en esta ejecución</h3>
              <span className="muted">Se mantienen visibles para que no desaparezcan del análisis aunque falte dependencia o entrenamiento.</span>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>Estado</th>
                  <th>Detalle</th>
                </tr>
              </thead>
              <tbody>
                {unavailableRows.map((row) => (
                  <tr key={`${row.model_name}-${row.strategy_name}`}>
                    <td>{row.model_name}</td>
                    <td>{row.strategy_label}</td>
                    <td>{row.notes[0] ?? 'Sin detalle adicional.'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function PredictionsChart({ rows, title }: { rows: Array<{ actual: number; predicted: number }>; title: string }) {
  if (rows.length === 0) {
    return (
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>{title}</h3>
            <span className="muted">No hay predicciones disponibles para este modelo en la ejecución actual.</span>
          </div>
        </div>
      </div>
    );
  }

  const chartData = [...rows]
    .sort((left, right) => left.actual - right.actual)
    .map((row, index) => ({
      index: index + 1,
      Real: row.actual,
      Predicción: row.predicted,
    }));

  return (
    <div className="panel table-panel">
      <div className="table-header">
        <div>
          <h3>{title}</h3>
          <span className="muted">
            Los casos del holdout se ordenan por valor real para visualizar la brecha entre realidad y predicción.
          </span>
        </div>
      </div>
      <div style={{ padding: '1rem 1.5rem', height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 16, right: 10, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.55} />
            <XAxis
              dataKey="index"
              tick={{ fill: 'var(--muted)', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
            <RechartsTooltip
              contentStyle={{
                backgroundColor: 'var(--bg)',
                borderRadius: '8px',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '10px' }} />
            <Line type="monotone" dataKey="Real" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            <Line type="monotone" dataKey="Predicción" stroke="#0f9b72" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TargetTransformationPanel({ payload }: Props) {
  const diagnostics = payload.target_transformation_diagnostics;

  if (!diagnostics || diagnostics.steps.length === 0) {
    return null;
  }

  return (
    <section className="stack">
      <div className="panel">
        <div className="section-header">
          <div>
            <h3>Proceso de transformación del target</h3>
            <p className="muted">
              Diagnóstico calculado sobre <strong>train</strong> para evitar leakage temporal. Muestra el efecto de `log1p`
              y de la normalización de outliers residuales.
            </p>
          </div>
        </div>
        <div className="chips" style={{ marginTop: '1rem' }}>
          {diagnostics.steps.map((step) => (
            <div key={step.step_key} className="chip info">
              <strong>{step.step_label}</strong>
              <span>
                Escala: {step.scale} | skew={formatMetric(step.stats.skew)} | outliers={step.stats.outlier_count} (
                {(step.stats.outlier_ratio * 100).toFixed(2)}%)
              </span>
              {step.notes.map((note) => (
                <small key={note}>{note}</small>
              ))}
            </div>
          ))}
        </div>
      </div>

      <BoxplotPanel
        series={diagnostics.boxplot_data}
        title="Diagramas de caja y bigote del pipeline"
        emptyMessage="No hay suficientes datos para boxplots del target transformado."
      />

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Estadística descriptiva antes y después</h3>
            <span className="muted">Resumen de tendencia central, dispersión y presión de outliers en cada etapa.</span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Etapa</th>
                <th>Escala</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>P25</th>
                <th>Median</th>
                <th>P75</th>
                <th>Max</th>
                <th>Skew</th>
                <th>Outliers</th>
              </tr>
            </thead>
            <tbody>
              {diagnostics.steps.map((step) => (
                <tr key={step.step_key}>
                  <td>{step.step_label}</td>
                  <td>{step.scale}</td>
                  <td>{step.stats.count}</td>
                  <td>{formatMetric(step.stats.mean)}</td>
                  <td>{formatMetric(step.stats.std)}</td>
                  <td>{formatMetric(step.stats.p25)}</td>
                  <td>{formatMetric(step.stats.p50)}</td>
                  <td>{formatMetric(step.stats.p75)}</td>
                  <td>{formatMetric(step.stats.max)}</td>
                  <td>{formatMetric(step.stats.skew)}</td>
                  <td>
                    {step.stats.outlier_count} ({(step.stats.outlier_ratio * 100).toFixed(2)}%)
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function RegressionView({ payload }: Props) {
  const comparison = payload.strategy_comparison;
  const bestRegression = comparison?.best_regression;
  const bestHeuristic = comparison?.best_heuristic;
  const benchmarkRows = [...payload.preprocessing_benchmarks].sort(
    (left, right) => sortByMetric(left.metrics.mae) - sortByMetric(right.metrics.mae)
  );
  const modelRows = [...payload.models].sort(
    (left, right) => sortByMetric(left.metrics.mae) - sortByMetric(right.metrics.mae)
  );
  const baselineRow = benchmarkRows.find((row) => row.strategy_name === 'baseline' || row.model_name === 'Global Median');
  const initialSelectedModelIndex = payload.models.findIndex((model) => isBestRegressionModel(model, bestRegression));
  const [selectedModelIndex, setSelectedModelIndex] = useState(initialSelectedModelIndex >= 0 ? initialSelectedModelIndex : 0);
  const selectedModel = payload.models[Math.min(selectedModelIndex, Math.max(payload.models.length - 1, 0))];
  const bestRegressionLabel = strategyEntryText(bestRegression);
  const bestHeuristicLabel = bestHeuristic?.model_name ?? 'n/a';
  const comparisonWinner = winnerLabel(comparison?.winner);
  const leaderMetricLabel =
    comparison?.winner === 'regression'
      ? 'MAE líder (regresión)'
      : comparison?.winner === 'heuristic'
        ? 'MAE líder (heurística)'
        : comparison?.winner === 'tie'
          ? 'MAE líder (empate)'
          : 'MAE líder';
  const leaderMetric =
    comparison?.winner === 'heuristic'
      ? bestHeuristic?.metrics.mae
      : comparison?.winner === 'regression'
        ? bestRegression?.metrics.mae
        : comparison?.winner === 'tie'
          ? Math.min(bestRegression?.metrics.mae ?? Number.POSITIVE_INFINITY, bestHeuristic?.metrics.mae ?? Number.POSITIVE_INFINITY)
          : bestRegression?.metrics.mae;
  const comparisonRows = [
    {
      label: 'Mejor regresión',
      selection: bestRegressionLabel,
      metrics: bestRegression?.metrics,
    },
    {
      label: 'Mejor heurística',
      selection: bestHeuristicLabel,
      metrics: bestHeuristic?.metrics,
    },
    {
      label: 'Baseline',
      selection: baselineRow ? `${baselineRow.model_name} · ${baselineRow.strategy_label}` : 'n/a',
      metrics: baselineRow?.metrics,
    },
  ];

  const topBenchmarkChart = modelRows
    .filter((row) => row.metrics.mae != null)
    .map((row) => ({
      name: row.model_name,
      MAE: row.metrics.mae ?? 0,
      MedAE: row.metrics.medae ?? 0,
      isBestRegression: isBestRegressionModel(row, bestRegression),
    }));

  const importanceData = selectedModel
    ? [...selectedModel.feature_effects]
        .sort((left, right) => right.coefficient - left.coefficient)
        .map((effect) => ({
          feature: effect.feature,
          value: effect.coefficient,
        }))
    : [];

  return (
    <section className="stack">
      <TargetTransformationPanel payload={payload} />

      <div className="panel">
        <div className="table-header">
          <div>
            <h3>Resumen comparativo</h3>
            <span className="muted">La pestaña resume el ganador global antes de abrir el detalle por segmento.</span>
          </div>
        </div>
        <div className="mini-grid">
          <article className="mini-panel">
            <strong>Train weeks</strong>
            <p>{payload.split.train_weeks.join(', ') || 'n/a'}</p>
          </article>
          <article className="mini-panel">
            <strong>Test weeks</strong>
            <p>{payload.split.test_weeks.join(', ') || 'n/a'}</p>
          </article>
          <article className="mini-panel">
            <strong>Mejor regresión</strong>
            <p>{bestRegressionLabel}</p>
          </article>
          <article className="mini-panel">
            <strong>Mejor heurística</strong>
            <p>{bestHeuristicLabel}</p>
          </article>
          <article className="mini-panel">
            <strong>Ganador</strong>
            <p>{comparisonWinner}</p>
          </article>
          <article className="mini-panel">
            <strong>{leaderMetricLabel}</strong>
            <p>{formatMetric(Number.isFinite(leaderMetric ?? Number.NaN) ? leaderMetric : null)}</p>
          </article>
          <article className="mini-panel">
            <strong>Gap MAE</strong>
            <p>{formatSignedMetric(comparison?.mae_gap)}</p>
            <small className="muted">heurística - regresión</small>
          </article>
        </div>
      </div>

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Comparativa principal</h3>
            <span className="muted">Resumen directo entre el ganador de regresión, la mejor heurística y el baseline robusto.</span>
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Referencia</th>
                <th>Selección</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MedAE</th>
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td>{row.selection}</td>
                  <td>{formatMetric(row.metrics?.mae)}</td>
                  <td>{formatMetric(row.metrics?.rmse)}</td>
                  <td>{formatMetric(row.metrics?.medae)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted" style={{ padding: '0.85rem 1.5rem 1.2rem' }}>
          {comparison?.narrative ?? 'Sin comparación disponible entre regresión, heurística y baseline.'}
        </p>
      </div>

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Benchmark por modelo</h3>
            <span className="muted">
              Una fila por modelo. Si un booster no corrió, queda listado aquí y en el detalle por modelo.
            </span>
          </div>
        </div>
        <div style={{ padding: '1rem 1.5rem', height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topBenchmarkChart} margin={{ top: 16, right: 24, left: 12, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.55} />
              <XAxis
                dataKey="name"
                interval={0}
                angle={-20}
                textAnchor="end"
                height={72}
                tick={{ fill: 'var(--muted)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: 'var(--bg)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                }}
              />
              <Legend />
              <Bar dataKey="MAE" radius={[4, 4, 0, 0]} fill="#21495f">
                {topBenchmarkChart.map((entry) => (
                  <Cell
                    key={`mae-${entry.name}`}
                    fill={entry.isBestRegression ? '#0f9b72' : '#21495f'}
                  />
                ))}
              </Bar>
              <Bar dataKey="MedAE" radius={[4, 4, 0, 0]} fill="#cf5c36" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <ModelBenchmarkTable models={modelRows} bestRegression={bestRegression} />
      </div>

      <StrategyBenchmarkTables rows={benchmarkRows} bestRegression={bestRegression} />

      {selectedModel ? (
        <>
          <div className="panel">
            <div className="table-header">
              <div>
                <h3>Detalle por modelo</h3>
                <span className="muted">Cada tarjeta representa la mejor estrategia retenida por modelo o su estado de disponibilidad.</span>
              </div>
            </div>
            <div className="tag-list" style={{ marginTop: '0.75rem' }}>
              {payload.models.map((model, index) => (
                <button
                  key={`${model.model_name}-${model.strategy_name ?? index}`}
                  type="button"
                  className={`tag ${selectedModelIndex === index ? 'active' : ''}`}
                  onClick={() => setSelectedModelIndex(index)}
                  style={
                    selectedModelIndex === index
                      ? { background: 'var(--accent)', color: '#fff', cursor: 'pointer' }
                      : { cursor: 'pointer' }
                  }
                >
                  {modelBadgeText(model)}
                </button>
              ))}
            </div>
            {selectedModel.notes?.length ? (
              <div className="chips" style={{ marginTop: '1rem' }}>
                {selectedModel.notes.map((note) => (
                  <div key={note} className="chip info">
                    <strong>Contexto</strong>
                    <span>{note}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>

          <div className="grid-2">
            <div className="panel table-panel">
              <div className="table-header">
                <div>
                  <h3>Importancia de variables</h3>
                  <span className="muted">Se muestran las variables más influyentes del modelo seleccionado.</span>
                </div>
              </div>
              {importanceData.length > 0 ? (
                <div style={{ padding: '1rem 1.5rem', height: 360 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={importanceData} layout="vertical" margin={{ top: 8, right: 20, left: 120, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" opacity={0.55} />
                      <XAxis type="number" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                      <YAxis
                        dataKey="feature"
                        type="category"
                        tick={{ fill: 'var(--text)', fontSize: 12 }}
                        width={120}
                        axisLine={false}
                        tickLine={false}
                      />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: 'var(--bg)',
                          borderRadius: '8px',
                          border: '1px solid var(--border)',
                          color: 'var(--text)',
                        }}
                      />
                      <Bar dataKey="value" fill="#0f9b72" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="muted" style={{ padding: '0 1.5rem 1.5rem' }}>
                  Este modelo no expone importancias legibles en el pipeline actual.
                </p>
              )}
            </div>

            <PredictionsChart rows={selectedModel.predictions} title="Alineación de predicciones" />
          </div>

          {selectedModel.tree_structure ? (
            <div className="panel table-panel">
              <div className="table-header">
                <div>
                  <h3>Árbol de decisión</h3>
                  <span className="muted">Solo aparece cuando el modelo seleccionado es un árbol interpretable.</span>
                </div>
              </div>
              <div style={{ padding: '2rem', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
                <TreeNodeView node={selectedModel.tree_structure} />
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <section className="panel">
          <h3>Regresiones</h3>
          <p className="muted">No hay modelos de regresión disponibles en esta ejecución.</p>
        </section>
      )}

      <div className="stack">
        {payload.segment_reports.map((report) => (
          <div key={report.family_key} className="panel table-panel">
            <div className="table-header">
              <div>
                <h3>Segmentación por {report.family_label}</h3>
                <span className="muted" style={{ display: 'block' }}>
                  MAE mejor regresión: <strong>{bestRegressionLabel}</strong>.
                </span>
                <span className="muted">
                  Se reportan solo segmentos representativos; el resto se consolida en <strong>Other</strong>.
                </span>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Segmento</th>
                    <th>Train</th>
                    <th>Test</th>
                    <th>Actual mean</th>
                    <th>Actual median</th>
                    <th>MAE mejor regresión</th>
                    <th>MAE heurística</th>
                    <th>MAE baseline</th>
                  </tr>
                </thead>
                <tbody>
                  {report.rows.map((row) => (
                    <tr key={`${report.family_key}-${row.segment}`}>
                      <td>{row.segment}</td>
                      <td>{row.train_count}</td>
                      <td>{row.test_count}</td>
                      <td>{formatMetric(row.actual_mean)}</td>
                      <td>{formatMetric(row.actual_median)}</td>
                      <td>{formatMetric(row.regression_mae)}</td>
                      <td>{formatMetric(row.heuristic_mae)}</td>
                      <td>{formatMetric(row.baseline_mae)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function HeuristicView({ payload }: Props) {
  const heuristicRows = [...payload.heuristic_models].sort(
    (left, right) => (left.metrics.mae ?? Number.POSITIVE_INFINITY) - (right.metrics.mae ?? Number.POSITIVE_INFINITY)
  );
  const [selectedHeuristicIndex, setSelectedHeuristicIndex] = useState(0);
  const selectedHeuristic = heuristicRows[Math.min(selectedHeuristicIndex, Math.max(heuristicRows.length - 1, 0))];

  const rankingData = heuristicRows.slice(0, 8).map((row) => ({
    name: row.model_name,
    MAE: row.metrics.mae ?? 0,
    MedAE: row.metrics.medae ?? 0,
  }));

  return (
    <section className="stack">
      <div className="panel">
        <div className="mini-grid">
          <article className="mini-panel">
            <strong>Mejor heurística</strong>
            <p>{payload.strategy_comparison?.best_heuristic?.model_name ?? 'n/a'}</p>
          </article>
          <article className="mini-panel">
            <strong>MAE heurística líder</strong>
            <p>{formatMetric(payload.strategy_comparison?.best_heuristic?.metrics.mae)}</p>
          </article>
          <article className="mini-panel">
            <strong>Comparación</strong>
            <p>{payload.strategy_comparison?.narrative ?? 'Sin comparación disponible'}</p>
          </article>
          <article className="mini-panel">
            <strong>Gap MAE</strong>
            <p>{formatMetric(payload.strategy_comparison?.mae_gap)}</p>
          </article>
        </div>
      </div>

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Ranking heurístico</h3>
            <span className="muted">Se comparan reglas por segmento, baseline robusto y backoff jerárquico.</span>
          </div>
        </div>
        <div style={{ padding: '1rem 1.5rem', height: 340 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rankingData} margin={{ top: 16, right: 24, left: 12, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.55} />
              <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
              <RechartsTooltip
                contentStyle={{
                  backgroundColor: 'var(--bg)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                }}
              />
              <Legend />
              <Bar dataKey="MAE" fill="#cf5c36" radius={[4, 4, 0, 0]} />
              <Bar dataKey="MedAE" fill="#21495f" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Heurística</th>
                <th>Familia</th>
                <th>MAE train</th>
                <th>MAE test</th>
                <th>MedAE test</th>
              </tr>
            </thead>
            <tbody>
              {heuristicRows.map((row) => (
                <tr key={row.model_name}>
                  <td>{row.model_name}</td>
                  <td>{row.family_label}</td>
                  <td>{formatMetric(row.train_metrics.mae)}</td>
                  <td>{formatMetric(row.metrics.mae)}</td>
                  <td>{formatMetric(row.metrics.medae)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selectedHeuristic ? (
        <>
          <div className="panel">
            <div className="table-header">
              <div>
                <h3>Detalle heurístico</h3>
                <span className="muted">
                  Selecciona una heurística para revisar su lógica y comportamiento en el holdout.
                </span>
              </div>
            </div>
            <div className="tag-list" style={{ marginTop: '0.75rem' }}>
              {heuristicRows.map((row, index) => (
                <button
                  key={`${row.model_name}-${row.family_key}`}
                  type="button"
                  className={`tag ${selectedHeuristicIndex === index ? 'active' : ''}`}
                  onClick={() => setSelectedHeuristicIndex(index)}
                  style={
                    selectedHeuristicIndex === index
                      ? { background: 'var(--accent)', color: '#fff', cursor: 'pointer' }
                      : { cursor: 'pointer' }
                  }
                >
                  {heuristicBadgeText(row)}
                </button>
              ))}
            </div>
            <p className="muted" style={{ marginTop: '1rem' }}>{selectedHeuristic.rule_summary}</p>
          </div>

          <div className="grid-2">
            <PredictionsChart rows={selectedHeuristic.predictions} title="Predicciones heurísticas" />

            <div className="panel table-panel">
              <div className="table-header">
                <div>
                  <h3>Uso de reglas</h3>
                  <span className="muted">Distribución de activación de cada nivel de la heurística.</span>
                </div>
              </div>
              <div className="mini-grid" style={{ padding: '1rem 1.5rem 1.5rem' }}>
                {selectedHeuristic.tier_usage.map((tier) => (
                  <article key={tier.source} className="mini-panel">
                    <strong>{tier.source}</strong>
                    <p>{tier.count}</p>
                  </article>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : (
        <section className="panel">
          <h3>Heurísticas</h3>
          <p className="muted">No hay heurísticas disponibles en esta ejecución.</p>
        </section>
      )}
    </section>
  );
}

function LearningView({ payload }: Props) {
  return (
    <section className="stack">
      {payload.learning_sections.map((section) => (
        <article key={section.slug} className="panel">
          <div className="section-header">
            <div>
              <h3>{section.title}</h3>
              <p className="muted">{section.summary}</p>
            </div>
          </div>
          <ul className="plain-list">
            {section.bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
        </article>
      ))}
    </section>
  );
}

export default function MlOverviewPanel({ payload }: Props) {
  const [tab, setTab] = useState<TabKey>('regresiones');

  return (
    <section className="stack">
      <div className="panel">
        <div className="section-header">
          <div>
            <h3>Semana 2.1</h3>
            <p className="muted">
              Comparación entre regresiones, heurísticas y aprendizajes académicos sobre el mismo holdout temporal.
            </p>
          </div>
        </div>

        <div className="segmented" style={{ marginTop: '0.75rem', flexWrap: 'wrap' }}>
          {(['regresiones', 'heuristicas', 'aprendizajes'] as TabKey[]).map((candidate) => (
            <button
              key={candidate}
              type="button"
              className={tab === candidate ? 'active' : ''}
              onClick={() => setTab(candidate)}
            >
              {tabLabel(candidate)}
            </button>
          ))}
        </div>
      </div>

      <WarningChips warnings={payload.warnings} />

      {tab === 'regresiones' ? <RegressionView payload={payload} /> : null}
      {tab === 'heuristicas' ? <HeuristicView payload={payload} /> : null}
      {tab === 'aprendizajes' ? <LearningView payload={payload} /> : null}
    </section>
  );
}
