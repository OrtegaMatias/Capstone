import { useMemo, useRef } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import type { VariabilityRow } from '../api/types';
import { exportSvgInContainerAsPng } from '../utils/chartExport';

type MetricKey = 'entropy' | 'gini_impurity' | 'coefficient_variation' | 'custom_index';

type Props = {
  rows: VariabilityRow[];
  metric: MetricKey;
};

export default function VariabilityRanking({ rows, metric }: Props) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  const data = useMemo(
    () =>
      rows
        .map((row) => ({ column: row.column, score: row[metric] }))
        .filter((row): row is { column: string; score: number } => row.score !== null && row.score !== undefined && Number.isFinite(row.score))
        .sort((a, b) => b.score - a.score)
        .slice(0, 20),
    [rows, metric],
  );

  const handleExport = async () => {
    if (data.length === 0) return;
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `variability-ranking-${metric}`);
    } catch (error) {
      console.error('chart export failed', error);
    }
  };

  const metricHint =
    metric === 'entropy' || metric === 'gini_impurity'
      ? 'Esta métrica aplica principalmente a columnas categóricas.'
      : metric === 'coefficient_variation'
        ? 'Esta métrica aplica a columnas numéricas con media distinta de cero.'
        : 'Custom index depende del modo configurado (freq_only u ordinal_map).';

  if (data.length === 0) {
    return (
      <div className="panel chart-panel">
        <div className="chart-header">
          <h3>Ranking por {metric}</h3>
        </div>
        <p className="muted">No hay columnas con valores válidos para esta métrica.</p>
        <p className="muted">{metricHint}</p>
      </div>
    );
  }

  return (
    <div className="panel chart-panel">
      <div className="chart-header">
        <h3>Ranking por {metric}</h3>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()}>
          Descargar PNG HD
        </button>
      </div>
      <div ref={chartRef} className="chart-export-area">
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={data} layout="vertical" margin={{ left: 32 }}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="column" width={140} />
            <Tooltip />
            <Bar dataKey="score" fill="#1367c8" radius={[0, 6, 6, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
