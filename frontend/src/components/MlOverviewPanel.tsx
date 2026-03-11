import { useState, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  LineChart, Line, Cell, ScatterChart, Scatter, ReferenceLine, ZAxis
} from 'recharts';
import html2canvas from 'html2canvas';
import type { MlEvaluationSummary, MlModelResult, TreeNode } from '../api/types';
import WarningChips from './WarningChips';

type Props = {
  payload: MlEvaluationSummary;
};

function TreeNodeView({ node, depth = 0 }: { node: TreeNode; depth?: number }) {
  if (!node) return null;
  const isLeaf = node.type === 'leaf';
  const bgColor = isLeaf ? '#10b981' : '#6366f1';
  const maxDepthForCollapse = 3;
  const [collapsed, setCollapsed] = useState(depth >= maxDepthForCollapse && !isLeaf);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: isLeaf ? 90 : 160 }}>
      <div
        onClick={() => !isLeaf && setCollapsed(!collapsed)}
        style={{
          backgroundColor: bgColor,
          color: 'white',
          padding: '6px 10px',
          borderRadius: isLeaf ? '20px' : '8px',
          fontSize: '11px',
          textAlign: 'center',
          cursor: isLeaf ? 'default' : 'pointer',
          maxWidth: 180,
          lineHeight: 1.4,
          boxShadow: '0 2px 6px rgba(0,0,0,0.15)',
          opacity: collapsed ? 0.7 : 1,
        }}
      >
        {isLeaf ? (
          <>
            <div style={{ fontWeight: 700 }}>≈ {node.value.toFixed(1)}d</div>
            <div style={{ fontSize: '9px', opacity: 0.8 }}>n={node.samples}</div>
          </>
        ) : (
          <>
            <div style={{ fontWeight: 700 }}>{node.feature}</div>
            <div>≤ {node.threshold}</div>
            <div style={{ fontSize: '9px', opacity: 0.8 }}>n={node.samples}{collapsed ? ' [+]' : ''}</div>
          </>
        )}
      </div>

      {!isLeaf && !collapsed && (node.left || node.right) && (
        <div style={{ display: 'flex', gap: '4px', marginTop: '4px', position: 'relative' }}>
          <div style={{ borderLeft: '2px solid var(--border)', borderTop: '2px solid var(--border)', width: 20, height: 16 }} />
          <div style={{ borderRight: '2px solid var(--border)', borderTop: '2px solid var(--border)', width: 20, height: 16 }} />
        </div>
      )}

      {!isLeaf && !collapsed && (node.left || node.right) && (
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <span style={{ fontSize: '9px', color: 'var(--muted)', marginBottom: 2 }}>Sí</span>
            {node.left && <TreeNodeView node={node.left} depth={depth + 1} />}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <span style={{ fontSize: '9px', color: 'var(--muted)', marginBottom: 2 }}>No</span>
            {node.right && <TreeNodeView node={node.right} depth={depth + 1} />}
          </div>
        </div>
      )}
    </div>
  );
}

function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function MlOverviewPanel({ payload }: Props) {
  const [selectedModelIndex, setSelectedModelIndex] = useState(0);

  if (!payload.models || payload.models.length === 0) {
    return (
      <section className="panel">
        <h3>Evaluacion ML</h3>
        <p className="muted">No hay modelos disponibles.</p>
        <WarningChips warnings={payload.warnings} />
      </section>
    );
  }

  const selectedModel = payload.models[selectedModelIndex];

  const predictionCsv = [
    'model_name,week,actual,predicted',
    ...payload.models.flatMap((m) =>
      m.predictions.map((row) => `${m.model_name},${row.week},${row.actual},${row.predicted}`)
    ),
  ].join('\n');

  const comparisonData = payload.models.map((m) => ({
    name: m.model_name,
    MAE: m.metrics.mae,
    RMSE: m.metrics.rmse,
  }));

  const bestModelName = comparisonData.length > 0
    ? comparisonData.reduce((prev, curr) => (prev.MAE! < curr.MAE! ? prev : curr)).name
    : '';

  const totalImportance = selectedModel.feature_effects.reduce((sum, e) => sum + e.coefficient, 0);
  const importanceData = [...selectedModel.feature_effects]
    .map(e => ({
      feature: e.feature,
      percentage: totalImportance > 0 ? (e.coefficient / totalImportance) * 100 : 0,
      raw_coefficient: e.coefficient,
    }))
    .sort((a, b) => b.percentage - a.percentage);
  const topFeature1 = importanceData[0]?.feature || '';
  const topFeature2 = importanceData[1]?.feature || '';

  const holdoutData = [...selectedModel.predictions]
    .sort((a, b) => a.actual - b.actual)
    .map((p, index) => ({
      index: index + 1,
      Real: p.actual,
      Predicción: p.predicted,
    }));

  const chartRef1 = useRef<HTMLDivElement>(null);
  const chartRef2 = useRef<HTMLDivElement>(null);
  const chartRef3 = useRef<HTMLDivElement>(null);
  const tableRef1 = useRef<HTMLDivElement>(null);

  const downloadChartAsPng = async (ref: React.RefObject<HTMLDivElement | null>, filename: string) => {
    if (!ref.current) return;
    try {
      const canvas = await html2canvas(ref.current, { backgroundColor: null, scale: 2 });
      const url = canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = filename;
      link.href = url;
      link.click();
    } catch (err) {
      console.error('Failed to export chart', err);
    }
  };

  return (
    <section className="stack">
      <div className="panel">
        <div className="section-header">
          <div>
            <h3>Evaluacion ML</h3>
            <p className="muted">
              Comparación de modelos no paramétricos sobre la última semana observada.
            </p>
          </div>
          <button type="button" onClick={() => downloadCsv(`${payload.week_id}-predictions.csv`, predictionCsv)}>
            Descargar predicciones
          </button>
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
            <strong>Filas train</strong>
            <p>{payload.split.train_rows}</p>
          </article>
          <article className="mini-panel">
            <strong>Filas test</strong>
            <p>{payload.split.test_rows}</p>
          </article>
        </div>
      </div>

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Comparación de Modelos</h3>
            <span className="muted">
              El modelo <strong>{bestModelName}</strong> tiene el menor error medio absoluto (MAE), siendo el más confiable.
            </span>
          </div>
          <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadChartAsPng(chartRef1, 'comparacion_modelos.png')}>
            ⬇️ Descargar PNG
          </button>
        </div>
        <div ref={chartRef1} style={{ padding: '1rem 1.5rem', marginTop: '0.5rem', height: 350, position: 'relative' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={comparisonData}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
              <XAxis dataKey="name" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
              <RechartsTooltip
                cursor={{ fill: 'var(--bg-elevated)', opacity: 0.4 }}
                contentStyle={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', color: 'var(--text)' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar dataKey="MAE" radius={[4, 4, 0, 0]} maxBarSize={60}>
                {comparisonData.map((entry, index) => (
                  <Cell key={`cell-mae-${index}`} fill={entry.name === bestModelName ? '#6366f1' : '#94a3b8'} opacity={entry.name === bestModelName ? 1 : 0.6} />
                ))}
              </Bar>
              <Bar dataKey="RMSE" radius={[4, 4, 0, 0]} maxBarSize={60}>
                {comparisonData.map((entry, index) => (
                  <Cell key={`cell-rmse-${index}`} fill={entry.name === bestModelName ? '#f43f5e' : '#cbd5e1'} opacity={entry.name === bestModelName ? 1 : 0.6} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '0.5rem 1rem 0' }}>
          <button
            type="button"
            className="outline-btn"
            style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
            onClick={() => downloadChartAsPng(tableRef1, 'tabla_comparacion_modelos.png')}
          >
            ⬇️ Descargar PNG
          </button>
        </div>
        <div ref={tableRef1} style={{ padding: '1rem' }}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>R²</th>
                <th>Baseline MAE</th>
              </tr>
            </thead>
            <tbody>
              {payload.models.map((model) => (
                <tr key={model.model_name}>
                  <td>
                    <strong>{model.model_name} {model.model_name === bestModelName ? '🏆' : ''}</strong>
                  </td>
                  <td>{model.metrics.mae?.toFixed(4) ?? '-'}</td>
                  <td>{model.metrics.rmse?.toFixed(4) ?? '-'}</td>
                  <td>{model.metrics.r2?.toFixed(4) ?? '-'}</td>
                  <td>{model.metrics.baseline_mae?.toFixed(4) ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        </div>
        
        <div style={{ padding: '1.5rem', background: 'var(--bg-elevated)', borderTop: '1px solid var(--border)', fontSize: '0.9rem' }}>
          <h4 style={{ marginBottom: '0.5rem', color: 'var(--text)' }}>¿Cómo leer estos resultados?</h4>
          <ul style={{ paddingLeft: '1.5rem', color: 'var(--muted)' }}>
            <li style={{ marginBottom: '0.5rem' }}><strong>MAE (Error Absoluto Medio):</strong> Es el margen de error promedio puro en días. Si el MAE es 4.5, significa que el modelo se equivoca en promedio por ±4.5 días al estimar.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>RMSE (Raíz del Error Cuadrático Medio):</strong> Penaliza los errores enormes. Si el RMSE es altísimo pero el MAE es bajo, significa que el modelo suele acertar, pero cuando falla, se equivoca por meses enteros.</li>
            <li style={{ marginBottom: '0.5rem' }}><strong>R² (R-Cuadrado):</strong> Mide la precisión general (0.0 a 1.0). Un 1.0 es perfección absoluta. Si es 0.0 (o negativo), significa que la predicción no es mejor que tirar los dados.</li>
            <li><strong>Baseline MAE:</strong> Es la trampa. Si decimos siempre el "Promedio Histórico" a ciegas, ese es nuestro error base. Tus modelos de Machine Learning (arriba) <strong>deben tener un MAE menor al Baseline</strong>, de lo contrario la IA no está aprendiendo nada útil.</li>
          </ul>
        </div>
      </div>

      <WarningChips warnings={payload.warnings} />

      <div className="panel">
        <h3>Analizar Modelo Detallado</h3>
        <p className="muted">Selecciona un modelo para ver sus predicciones y la importancia de las variables (feature importances).</p>
        <div className="tag-list" style={{ marginTop: '0.5rem', marginBottom: '1rem' }}>
          {payload.models.map((model, index) => (
            <button
              key={model.model_name}
              type="button"
              className={`tag ${selectedModelIndex === index ? 'active' : ''}`}
              onClick={() => setSelectedModelIndex(index)}
              style={selectedModelIndex === index ? { background: 'var(--accent)', color: 'white', cursor: 'pointer' } : { cursor: 'pointer' }}
            >
              {model.model_name}
            </button>
          ))}
        </div>

        <div className="grid-2">
          <div className="panel table-panel">
            <div className="table-header">
              <div>
                <h3>Importancia de Variables</h3>
                <span className="muted" style={{ display: 'block', maxWidth: '300px' }}>
                  Las variables que más definen los días de depósito son <strong>{topFeature1}</strong> y <strong>{topFeature2}</strong>.
                </span>
              </div>
              <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadChartAsPng(chartRef2, `importancia_${selectedModel.model_name}.png`)}>
                ⬇️ Descargar PNG
              </button>
            </div>
            <div ref={chartRef2} style={{ padding: '1rem 1.5rem', height: 380, marginTop: '0.5rem', position: 'relative' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={importanceData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" opacity={0.5} />
                  <XAxis type="number" unit="%" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                  <YAxis dataKey="feature" type="category" tick={{ fill: 'var(--text)', fontSize: 12 }} axisLine={false} tickLine={false} width={100} />
                  <RechartsTooltip
                    cursor={{ fill: 'var(--bg-elevated)', opacity: 0.4 }}
                    contentStyle={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', color: 'var(--text)' }}
                    formatter={(value: number) => [`${value.toFixed(2)}%`, 'Importancia']}
                  />
                  <Bar dataKey="percentage" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Importancia</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedModel.feature_effects.map((effect) => (
                    <tr key={effect.feature}>
                      <td>{effect.feature}</td>
                      <td>{effect.coefficient.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel table-panel">
            <div className="table-header">
              <div>
                <h3>Alineación de Predicciones</h3>
                <span className="muted" style={{ display: 'block', maxWidth: '420px' }}>
                  Los {holdoutData.length.toLocaleString()} contenedores del test están ordenados por su atraso real (gris punteada). La línea verde muestra lo que el modelo predice. Si ambas líneas se superponen, el modelo es preciso.
                </span>
              </div>
              <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadChartAsPng(chartRef3, `predicciones_${selectedModel.model_name}.png`)}>
                ⬇️ Descargar PNG
              </button>
            </div>
            <div ref={chartRef3} style={{ padding: '1rem 1.5rem', marginTop: '0.5rem', height: 400, position: 'relative' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={holdoutData}
                  margin={{ top: 20, right: 10, left: 0, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                  <XAxis dataKey="index" tick={{ fill: 'var(--muted)', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'Contenedores (ordenados por atraso real)', position: 'insideBottom', offset: -12, fill: 'var(--muted)' }} />
                  <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} label={{ value: 'Días en Depósito', angle: -90, position: 'insideLeft', fill: 'var(--muted)' }} />
                  <RechartsTooltip
                    contentStyle={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', color: 'var(--text)' }}
                    labelFormatter={(label) => `Contenedor #${label}`}
                  />
                  <Legend wrapperStyle={{ paddingTop: '10px' }} />
                  <Line type="monotone" dataKey="Real" stroke="#64748b" strokeWidth={2} strokeDasharray="5 5" dot={false} activeDot={{ r: 4 }} />
                  <Line type="monotone" dataKey="Predicción" stroke="#10b981" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>week</th>
                    <th>actual</th>
                    <th>predicted</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedModel.predictions.map((row, index) => (
                    <tr key={`${row.week}-${index}`}>
                      <td>{row.week}</td>
                      <td>{row.actual.toFixed(2)}</td>
                      <td>{row.predicted.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Features usadas</h3>
        <p className="muted">El modelo mezcla variables numericas y categoricas derivadas del dataset canonico.</p>
        <div className="tag-list">
          {payload.feature_columns.map((feature) => (
            <span key={feature} className="tag">
              {feature}
            </span>
          ))}
        </div>
      </div>

      {selectedModel.tree_structure ? (
        <div className="panel table-panel">
          <div className="table-header">
            <div>
              <h3>🌳 Estructura del Árbol de Decisión</h3>
              <span className="muted" style={{ display: 'block', maxWidth: '500px' }}>
                Cada nodo muestra la variable y umbral de división. Las hojas verdes muestran el valor predicho promedio en días. Haz clic en un nodo para expandir/colapsar sus ramas.
              </span>
            </div>
            <button
              type="button"
              className="outline-btn"
              style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }}
              onClick={() => {
                const el = document.getElementById('tree-viz');
                if (!el) return;
                html2canvas(el, { backgroundColor: '#ffffff', scale: 2 }).then(canvas => {
                  const url = canvas.toDataURL('image/png');
                  const link = document.createElement('a');
                  link.download = `arbol_decision_${selectedModel.model_name}.png`;
                  link.href = url;
                  link.click();
                });
              }}
            >
              ⬇️ Descargar PNG
            </button>
          </div>
          <div id="tree-viz" style={{ padding: '2rem', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
            <TreeNodeView node={selectedModel.tree_structure} />
          </div>
        </div>
      ) : null}
    </section>
  );
}
