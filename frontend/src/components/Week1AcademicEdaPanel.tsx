import type { OpticsSourceResult, WeekAcademicEDAResponse, WeekClusteringResponse, Week1SourceEdaSection } from '../api/types';
import BoxplotPanel from './BoxplotPanel';
import DataTable from './DataTable';
import PlotlyFigureCard from './PlotlyFigureCard';
import WarningChips from './WarningChips';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, ZAxis, Legend,
  ComposedChart, Area, ReferenceArea, BarChart, Bar
} from 'recharts';
import html2canvas from 'html2canvas';

type Props = {
  payload: WeekAcademicEDAResponse;
  clustering?: WeekClusteringResponse;
  clusteringLoading?: boolean;
};

function formatValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-';
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}%`;
}

const CLUSTER_COLORS = ['#24475b', '#b06a31', '#4e7f52', '#8c4f73', '#90713d', '#3e6d84', '#b44d3a', '#5d5b97'];

function clusterColor(clusterId: number | null | undefined, isNoise = false): string {
  if (isNoise || clusterId === null || clusterId === undefined) return '#98a4aa';
  return CLUSTER_COLORS[Math.abs(clusterId) % CLUSTER_COLORS.length];
}

const downloadElementAsPng = async (id: string, filename: string) => {
  const el = document.getElementById(id);
  if (!el) return;
  try {
    const canvas = await html2canvas(el, { backgroundColor: null, scale: 2 });
    const url = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = filename;
    link.href = url;
    link.click();
  } catch (err) {
    console.error('Failed to export chart', err);
  }
};

function OpticsRechartsCards({ sourceKey, optics }: { sourceKey: string; optics: OpticsSourceResult }) {
  const descriptionMap = new Map<string, string>();
  optics.cluster_summary.forEach(row => descriptionMap.set(row.cluster_label, row.description));

  const clusters = new Map<string, any[]>();
  optics.embedding_points.forEach(point => {
    const arr = clusters.get(point.cluster_label) || [];
    arr.push({
      ...point,
      description: descriptionMap.get(point.cluster_label) || 'Sin descripción',
    });
    clusters.set(point.cluster_label, arr);
  });
  
  const umapSeries: any[] = [];
  if (clusters.has('ruido')) {
    umapSeries.push({ name: 'ruido', data: clusters.get('ruido'), color: '#98a4aa', isNoise: true });
    clusters.delete('ruido');
  }
  
  const allReachData = optics.reachability.map(p => {
    const cSummary = optics.cluster_summary.find(s => s.cluster_label === p.cluster_label);
    return {
      order: p.order,
      reachability: p.reachability,
      cluster_label: p.cluster_label,
      is_noise: p.cluster_label === 'ruido',
      fill: clusterColor(cSummary?.cluster_id, p.cluster_label === 'ruido')
    };
  });

  Array.from(clusters.entries()).forEach(([name, data]) => {
     umapSeries.push({ name, data, color: clusterColor(data[0].cluster_id, false), isNoise: false });
  });

  const umapId = `umap-${sourceKey}`;
  const reachId = `reach-${sourceKey}`;

  const CustomUmapTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', padding: '10px', color: 'var(--text)', fontSize: '13px', maxWidth: '300px' }}>
          <strong style={{ color: payload[0].color }}>{data.cluster_label}</strong>
          <p style={{ margin: '4px 0', fontSize: '12px' }}><i>{data.description}</i></p>
          <div>Peso visual: {data.display_weight.toFixed(2)}</div>
          <div style={{ color: 'var(--muted)' }}>X: {data.x.toFixed(3)} | Y: {data.y.toFixed(3)}</div>
        </div>
      );
    }
    return null;
  };

  const CustomReachTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', padding: '10px', color: 'var(--text)', fontSize: '13px' }}>
          <div>Cluster: <strong style={{ color: data.fill }}>{data.cluster_label}</strong></div>
          <div>Orden: {data.order}</div>
          <div>Reachability: {data.reachability.toFixed(4)}</div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="grid-1 stack" style={{ marginBottom: '2rem' }}>
      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Proyección UMAP 2D</h3>
            <span className="muted" style={{ display: 'block', maxWidth: '600px', fontSize: '0.9rem' }}>
              <strong>Visualización de Embeddings:</strong> Los puntos agrupados representan contenedores con características muy similares, colapsadas en 2 dimensiones. Los grises son valores atípicos (ruido) sin patrón definido.<br/>
              <em>Trustworthiness: {formatValue(optics.embedding_quality.trustworthiness)} | Coords: {optics.overlap_stats.unique_coordinates_raw} → {optics.overlap_stats.unique_coordinates_display}</em>
            </span>
          </div>
          <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadElementAsPng(umapId, `umap_${sourceKey}.png`)}>
            ⬇️ Descargar PNG
          </button>
        </div>
        <div id={umapId} style={{ padding: '1rem', marginTop: '0.5rem', height: 450, position: 'relative' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.4} stroke="var(--border)"/>
              <XAxis type="number" dataKey="x" name="UMAP 1" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} label={{ value: 'Dimensión UMAP 1', position: 'insideBottom', offset: -15, fill: 'var(--muted)' }} />
              <YAxis type="number" dataKey="y" name="UMAP 2" tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} label={{ value: 'Dimensión UMAP 2', angle: -90, position: 'insideLeft', fill: 'var(--muted)' }} />
              <ZAxis type="number" dataKey="display_weight" range={[20, 300]} />
              <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomUmapTooltip />} />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              {umapSeries.map((series) => (
                <Scatter key={series.name} name={series.name} data={series.data} fill={series.color} opacity={series.isNoise ? 0.3 : 0.85} shape="circle" />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel table-panel">
        <div className="table-header">
          <div>
            <h3>Alcanzabilidad OPTICS (Densidad Pura)</h3>
            <span className="muted" style={{ display: 'block', maxWidth: '600px', fontSize: '0.9rem' }}>
              <strong>Estructura matemática:</strong> Los "valles" coloreados indican grupos de contenedores muy densos y predecibles. Los "picos" grises separan a los clusters entre sí marcando los bordes del ruido.
            </span>
          </div>
          <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadElementAsPng(reachId, `reachability_${sourceKey}.png`)}>
            ⬇️ Descargar PNG
          </button>
        </div>
        <div id={reachId} style={{ padding: '1rem', marginTop: '0.5rem', height: 350, position: 'relative' }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={allReachData} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.4} vertical={false} stroke="var(--border)" />
              <XAxis dataKey="order" tick={{ fill: 'var(--muted)', fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'Orden Estocástico OPTICS', position: 'insideBottom', offset: -10, fill: 'var(--muted)' }} />
              <YAxis tick={{ fill: 'var(--muted)' }} axisLine={false} tickLine={false} label={{ value: 'Distancia de Alcanzabilidad', angle: -90, position: 'insideLeft', offset: -10, fill: 'var(--muted)' }} />
              
              {optics.cluster_ranges.map((range, i) => (
                <ReferenceArea 
                  key={`area-${i}`} 
                  x1={range.start_order} 
                  x2={range.end_order} 
                  fill={clusterColor(range.cluster_id)} 
                  fillOpacity={0.15} 
                />
              ))}

              <RechartsTooltip content={<CustomReachTooltip />} />
              <Area type="monotone" dataKey="reachability" stroke="#c6c0b8" fill="none" strokeWidth={1} isAnimationActive={false} />
              <Scatter dataKey="reachability" isAnimationActive={false}>
                {allReachData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} opacity={entry.is_noise ? 0.2 : 0.9} />
                ))}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function SourceOverviewCards({ source }: { source: Week1SourceEdaSection }) {
  const metrics = [
    ['Filas', source.overview_metrics.row_count],
    ['Columnas', source.overview_metrics.column_count],
    ['Completitud', formatPct(source.overview_metrics.completeness_pct)],
    ['Celdas NA', source.overview_metrics.missing_cells],
    ['Duplicados', source.overview_metrics.duplicate_rows_exact],
    ['Numéricas', source.overview_metrics.numeric_variables],
    ['Categóricas', source.overview_metrics.categorical_variables],
  ];

  return (
    <div className="academic-metrics">
      {metrics.map(([label, value]) => (
        <article key={label} className="academic-metric-card">
          <span>{label}</span>
          <strong>{typeof value === 'number' ? value.toLocaleString() : value}</strong>
        </article>
      ))}
    </div>
  );
}

function CorrelationHeatmap({
  title,
  labels,
  matrix,
  fileBaseName,
}: {
  title: string;
  labels: string[];
  matrix: Array<Array<number | null>>;
  fileBaseName: string;
}) {
  if (!labels.length) return null;
  return (
    <PlotlyFigureCard
      title={title}
      fileBaseName={fileBaseName}
      data={[
        {
          type: 'heatmap',
          x: labels,
          y: labels,
          z: matrix,
          colorscale: [
            [0, '#14354a'],
            [0.5, '#f5f0e7'],
            [1, '#b46a2f'],
          ],
          zmin: -1,
          zmax: 1,
        },
      ]}
      layout={{
        xaxis: { automargin: true },
        yaxis: { automargin: true },
      }}
    />
  );
}

function SourceSection({ sourceKey, source }: { sourceKey: string; source: Week1SourceEdaSection }) {
  const numericBoxplotSeries = source.univariate_numeric.variables.map((item) => ({
    feature: item.column,
    groups: [{ group: source.title, values: item.values }],
  }));

  return (
    <section id={sourceKey} className="stack">
      <div className="academic-section-head panel">
        <div>
          <span className="eyebrow">{source.title}</span>
          <h2>{source.title}</h2>
          <p className="muted">
            {source.target_variable
              ? `Fuente con variable objetivo ${source.target_variable}.`
              : 'Fuente descriptiva sin variable objetivo numérica principal.'}
          </p>
        </div>
      </div>

      <div className="panel">
        <h3>Resumen ejecutivo</h3>
        <SourceOverviewCards source={source} />
      </div>

      <div className="grid-2">
        <div className="panel table-panel">
          <h3>Auditoría y estructura</h3>
          <div className="table-wrap">
            <table>
              <tbody>
                {Object.entries(source.dataset_audit.variable_type_counts).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{value}</td>
                  </tr>
                ))}
                <tr>
                  <td>Identificadores únicos</td>
                  <td>{source.dataset_audit.unique_identifier_columns.join(', ') || 'ninguno'}</td>
                </tr>
                <tr>
                  <td>Completitud</td>
                  <td>{formatPct(source.dataset_audit.completeness_pct)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel table-panel">
          <h3>Nulos y cardinalidad</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>% nulo</th>
                  <th>Únicos</th>
                </tr>
              </thead>
              <tbody>
                {source.dataset_audit.variable_dictionary.map((item) => (
                  <tr key={item.name}>
                    <td>{item.name}</td>
                    <td>{formatPct(item.missing_pct)}</td>
                    <td>{item.unique_values ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="panel table-panel">
        <h3>Diccionario de variables</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Variable</th>
                <th>Tipo lógico</th>
                <th>Tipo observado</th>
                <th>Rol</th>
                <th>% nulo</th>
                <th>Únicos</th>
                <th>Descripción</th>
                <th>Notas</th>
              </tr>
            </thead>
            <tbody>
              {source.dataset_audit.variable_dictionary.map((item) => (
                <tr key={item.name}>
                  <td>{item.name}</td>
                  <td>{item.logical_type}</td>
                  <td>{item.observed_type}</td>
                  <td>{item.analytical_role}</td>
                  <td>{formatPct(item.missing_pct)}</td>
                  <td>{item.unique_values ?? '-'}</td>
                  <td>{item.business_description}</td>
                  <td>{item.notes ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <DataTable columns={source.columns} rows={source.sample_rows} />

      <div className="panel">
        <h3>Hallazgos de calidad</h3>
        {source.data_quality.findings.length ? (
          <div className="quality-grid">
            {source.data_quality.findings.map((finding, index) => (
              <article key={`${finding.kind}-${index}`} className={`quality-card ${finding.severity}`}>
                <div className="quality-card-top">
                  <strong>{finding.kind}</strong>
                  <span>{finding.affected_columns.join(', ') || 'dataset'}</span>
                </div>
                <p><strong>Hallazgo:</strong> {finding.hallazgo}</p>
                <p><strong>Riesgo:</strong> {finding.riesgo}</p>
                <p><strong>Decisión:</strong> {finding.decision}</p>
              </article>
            ))}
          </div>
        ) : (
          <p className="muted">No se detectaron hallazgos críticos adicionales.</p>
        )}
      </div>

      <div className="panel">
        <h3>Advertencias metodológicas</h3>
        <WarningChips warnings={source.warnings} />
      </div>

      <div className="panel">
        <h3>Análisis univariado numérico</h3>
        {source.univariate_numeric.status === 'available' ? (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variable</th>
                    <th>Media</th>
                    <th>Mediana</th>
                    <th>Desv. est.</th>
                    <th>IQR</th>
                    <th>Asimetría</th>
                    <th>Curtosis</th>
                    <th>Coef. var.</th>
                  </tr>
                </thead>
                <tbody>
                  {source.univariate_numeric.variables.map((item) => (
                    <tr key={item.column}>
                      <td>{item.column}</td>
                      <td>{formatValue(item.stats.mean)}</td>
                      <td>{formatValue(item.stats.median)}</td>
                      <td>{formatValue(item.stats.std)}</td>
                      <td>{formatValue(item.stats.iqr)}</td>
                      <td>{formatValue(item.stats.skewness)}</td>
                      <td>{formatValue(item.stats.kurtosis)}</td>
                      <td>{formatValue(item.stats.coefficient_variation)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="grid-2">
              {source.univariate_numeric.variables.map((item) => {
                const chartId = `hist-${sourceKey}-${item.column}`;
                const data = item.histogram_bins.map((bin) => ({
                  range: `${formatValue(bin.left)} a ${formatValue(bin.right)}`,
                  count: bin.count
                }));

                const CustomTooltip = ({ active, payload, label }: any) => {
                  if (active && payload && payload.length) {
                    return (
                      <div style={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', padding: '10px', color: 'var(--text)' }}>
                        <strong style={{ display: 'block', marginBottom: '4px' }}>Rango: {label}</strong>
                        <span style={{ color: '#2e5c73' }}>Frecuencia absoluta: {payload[0].value.toLocaleString()}</span>
                      </div>
                    );
                  }
                  return null;
                };

                return (
                  <div key={chartId} className="panel table-panel">
                    <div className="table-header">
                      <div>
                        <h4 style={{ margin: 0, fontSize: '1rem' }}>Histograma de {item.column}</h4>
                        <span className="muted" style={{ display: 'block', fontSize: '0.8rem' }}>
                          Distribución de frecuencias para la variable continua.
                        </span>
                      </div>
                      <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.2rem 0.5rem' }} onClick={() => downloadElementAsPng(chartId, `${chartId}.png`)}>
                        ⬇️ PNG
                      </button>
                    </div>
                    <div id={chartId} style={{ height: 250, marginTop: '1rem', position: 'relative' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                          <XAxis dataKey="range" tick={{ fill: 'var(--muted)', fontSize: 10 }} axisLine={false} tickLine={false} angle={-30} textAnchor="end" height={50} />
                          <YAxis tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                          <RechartsTooltip cursor={{ fill: 'var(--bg-elevated)', opacity: 0.5 }} content={<CustomTooltip />} />
                          <Bar dataKey="count" fill="#2e5c73" radius={[2, 2, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
            <BoxplotPanel
              title={`Distribuciones numéricas ${source.title}`}
              emptyMessage="No hay suficientes valores numéricos para boxplots."
              series={numericBoxplotSeries}
            />
          </>
        ) : (
          <p className="muted">{source.univariate_numeric.message}</p>
        )}
      </div>

      <div className="panel">
        <h3>Análisis univariado categórico</h3>
        {source.univariate_categorical.status === 'available' ? (
          <div className="grid-2">
            {source.univariate_categorical.variables.map((item) => {
              const chartId = `cat-${sourceKey}-${item.column}`;
              const data = item.top_categories.map((row) => ({
                name: row.value,
                pct: row.pct
              }));
              
              const CustomTooltip = ({ active, payload, label }: any) => {
                if (active && payload && payload.length) {
                  return (
                    <div style={{ backgroundColor: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)', padding: '10px', color: 'var(--text)' }}>
                      <strong style={{ display: 'block', marginBottom: '4px' }}>{label}</strong>
                      <span style={{ color: '#b06a31' }}>Frecuencia relativa: {payload[0].value.toFixed(2)}%</span>
                    </div>
                  );
                }
                return null;
              };

              return (
                <div key={chartId} className="panel table-panel">
                  <div className="table-header">
                    <div>
                      <h4 style={{ margin: 0, fontSize: '1rem' }}>Distribución de {item.column}</h4>
                      {item.other_count > 0 && (
                        <span className="muted" style={{ display: 'block', fontSize: '0.8rem' }}>
                          Se agrupan {item.other_count} registros menos frecuentes en "otras".
                        </span>
                      )}
                    </div>
                    <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.2rem 0.5rem' }} onClick={() => downloadElementAsPng(chartId, `${chartId}.png`)}>
                      ⬇️ PNG
                    </button>
                  </div>
                  <div id={chartId} style={{ height: 250, marginTop: '1rem', position: 'relative' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" opacity={0.5} />
                        <XAxis dataKey="name" tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} angle={-45} textAnchor="end" height={60} />
                        <YAxis unit="%" tick={{ fill: 'var(--muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                        <RechartsTooltip cursor={{ fill: 'var(--bg-elevated)', opacity: 0.5 }} content={<CustomTooltip />} />
                        <Bar dataKey="pct" fill="#b06a31" radius={[4, 4, 0, 0]} maxBarSize={50} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">{source.univariate_categorical.message}</p>
        )}
      </div>

      <div className="panel">
        <h3>Análisis bivariado</h3>
        {source.bivariate_numeric_numeric.status === 'available' ? (
          <div className="stack">
            {source.bivariate_numeric_numeric.target_rankings.length ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Pearson</th>
                      <th>Spearman</th>
                    </tr>
                  </thead>
                  <tbody>
                    {source.bivariate_numeric_numeric.target_rankings.map((item) => (
                      <tr key={item.feature}>
                        <td>{item.feature}</td>
                        <td>{formatValue(item.pearson)}</td>
                        <td>{formatValue(item.spearman)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            <div className="grid-2">
              <CorrelationHeatmap
                title={`Correlación Pearson - ${source.title}`}
                fileBaseName={`${sourceKey}-pearson`}
                labels={source.bivariate_numeric_numeric.labels}
                matrix={source.bivariate_numeric_numeric.pearson_matrix}
              />
              <CorrelationHeatmap
                title={`Correlación Spearman - ${source.title}`}
                fileBaseName={`${sourceKey}-spearman`}
                labels={source.bivariate_numeric_numeric.labels}
                matrix={source.bivariate_numeric_numeric.spearman_matrix}
              />
            </div>
          </div>
        ) : (
          <p className="muted">{source.bivariate_numeric_numeric.message}</p>
        )}

        {source.bivariate_categorical_numeric.status === 'available' ? (
          <div className="stack">
            <div className="panel table-panel nested-panel">
              <h4>Comparación categórica vs numérica</h4>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Prueba</th>
                      <th>Valor p</th>
                      <th>Efecto</th>
                      <th>Grupos</th>
                    </tr>
                  </thead>
                  <tbody>
                    {source.bivariate_categorical_numeric.rows.map((row) => (
                      <tr key={row.feature}>
                        <td>{row.feature}</td>
                        <td>{row.test_used}</td>
                        <td>{formatValue(row.p_value)}</td>
                        <td>{formatValue(row.effect_size)}</td>
                        <td>{row.n_groups ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <BoxplotPanel
              title={`Distribuciones por grupo - ${source.title}`}
              emptyMessage="No hay grupos suficientes para boxplots por categoría."
              series={source.bivariate_categorical_numeric.boxplot_data}
            />
          </div>
        ) : (
          <p className="muted">{source.bivariate_categorical_numeric.message}</p>
        )}

        {source.bivariate_categorical_categorical.status === 'available' ? (
          <div className="stack">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Variable X</th>
                    <th>Variable Y</th>
                    <th>Chi²</th>
                    <th>Valor p</th>
                    <th>V de Cramér</th>
                    <th>N</th>
                  </tr>
                </thead>
                <tbody>
                  {source.bivariate_categorical_categorical.rows.map((row) => (
                    <tr key={`${row.feature_x}-${row.feature_y}`}>
                      <td>{row.feature_x}</td>
                      <td>{row.feature_y}</td>
                      <td>{formatValue(row.chi2)}</td>
                      <td>{formatValue(row.p_value)}</td>
                      <td>{formatValue(row.cramers_v)}</td>
                      <td>{row.sample_size}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {source.bivariate_categorical_categorical.contingency_preview ? (
              <div className="panel table-panel nested-panel">
                <h4>
                  Tabla de contingencia: {source.bivariate_categorical_categorical.contingency_preview.row_label} vs{' '}
                  {source.bivariate_categorical_categorical.contingency_preview.column_label}
                </h4>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>{source.bivariate_categorical_categorical.contingency_preview.row_label}</th>
                        {source.bivariate_categorical_categorical.contingency_preview.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {source.bivariate_categorical_categorical.contingency_preview.rows.map((row, index) => (
                        <tr key={`preview-${index}`}>
                          {Object.values(row).map((value, valueIndex) => (
                            <td key={`value-${valueIndex}`}>{String(value)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          <p className="muted">{source.bivariate_categorical_categorical.message}</p>
        )}

        <div className="academic-temporal-note">
          <strong>Diagnóstico temporal:</strong> {source.temporal_diagnostics.message}
        </div>
      </div>
    </section>
  );
}

export default function Week1AcademicEdaPanel({ payload, clustering, clusteringLoading = false }: Props) {
  const inSource = payload.sources.in;
  const outSource = payload.sources.out;

  return (
    <section className="stack">
      <nav className="academic-nav panel">
        <a href="#problem">Problema</a>
        <a href="#in">IN</a>
        <a href="#out">OUT</a>
        <a href="#comparison">Comparativo</a>
        <a href="#imputation">Imputación</a>
        <a href="#outliers">Outliers</a>
        <a href="#clustering">OPTICS</a>
        <a href="#insights">Hallazgos</a>
        <a href="#report">Reporte</a>
      </nav>

      <section id="problem" className="stack">
        <div className="academic-cover panel">
          <div>
            <span className="eyebrow">Semana 1</span>
            <h2>EDA académico dual por fuentes</h2>
            <p>{payload.problem_definition.domain_context}</p>
          </div>
          <div className="academic-cover-side">
            <div>
              <strong>Objetivo</strong>
              <p>{payload.problem_definition.objective}</p>
            </div>
            <div>
              <strong>Unidad de observación</strong>
              <p>{payload.problem_definition.unit_of_observation}</p>
            </div>
            <div>
              <strong>Variable objetivo</strong>
              <p>{payload.problem_definition.target_variable ?? 'No aplica'}</p>
            </div>
          </div>
        </div>

        <div className="panel">
          <h3>Hipótesis iniciales</h3>
          <ul className="plain-list">
            {payload.problem_definition.initial_hypotheses.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <SourceSection sourceKey="in" source={inSource} />
      <SourceSection sourceKey="out" source={outSource} />

      <section id="comparison" className="stack">
        <div className="academic-section-head panel">
          <div>
            <span className="eyebrow">Comparativo</span>
            <h2>IN vs OUT</h2>
            <p className="muted">Comparación visual y narrativa solo sobre variables compartidas.</p>
          </div>
        </div>
        <div className="panel">
          <h3>Notas comparativas</h3>
          <ul className="plain-list">
            {payload.comparison.notes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div className="grid-2">
          {payload.comparison.categorical_comparisons.map((item) => (
            <PlotlyFigureCard
              key={`comparison-${item.column}`}
              title={`Distribución comparada de ${item.column}`}
              fileBaseName={`comparacion-${item.column}`}
              note={item.note ?? undefined}
              data={[
                {
                  type: 'bar',
                  name: 'IN',
                  x: item.categories.map((row) => row.category),
                  y: item.categories.map((row) => row.in_pct),
                  marker: { color: '#294e63' },
                },
                {
                  type: 'bar',
                  name: 'OUT',
                  x: item.categories.map((row) => row.category),
                  y: item.categories.map((row) => row.out_pct),
                  marker: { color: '#b06a31' },
                },
              ]}
              layout={{
                barmode: 'group',
                xaxis: { title: item.column, automargin: true },
                yaxis: { title: 'Porcentaje' },
                legend: { orientation: 'h' },
              }}
            />
          ))}
        </div>
      </section>

      <section id="imputation" className="stack">
        <div className="academic-section-head panel">
          <div>
            <span className="eyebrow">Imputación</span>
            <h2>Capa limpia derivada</h2>
            <p className="muted">
              {payload.imputation.imputation_applied
                ? 'Se detectaron NA y se construyó una vista imputada derivada.'
                : 'Los seeds actuales no requieren imputación, pero la estrategia queda documentada.'}
            </p>
          </div>
        </div>
        <div className="grid-2">
          {(['in', 'out'] as const).map((sourceKey) => (
            <div key={sourceKey} className="panel table-panel">
              <h3>{sourceKey.toUpperCase()}</h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>NA raw</th>
                      <th>% raw</th>
                      <th>Estrategia</th>
                      <th>Imputados</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(payload.imputation.strategy_by_column[sourceKey]).map((column) => (
                      <tr key={`${sourceKey}-${column}`}>
                        <td>{column}</td>
                        <td>{payload.imputation.raw_missing_summary[sourceKey][column]?.count ?? 0}</td>
                        <td>{formatPct(payload.imputation.raw_missing_summary[sourceKey][column]?.pct)}</td>
                        <td>{payload.imputation.strategy_by_column[sourceKey][column]}</td>
                        <td>{payload.imputation.imputed_counts[sourceKey][column]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted">Dataset analítico: {payload.imputation.analysis_dataset_paths[sourceKey]}</p>
            </div>
          ))}
        </div>
        <div className="panel">
          <h3>Notas de imputación</h3>
          <ul className="plain-list">
            {payload.imputation.notes.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <section id="outliers" className="stack">
        <div className="academic-section-head panel">
          <div>
            <span className="eyebrow">Outliers</span>
            <h2>Diagnóstico sin filtrado</h2>
            <p className="muted">La política activa es {payload.outliers.policy.replace(/_/g, ' ')}.</p>
          </div>
        </div>
        <div className="grid-2">
          {(['in', 'out'] as const).map((sourceKey) => {
            const outlierSource = payload.outliers.sources[sourceKey];
            return (
              <div key={`outlier-${sourceKey}`} className="panel table-panel">
                <h3>{sourceKey.toUpperCase()}</h3>
                <p className="muted">{outlierSource.interpretation}</p>
                <p className="muted">
                  Filas marcadas: {outlierSource.flagged_counts} ({formatPct(outlierSource.flagged_ratio)})
                </p>
                {outlierSource.status === 'available' ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Variable</th>
                          <th>Marcados</th>
                          <th>%</th>
                          <th>Límite inferior</th>
                          <th>Límite superior</th>
                          <th>Máx robust z</th>
                        </tr>
                      </thead>
                      <tbody>
                        {outlierSource.columns.map((row) => (
                          <tr key={`${sourceKey}-${row.column}`}>
                            <td>{row.column}</td>
                            <td>{row.flagged_count}</td>
                            <td>{formatPct(row.flagged_ratio)}</td>
                            <td>{formatValue(row.lower_bound)}</td>
                            <td>{formatValue(row.upper_bound)}</td>
                            <td>{formatValue(row.max_abs_robust_z)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="muted">No aplica para esta fuente.</p>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <section id="clustering" className="stack">
        <div className="academic-section-head panel">
          <div>
            <span className="eyebrow">Clustering</span>
            <h2>OPTICS por fuente</h2>
            <p className="muted">OPTICS se ejecuta sobre el espacio analítico y la visualización oficial usa UMAP 2D.</p>
          </div>
        </div>
        {clusteringLoading ? <p className="muted">Calculando clustering de la semana 1...</p> : null}
        {clustering ? (
          <div className="stack">
            {(['in', 'out'] as const).map((sourceKey) => {
              const optics = clustering.sources[sourceKey];
              return (
                <div key={`optics-${sourceKey}`} className="stack">
                  <div className="panel">
                    <h3>OPTICS {sourceKey.toUpperCase()}</h3>
                    <p className="muted">{optics.interpretation}</p>
                    <div className="academic-metrics">
                      <article className="academic-metric-card">
                        <span>Clusters</span>
                        <strong>{optics.cluster_count}</strong>
                      </article>
                      <article className="academic-metric-card">
                        <span>Ruido</span>
                        <strong>{formatPct(optics.noise_ratio)}</strong>
                      </article>
                      <article className="academic-metric-card">
                        <span>Variables</span>
                        <strong>{optics.feature_columns.length}</strong>
                      </article>
                      <article className="academic-metric-card">
                        <span>Embedding</span>
                        <strong>{optics.embedding_method.toUpperCase()}</strong>
                      </article>
                      <article className="academic-metric-card">
                        <span>Trustworthiness</span>
                        <strong>{formatValue(optics.embedding_quality.trustworthiness)}</strong>
                      </article>
                      <article className="academic-metric-card">
                        <span>Overlap raw</span>
                        <strong>{formatPct(optics.overlap_stats.overlap_pct)}</strong>
                      </article>
                    </div>
                    <p className="muted">Variables usadas: {optics.feature_columns.join(', ') || 'n/a'}</p>
                    <p className="muted">
                      Parámetros OPTICS seleccionados: min_samples={optics.selected_optics_parameters['min_samples']}, min_cluster_size=
                      {optics.selected_optics_parameters['min_cluster_size']}, xi={optics.selected_optics_parameters['xi']}
                    </p>
                    <p className="muted">
                      Embedding visual: n_neighbors={optics.embedding_parameters['n_neighbors']}, min_dist=
                      {optics.embedding_parameters['min_dist']}, jitter aplicado={optics.overlap_stats.jitter_applied ? 'sí' : 'no'}
                    </p>
                    <p className="muted">Artefacto: {optics.artifacts.json_path}</p>
                    <WarningChips warnings={optics.warnings} />
                  </div>

                  {optics.status === 'available' ? (() => {
                    return (
                      <>
                        <OpticsRechartsCards sourceKey={sourceKey} optics={optics} />

                        <div className="panel table-panel">
                          <div className="table-header">
                            <div>
                              <h4>Resumen de clusters</h4>
                            </div>
                            <button type="button" className="outline-btn" style={{ fontSize: '0.8rem', padding: '0.3rem 0.6rem' }} onClick={() => downloadElementAsPng(`cluster-table-${sourceKey}`, `resumen_clusters_${sourceKey}.png`)}>
                              ⬇️ Descargar PNG
                            </button>
                          </div>
                          <div className="table-wrap" id={`cluster-table-${sourceKey}`} style={{ backgroundColor: 'var(--bg)', padding: '1rem' }}>
                            <table>
                              <thead>
                                <tr>
                                  <th>Color</th>
                                  <th>Cluster</th>
                                  <th>Descripción</th>
                                  <th>Tamaño</th>
                                  <th>%</th>
                                  <th>Categorías dominantes</th>
                                </tr>
                              </thead>
                              <tbody>
                                {optics.cluster_summary.map((row) => (
                                  <tr key={`${sourceKey}-${row.cluster_label}`}>
                                    <td>
                                      <span
                                        className="cluster-swatch"
                                        style={{ backgroundColor: clusterColor(row.cluster_id, row.cluster_label === 'ruido') }}
                                      />
                                    </td>
                                    <td>{row.cluster_label}</td>
                                    <td>{row.description}</td>
                                    <td>{row.size}</td>
                                    <td>{formatPct(row.pct)}</td>
                                    <td>
                                      {Object.entries(row.top_categories)
                                        .map(([key, value]) => `${key}: ${value}`)
                                        .join(' | ') || '-'}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        <div className="panel table-panel">
                          <h4>Selección de hiperparámetros</h4>
                          <div className="table-wrap">
                            <table>
                              <thead>
                                <tr>
                                  <th>Estado</th>
                                  <th>min_samples</th>
                                  <th>min_cluster_size</th>
                                  <th>xi</th>
                                  <th>Clusters</th>
                                  <th>Ruido</th>
                                  <th>Cobertura no ruido</th>
                                  <th>Silhouette</th>
                                  <th>Rechazo</th>
                                </tr>
                              </thead>
                              <tbody>
                                {optics.candidate_search_summary.map((candidate, index) => (
                                  <tr key={`${sourceKey}-candidate-${index}`}>
                                    <td>{candidate.selected ? 'seleccionado' : candidate.rejected ? 'descartado' : 'válido'}</td>
                                    <td>{candidate.min_samples}</td>
                                    <td>{candidate.min_cluster_size}</td>
                                    <td>{candidate.xi}</td>
                                    <td>{candidate.cluster_count}</td>
                                    <td>{formatPct(candidate.noise_ratio)}</td>
                                    <td>{formatPct(candidate.non_noise_coverage)}</td>
                                    <td>{formatValue(candidate.silhouette_non_noise)}</td>
                                    <td>{candidate.rejection_reason ?? '-'}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      </>
                    );
                  })() : null}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">No se pudo cargar el detalle de clustering.</p>
        )}
      </section>

      <section id="insights" className="stack">
        <div className="academic-section-head panel">
          <div>
            <span className="eyebrow">Hallazgos</span>
            <h2>Conclusiones automáticas</h2>
          </div>
        </div>
        <div className="insight-grid">
          {payload.insights.map((insight) => (
            <article key={insight.title} className="insight-card">
              <h3>{insight.title}</h3>
              <p><strong>Evidencia:</strong> {insight.evidence}</p>
              <p><strong>Implicancia:</strong> {insight.implication}</p>
              <p><strong>Siguiente paso:</strong> {insight.next_step}</p>
            </article>
          ))}
        </div>
        <div className="panel">
          <h3>Advertencias globales</h3>
          <WarningChips warnings={payload.warnings} />
        </div>
      </section>
    </section>
  );
}
