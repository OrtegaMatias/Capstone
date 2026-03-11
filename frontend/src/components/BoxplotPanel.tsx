import { useRef } from 'react';
import Plot from 'react-plotly.js';

import type { BoxplotSeries } from '../api/types';
import { exportSvgInContainerAsPng } from '../utils/chartExport';

type Props = {
  series: BoxplotSeries[];
  title?: string;
  emptyMessage?: string;
};

export default function BoxplotPanel({
  series,
  title = 'Target vs Top Features',
  emptyMessage = 'No hay datos categóricos suficientes para boxplots en top variables.',
}: Props) {
  const plotRefs = useRef<Record<string, HTMLDivElement | null>>({});

  if (series.length === 0) {
    return (
      <div className="panel">
        <h3>{title}</h3>
        <p className="muted">{emptyMessage}</p>
      </div>
    );
  }

  const handleExport = async (feature: string) => {
    const container = plotRefs.current[feature];
    if (!container) return;
    try {
      await exportSvgInContainerAsPng(container, `boxplot-${feature}`);
    } catch (error) {
      console.error('chart export failed', error);
    }
  };

  return (
    <div className="panel">
      <h3>{title}</h3>
      {series.map((featureSeries) => (
        <div key={featureSeries.feature} className="plot-wrap">
          <div className="chart-header">
            <h4>{featureSeries.feature}</h4>
            <button type="button" className="chart-export-btn" onClick={() => void handleExport(featureSeries.feature)}>
              Descargar PNG HD
            </button>
          </div>
          <div
            className="chart-export-area"
            ref={(node) => {
              plotRefs.current[featureSeries.feature] = node;
            }}
          >
            <Plot
              data={featureSeries.groups.map((group) => ({
                y: group.values,
                type: 'box',
                name: group.group,
                boxpoints: 'outliers',
              }))}
              layout={{
                autosize: true,
                height: 340,
                margin: { t: 20, r: 20, b: 50, l: 50 },
                paper_bgcolor: '#f8f6f3',
                plot_bgcolor: '#ffffff',
              }}
              style={{ width: '100%' }}
              config={{ displayModeBar: false }}
              useResizeHandler
            />
          </div>
        </div>
      ))}
    </div>
  );
}
