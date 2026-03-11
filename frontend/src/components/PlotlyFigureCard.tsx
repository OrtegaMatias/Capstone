import { useRef } from 'react';
import Plot from 'react-plotly.js';

import { exportSvgInContainerAsPng } from '../utils/chartExport';

type Props = {
  title: string;
  fileBaseName: string;
  data: Array<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  note?: string;
};

export default function PlotlyFigureCard({ title, fileBaseName, data, layout, note }: Props) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, fileBaseName);
    } catch (error) {
      console.error('chart export failed', error);
    }
  };

  return (
    <div className="panel chart-panel">
      <div className="chart-header">
        <div>
          <h3>{title}</h3>
          {note ? <p className="muted chart-note">{note}</p> : null}
        </div>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()}>
          Descargar PNG HD
        </button>
      </div>
      <div ref={chartRef} className="chart-export-area">
        <Plot
          data={data}
          layout={{
            autosize: true,
            height: 340,
            margin: { t: 28, r: 24, b: 56, l: 56 },
            paper_bgcolor: '#f6f1e8',
            plot_bgcolor: '#fffdfa',
            font: { family: "Georgia, 'Times New Roman', serif", color: '#1d2b36', size: 13 },
            ...layout,
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
