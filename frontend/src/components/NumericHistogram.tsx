import { useRef } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { exportSvgInContainerAsPng } from '../utils/chartExport';

type Bin = {
  left: number;
  right: number;
  count: number;
};

type Props = {
  title: string;
  bins: Bin[];
};

export default function NumericHistogram({ title, bins }: Props) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  const chartData = bins.map((bin) => ({
    range: `${bin.left.toFixed(1)}-${bin.right.toFixed(1)}`,
    count: bin.count,
  }));

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `histogram-${title}`);
    } catch (error) {
      console.error('chart export failed', error);
    }
  };

  return (
    <div className="panel chart-panel">
      <div className="chart-header">
        <h4>{title}</h4>
        <button type="button" className="chart-export-btn" onClick={() => void handleExport()}>
          Descargar PNG HD
        </button>
      </div>
      <div ref={chartRef} className="chart-export-area">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="range" tick={{ fontSize: 10 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#d9822b" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
