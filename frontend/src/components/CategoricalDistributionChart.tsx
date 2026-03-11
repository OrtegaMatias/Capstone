import { useRef } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { exportSvgInContainerAsPng } from '../utils/chartExport';

type Item = {
  value: string;
  count: number;
  pct: number;
};

type Props = {
  title: string;
  data: Item[];
};

export default function CategoricalDistributionChart({ title, data }: Props) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  const handleExport = async () => {
    if (!chartRef.current) return;
    try {
      await exportSvgInContainerAsPng(chartRef.current, `categorical-${title}`);
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
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="4 4" />
            <XAxis dataKey="value" tick={{ fontSize: 12 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#0e9f9c" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
