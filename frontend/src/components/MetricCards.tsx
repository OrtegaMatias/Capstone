type Props = {
  metrics: Record<string, string | number | null | undefined>;
};

export default function MetricCards({ metrics }: Props) {
  return (
    <div className="cards-grid">
      {Object.entries(metrics).map(([key, value]) => (
        <article key={key} className="metric-card">
          <p>{key}</p>
          <strong>
            {typeof value === 'number'
              ? value.toLocaleString(undefined, { maximumFractionDigits: 4 })
              : value ?? '-'}
          </strong>
        </article>
      ))}
    </div>
  );
}
