import { useMemo, useState } from 'react';

import type { AnovaRow } from '../api/types';

type Props = {
  rows: AnovaRow[];
};

type SortKey = 'p_value' | 'effect_size';

export default function AnovaTable({ rows }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>('p_value');

  const sortedRows = useMemo(() => {
    return [...rows].sort((a, b) => {
      const av = a[sortKey] ?? Number.POSITIVE_INFINITY;
      const bv = b[sortKey] ?? Number.POSITIVE_INFINITY;
      return av - bv;
    });
  }, [rows, sortKey]);

  return (
    <div className="panel table-panel">
      <div className="table-header">
        <h3>ANOVA / OLS / Kruskal</h3>
        <div className="segmented">
          <button className={sortKey === 'p_value' ? 'active' : ''} onClick={() => setSortKey('p_value')}>
            Sort p-value
          </button>
          <button className={sortKey === 'effect_size' ? 'active' : ''} onClick={() => setSortKey('effect_size')}>
            Sort effect
          </button>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Type</th>
              <th>Test</th>
              <th>Statistic</th>
              <th>p-value</th>
              <th>Effect</th>
              <th>Groups</th>
              <th>Kruskal p</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.feature}>
                <td>{row.feature}</td>
                <td>{row.feature_type}</td>
                <td>{row.test_used}</td>
                <td>{row.statistic?.toFixed(4) ?? '-'}</td>
                <td>{row.p_value?.toExponential(3) ?? '-'}</td>
                <td>{row.effect_size?.toFixed(4) ?? '-'}</td>
                <td>{row.n_groups ?? '-'}</td>
                <td>{row.kruskal_p_value?.toExponential(3) ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
