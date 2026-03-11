import type { PivotAggFunc, PivotQueryResponse } from '../api/types';
import type { CSSProperties } from 'react';

type Props = {
  payload: PivotQueryResponse;
};

function getAggLabel(agg: PivotAggFunc): string {
  if (agg === 'count') return 'Conteo';
  if (agg === 'sum') return 'Suma';
  if (agg === 'mean') return 'Promedio';
  if (agg === 'median') return 'Mediana';
  if (agg === 'rate_gt_7') return '% > 7 dias';
  if (agg === 'rate_gt_14') return '% > 14 dias';
  return '% > 30 dias';
}

function formatValue(value: number | null, agg: PivotAggFunc): string {
  if (value === null || Number.isNaN(value)) return '-';
  if (agg === 'count') return `${Math.round(value)}`;
  if (agg.startsWith('rate_gt_')) return `${value.toFixed(2)}%`;
  return `${value.toFixed(2)}`;
}

function getHeatAlpha(value: number | null, maxAbs: number): number {
  if (value === null || maxAbs <= 0) return 0;
  return Math.max(0.06, Math.min(0.42, 0.08 + (Math.abs(value) / maxAbs) * 0.34));
}

function toRgba(alpha: number): string {
  return `rgba(19, 103, 200, ${alpha.toFixed(3)})`;
}

function downloadDataUrl(dataUrl: string, filename: string) {
  const anchor = document.createElement('a');
  anchor.href = dataUrl;
  anchor.download = filename;
  anchor.click();
}

function exportPivotPng(payload: PivotQueryResponse, maxAbs: number): void {
  const title = `${payload.agg_func}(${payload.value_field}) | ${payload.row_dim} x ${payload.col_dim}`;
  const legendText = 'Leyenda: cada celda muestra la metrica agregada; color mas intenso = mayor magnitud.';
  const exportedText = `Exportado: ${new Date().toLocaleString()}`;
  const columns = [payload.row_dim, ...payload.matrix.columns, 'Total general'];
  const rows = payload.matrix.rows;
  const scale = 3;

  const measureCanvas = document.createElement('canvas');
  const mctx = measureCanvas.getContext('2d');
  if (!mctx) return;

  const fontHeader = '700 13px Manrope, sans-serif';
  const fontValue = '600 12px Manrope, sans-serif';
  const fontTitle = '700 15px Manrope, sans-serif';
  const fontMeta = '400 11px Manrope, sans-serif';
  const cellPaddingX = 10;
  const outerPad = 18;
  const infoBlockHeight = 58;
  const headerHeight = 32;
  const rowHeight = 30;

  const colWidths = columns.map((header, index) => {
    mctx.font = fontHeader;
    let width = mctx.measureText(header).width + cellPaddingX * 2;

    if (index === 0) {
      for (const row of rows) {
        width = Math.max(width, mctx.measureText(row.row_key).width + cellPaddingX * 2);
      }
      width = Math.max(width, mctx.measureText('Total general').width + cellPaddingX * 2);
    } else if (index === columns.length - 1) {
      for (const row of rows) {
        mctx.font = fontValue;
        width = Math.max(width, mctx.measureText(formatValue(row.row_total.value, payload.agg_func)).width + cellPaddingX * 2);
      }
      mctx.font = fontValue;
      width = Math.max(width, mctx.measureText(formatValue(payload.matrix.grand_total.value, payload.agg_func)).width + cellPaddingX * 2);
    } else {
      const colKey = columns[index];
      for (const row of rows) {
        const cell = row.cells.find((c) => c.col_key === colKey);
        if (!cell) continue;
        mctx.font = fontValue;
        width = Math.max(width, mctx.measureText(formatValue(cell.value, payload.agg_func)).width + cellPaddingX * 2);
      }
      const colTotal = payload.matrix.column_totals.find((c) => c.col_key === colKey);
      if (colTotal) {
        mctx.font = fontValue;
        width = Math.max(width, mctx.measureText(formatValue(colTotal.value, payload.agg_func)).width + cellPaddingX * 2);
      }
    }

    return Math.max(72, Math.ceil(width));
  });

  const tableWidth = colWidths.reduce((a, b) => a + b, 0);
  const tableHeight = headerHeight + (rows.length + 1) * rowHeight;

  mctx.font = fontTitle;
  const titleWidth = mctx.measureText(title).width;
  mctx.font = fontMeta;
  const legendWidth = mctx.measureText(legendText).width;
  const exportedWidth = mctx.measureText(exportedText).width;
  const infoWidth = Math.ceil(Math.max(titleWidth, legendWidth, exportedWidth));

  const logicalWidth = Math.ceil(Math.max(tableWidth + outerPad * 2, infoWidth + outerPad * 2));
  const yStart = outerPad + infoBlockHeight;
  const logicalHeight = Math.ceil(yStart + tableHeight + outerPad);
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(logicalWidth * scale);
  canvas.height = Math.round(logicalHeight * scale);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(scale, scale);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, logicalWidth, logicalHeight);

  ctx.fillStyle = '#173043';
  ctx.font = fontTitle;
  ctx.fillText(title, outerPad, outerPad + 10);
  ctx.font = fontMeta;
  ctx.fillStyle = '#5d6b76';
  ctx.fillText(legendText, outerPad, outerPad + 28);
  ctx.fillText(exportedText, outerPad, outerPad + 44);

  let x = outerPad;

  for (let i = 0; i < columns.length; i += 1) {
    const w = colWidths[i];
    ctx.fillStyle = '#f6f8f9';
    ctx.fillRect(x, yStart, w, headerHeight);
    ctx.strokeStyle = '#d3d8db';
    ctx.strokeRect(x, yStart, w, headerHeight);

    ctx.font = fontHeader;
    ctx.fillStyle = '#173043';
    ctx.fillText(columns[i], x + cellPaddingX, yStart + 21);
    x += w;
  }

  for (let r = 0; r < rows.length; r += 1) {
    const row = rows[r];
    let cx = outerPad;
    const y = yStart + headerHeight + r * rowHeight;

    for (let c = 0; c < columns.length; c += 1) {
      const w = colWidths[c];
      const isTotalCol = c === columns.length - 1;

      let fill = '#ffffff';
      let valueText = '';

      if (c === 0) {
        fill = '#fbfcfd';
        valueText = row.row_key;
      } else if (isTotalCol) {
        fill = '#eef3f6';
        valueText = formatValue(row.row_total.value, payload.agg_func);
      } else {
        const colKey = columns[c];
        const cell = row.cells.find((cellItem) => cellItem.col_key === colKey);
        if (cell) {
          const alpha = getHeatAlpha(cell.value, maxAbs);
          fill = alpha > 0 ? toRgba(alpha) : '#ffffff';
          valueText = formatValue(cell.value, payload.agg_func);
        } else {
          valueText = '-';
        }
      }

      ctx.fillStyle = fill;
      ctx.fillRect(cx, y, w, rowHeight);
      ctx.strokeStyle = '#d3d8db';
      ctx.strokeRect(cx, y, w, rowHeight);

      ctx.font = c === 0 ? fontHeader : fontValue;
      ctx.fillStyle = '#173043';
      ctx.fillText(valueText, cx + cellPaddingX, y + 19);

      cx += w;
    }
  }

  // Totals row
  const totalsY = yStart + headerHeight + rows.length * rowHeight;
  let tx = outerPad;
  for (let c = 0; c < columns.length; c += 1) {
    const w = colWidths[c];
    ctx.fillStyle = '#dce8ef';
    ctx.fillRect(tx, totalsY, w, rowHeight);
    ctx.strokeStyle = '#d3d8db';
    ctx.strokeRect(tx, totalsY, w, rowHeight);

    if (c === 0) {
      ctx.font = fontHeader;
      ctx.fillStyle = '#173043';
      ctx.fillText('Total general', tx + cellPaddingX, totalsY + 19);
    } else if (c === columns.length - 1) {
      ctx.font = fontValue;
      ctx.fillStyle = '#173043';
      ctx.fillText(formatValue(payload.matrix.grand_total.value, payload.agg_func), tx + cellPaddingX, totalsY + 19);
    } else {
      const colKey = columns[c];
      const total = payload.matrix.column_totals.find((item) => item.col_key === colKey);
      if (total) {
        ctx.font = fontValue;
        ctx.fillStyle = '#173043';
        ctx.fillText(formatValue(total.value, payload.agg_func), tx + cellPaddingX, totalsY + 19);
      }
    }

    tx += w;
  }

  const filename = `pivot-${payload.source}-${payload.row_dim}-x-${payload.col_dim}.png`
    .replace(/\s+/g, '_')
    .toLowerCase();
  downloadDataUrl(canvas.toDataURL('image/png'), filename);
}

export default function PivotMatrixTable({ payload }: Props) {
  const values = payload.matrix.rows.flatMap((row) => row.cells.map((cell) => cell.value).filter((v): v is number => v !== null));
  const maxAbs = values.length > 0 ? Math.max(...values.map((v) => Math.abs(v))) : 0;

  return (
    <div className="panel table-panel pivot-table-panel">
      <div className="table-header">
        <h4>
          {payload.agg_func}({payload.value_field}) | {payload.row_dim} x {payload.col_dim}
        </h4>
        <button type="button" className="pivot-export-btn" onClick={() => exportPivotPng(payload, maxAbs)}>
          Exportar PNG HD
        </button>
      </div>
      <div className="pivot-legend">
        <span className="pivot-legend-item">
          <strong>Metrica:</strong> {getAggLabel(payload.agg_func)} de <strong>{payload.value_field}</strong>
        </span>
        <span className="pivot-legend-item">
          <strong>Lectura:</strong> color mas intenso = valor relativamente mayor en esta tabla
        </span>
        <span className="pivot-legend-item">
          <strong>Ejes:</strong> filas = {payload.row_dim}, columnas = {payload.col_dim}
        </span>
      </div>

      <div className="table-wrap">
        <table className="pivot-compact-table">
          <thead>
            <tr>
              <th className="pivot-row-header">{payload.row_dim}</th>
              {payload.matrix.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
              <th className="pivot-total-col">Total general</th>
            </tr>
          </thead>
          <tbody>
            {payload.matrix.rows.map((row) => (
              <tr key={row.row_key}>
                <td className="pivot-row-key">{row.row_key}</td>
                {row.cells.map((cell) => (
                  <td
                    key={`${row.row_key}-${cell.col_key}`}
                    className="pivot-value-cell"
                    style={{ '--heat-alpha': `${getHeatAlpha(cell.value, maxAbs)}` } as CSSProperties}
                    title={`${row.row_key} | ${cell.col_key}: ${formatValue(cell.value, payload.agg_func)} (n=${cell.count})${cell.low_sample ? ' [muestra baja]' : ''}`}
                  >
                    <div className="pivot-cell">
                      <span>{formatValue(cell.value, payload.agg_func)}</span>
                    </div>
                  </td>
                ))}
                <td className="pivot-row-total">
                  <div className="pivot-cell">
                    <span>{formatValue(row.row_total.value, payload.agg_func)}</span>
                  </div>
                </td>
              </tr>
            ))}

            <tr className="pivot-total-row">
              <td className="pivot-row-key">Total general</td>
              {payload.matrix.column_totals.map((total) => (
                <td key={total.col_key} className="pivot-row-total">
                  <div className="pivot-cell">
                    <span>{formatValue(total.value, payload.agg_func)}</span>
                  </div>
                </td>
              ))}
              <td className="pivot-row-total">
                <div className="pivot-cell">
                  <span>{formatValue(payload.matrix.grand_total.value, payload.agg_func)}</span>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
