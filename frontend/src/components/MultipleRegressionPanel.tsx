import { useMemo, useState } from 'react';
import { BlockMath, InlineMath } from 'react-katex';

import type {
  MultipleRegressionAnovaRow,
  MultipleRegressionCoefficient,
  MultipleRegressionResponse,
} from '../api/types';
import WarningChips from './WarningChips';

type Props = {
  payload: MultipleRegressionResponse;
};

type SortMode = 'coef_p' | 'coef_abs_t';
type ParsedCoefficientTerm =
  | { kind: 'intercept' }
  | { kind: 'numeric'; feature: string }
  | { kind: 'categorical'; feature: string; level: string };
type FormulaParts = {
  numericFeatures: string[];
  categoricalFeatures: string[];
};
type FormulaSummary = {
  equationLines: string[];
  equationLatex: string;
  numericFeatures: string[];
  categoricalFeatures: string[];
};
type FormulaDefinition = {
  key: string;
  symbol: string;
  symbolLatex: string;
  meaning: string;
};

function formatNumber(value: number | null, digits = 4): string {
  if (value === null || Number.isNaN(value)) return '-';
  return value.toFixed(digits);
}

function formatExp(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '-';
  return value.toExponential(3);
}

function prettifyRawToken(token: string): string {
  return token
    .replace(/C\(Q\("([^"]+)"\)\)\[T\.(.+)\]/g, '$1=$2 (vs ref)')
    .replace(/C\(Q\("([^"]+)"\)\)/g, 'cat($1)')
    .replace(/Q\("([^"]+)"\)/g, '$1')
    .replace(/C\(([^)]+)\)\[T\.(.+)\]/g, '$1=$2 (vs ref)')
    .replace(/C\(([^)]+)\)/g, 'cat($1)');
}

function prettifyFormula(raw: string | null): string {
  if (!raw) return '-';
  const cleaned = prettifyRawToken(raw);
  const [left, right] = cleaned.split('~').map((part) => part.trim());
  if (!right) return cleaned;
  const terms = right.split('+').map((term) => term.trim()).filter(Boolean);
  if (terms.length === 0) return `${left} ~ 1`;
  return `${left} ~\n  ${terms.join('\n  + ')}`;
}

function prettifyTerm(raw: string): string {
  return prettifyRawToken(raw);
}

function parseCoefficientTerm(raw: string): ParsedCoefficientTerm {
  if (raw === 'Intercept') {
    return { kind: 'intercept' };
  }

  const categoricalQuoted = raw.match(/^C\(Q\("([^"]+)"\)\)\[T\.(.+)\]$/);
  if (categoricalQuoted) {
    return { kind: 'categorical', feature: categoricalQuoted[1], level: categoricalQuoted[2] };
  }

  const categoricalSimple = raw.match(/^C\(([^)]+)\)\[T\.(.+)\]$/);
  if (categoricalSimple) {
    return { kind: 'categorical', feature: categoricalSimple[1], level: categoricalSimple[2] };
  }

  const numericQuoted = raw.match(/^Q\("([^"]+)"\)$/);
  if (numericQuoted) {
    return { kind: 'numeric', feature: numericQuoted[1] };
  }

  return { kind: 'numeric', feature: raw };
}

function buildFormulaParts(payload: MultipleRegressionResponse): FormulaParts {
  const numericTerms = new Set<string>();
  const categoricalGroups = new Map<string, Set<string>>();

  for (const coefficient of payload.coefficients) {
    const parsed = parseCoefficientTerm(coefficient.term);
    if (parsed.kind === 'intercept') continue;
    if (parsed.kind === 'numeric') {
      numericTerms.add(parsed.feature);
      continue;
    }
    const levels = categoricalGroups.get(parsed.feature) ?? new Set<string>();
    levels.add(parsed.level);
    categoricalGroups.set(parsed.feature, levels);
  }

  return {
    numericFeatures: Array.from(numericTerms).sort(),
    categoricalFeatures: Array.from(categoricalGroups.keys()).sort(),
  };
}

function buildFormulaSummary(payload: MultipleRegressionResponse): FormulaSummary {
  const parts = buildFormulaParts(payload);
  const equationLines: string[] = ['ŷ = β₀'];
  const latexTerms: string[] = ['\\beta_0'];
  for (const feature of parts.numericFeatures) {
    equationLines.push(` + β_${feature} · ${feature}`);
    const label = feature.replace(/\\/g, '\\\\').replace(/_/g, '\\_');
    latexTerms.push(`\\beta_{\\mathrm{${label}}}\\,\\mathrm{${label}}`);
  }
  for (const feature of parts.categoricalFeatures) {
    equationLines.push(` + Σ_{k=1}^{K_${feature}−1} γ_${feature},k · D_${feature},k`);
    const label = feature.replace(/\\/g, '\\\\').replace(/_/g, '\\_');
    latexTerms.push(
      `\\sum_{k=1}^{K_{\\mathrm{${label}}}-1}\\gamma_{\\mathrm{${label}},k}\\,D_{\\mathrm{${label}},k}`,
    );
  }
  equationLines.push(' + ε');

  return {
    equationLines,
    equationLatex: `\\hat{y} = ${latexTerms.join(' + ')} + \\varepsilon`,
    numericFeatures: parts.numericFeatures,
    categoricalFeatures: parts.categoricalFeatures,
  };
}

function buildFormulaDefinitions(summary: FormulaSummary): FormulaDefinition[] {
  const definitions: FormulaDefinition[] = [
    { key: 'yhat', symbol: 'ŷ', symbolLatex: '\\hat{y}', meaning: 'predicción del modelo para DaysInDeposit.' },
    { key: 'beta0', symbol: 'β₀', symbolLatex: '\\beta_0', meaning: 'intercepto: valor base esperado.' },
  ];

  for (const feature of summary.numericFeatures) {
    const symbol = `β_${feature}`;
    const label = feature.replace(/\\/g, '\\\\').replace(/_/g, '\\_');
    definitions.push({
      key: `beta-${feature}`,
      symbol,
      symbolLatex: `\\beta_{\\mathrm{${label}}}`,
      meaning: `efecto marginal de la variable numérica ${feature}.`,
    });
  }

  for (const feature of summary.categoricalFeatures) {
    const label = feature.replace(/\\/g, '\\\\').replace(/_/g, '\\_');
    definitions.push({
      key: `gamma-${feature}`,
      symbol: `γ_${feature},k`,
      symbolLatex: `\\gamma_{\\mathrm{${label}},k}`,
      meaning: `ajuste del nivel k de ${feature} respecto de su categoría base.`,
    });
    definitions.push({
      key: `dummy-${feature}`,
      symbol: `D_${feature},k`,
      symbolLatex: `D_{\\mathrm{${label}},k}`,
      meaning: `dummy (0/1): vale 1 si la fila está en el nivel k de ${feature}.`,
    });
    definitions.push({
      key: `k-${feature}`,
      symbol: `K_${feature}`,
      symbolLatex: `K_{\\mathrm{${label}}}`,
      meaning: `cantidad total de niveles observados en ${feature}.`,
    });
  }

  definitions.push({
    key: 'eps',
    symbol: 'ε',
    symbolLatex: '\\varepsilon',
    meaning: 'error residual no explicado por el modelo.',
  });
  return definitions;
}

function compareAnovaPriority(a: MultipleRegressionAnovaRow, b: MultipleRegressionAnovaRow): number {
  const ap = a.p_value ?? Number.POSITIVE_INFINITY;
  const bp = b.p_value ?? Number.POSITIVE_INFINITY;
  if (ap !== bp) return ap - bp;
  const ae = a.partial_eta_squared ?? Number.NEGATIVE_INFINITY;
  const be = b.partial_eta_squared ?? Number.NEGATIVE_INFINITY;
  return be - ae;
}

function buildAnovaConclusion(row: MultipleRegressionAnovaRow): string {
  const p = row.p_value ?? Number.POSITIVE_INFINITY;
  const eta = row.partial_eta_squared ?? 0;
  if (p <= 0.01 && eta >= 0.06) return 'Alta evidencia y efecto relevante';
  if (p <= 0.05 && eta >= 0.01) return 'Evidencia estadística con efecto moderado';
  if (p <= 0.1) return 'Tendencia débil, revisar con más datos';
  return 'Sin evidencia sólida de efecto';
}

function normalizeFilename(base: string): string {
  return base
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function triggerDownload(dataUrl: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = dataUrl;
  anchor.download = filename.endsWith('.png') ? filename : `${filename}.png`;
  anchor.click();
}

function exportFormulaAsPng(summary: FormulaSummary, definitions: FormulaDefinition[]): void {
  const numericFeaturesLabel = summary.numericFeatures.length > 0 ? summary.numericFeatures.join(', ') : 'ninguna';
  const categoricalFeaturesLabel =
    summary.categoricalFeatures.length > 0 ? summary.categoricalFeatures.join(', ') : 'ninguna';

  const lines: string[] = [
    'Regresión Múltiple (OUT -> DaysInDeposit)',
    '',
    'Ecuación por términos:',
    ...summary.equationLines,
    '',
    'Leyenda de símbolos:',
    ...definitions.map((item) => `  ${item.symbol}: ${item.meaning}`),
    '',
    'Objetivo: y = DaysInDeposit.',
    `Variables numéricas (x): ${numericFeaturesLabel}`,
    `Grupos categóricos (G): ${categoricalFeaturesLabel}`,
    'Cada variable categórica usa K_g−1 dummies (con una categoría base omitida).',
  ];

  const logicalWidth = 1900;
  const marginX = 56;
  const marginY = 52;
  const lineHeight = 36;
  const logicalHeight = marginY * 2 + lines.length * lineHeight;
  const scale = 3;

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(logicalWidth * scale);
  canvas.height = Math.round(logicalHeight * scale);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.scale(scale, scale);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, logicalWidth, logicalHeight);

  ctx.fillStyle = '#173043';
  ctx.font = "700 40px 'Cambria Math', 'Times New Roman', serif";
  ctx.fillText('Fórmula del modelo', marginX, marginY);

  let y = marginY + 46;
  for (const line of lines) {
    if (line === '') {
      y += 12;
      continue;
    }
    const isEquation = line.startsWith('ŷ =') || line.startsWith(' + ');
    const isSectionTitle = line.endsWith(':');
    ctx.font = isEquation
      ? "700 30px 'Cambria Math', 'Times New Roman', serif"
      : isSectionTitle
        ? "700 26px 'Cambria Math', 'Times New Roman', serif"
        : "500 24px 'Cambria Math', 'Times New Roman', serif";
    ctx.fillStyle = isEquation ? '#102a3a' : isSectionTitle ? '#173043' : '#2f5064';
    ctx.fillText(line, marginX, y);
    y += lineHeight;
  }

  triggerDownload(canvas.toDataURL('image/png'), normalizeFilename('formula_regresion_multiple_hd'));
}

function exportAnovaAsPng(rows: MultipleRegressionAnovaRow[]): void {
  const headers = ['Rank', 'Variable', 'F', 'p-value', 'eta² parcial', 'Conclusión'];
  const tableRows = rows.map((row, index) => [
    `${index + 1}`,
    row.feature,
    formatNumber(row.f_value, 4),
    formatExp(row.p_value),
    formatNumber(row.partial_eta_squared, 4),
    buildAnovaConclusion(row),
  ]);

  const logicalWidthMin = 1600;
  const marginX = 44;
  const marginTop = 36;
  const headerHeight = 42;
  const rowHeight = 36;
  const scale = 3;

  const measureCanvas = document.createElement('canvas');
  const mctx = measureCanvas.getContext('2d');
  if (!mctx) return;

  const headerFont = "700 22px 'Manrope', 'Segoe UI', sans-serif";
  const cellFont = "500 20px 'Manrope', 'Segoe UI', sans-serif";
  const cellPaddingX = 12;

  const colWidths = headers.map((header, colIndex) => {
    mctx.font = headerFont;
    let width = Math.ceil(mctx.measureText(header).width + cellPaddingX * 2);
    for (const row of tableRows) {
      mctx.font = cellFont;
      width = Math.max(width, Math.ceil(mctx.measureText(row[colIndex]).width + cellPaddingX * 2));
    }
    if (colIndex === 1) width = Math.max(width, 260);
    if (colIndex === 5) width = Math.max(width, 360);
    return width;
  });

  const tableWidth = colWidths.reduce((acc, w) => acc + w, 0);
  const logicalWidth = Math.max(logicalWidthMin, tableWidth + marginX * 2);
  const logicalHeight = marginTop + headerHeight + tableRows.length * rowHeight + marginX;

  const canvas = document.createElement('canvas');
  canvas.width = Math.round(logicalWidth * scale);
  canvas.height = Math.round(logicalHeight * scale);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.scale(scale, scale);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, logicalWidth, logicalHeight);

  let x = marginX;
  let y = marginTop;

  for (let c = 0; c < headers.length; c += 1) {
    const w = colWidths[c];
    ctx.fillStyle = '#eef4f8';
    ctx.fillRect(x, y, w, headerHeight);
    ctx.strokeStyle = '#d3dbe2';
    ctx.strokeRect(x, y, w, headerHeight);
    ctx.fillStyle = '#173043';
    ctx.font = headerFont;
    ctx.fillText(headers[c], x + cellPaddingX, y + 28);
    x += w;
  }

  y += headerHeight;
  for (let r = 0; r < tableRows.length; r += 1) {
    x = marginX;
    const isEven = r % 2 === 0;
    for (let c = 0; c < headers.length; c += 1) {
      const w = colWidths[c];
      ctx.fillStyle = isEven ? '#ffffff' : '#f8fbfd';
      ctx.fillRect(x, y, w, rowHeight);
      ctx.strokeStyle = '#dfe6ec';
      ctx.strokeRect(x, y, w, rowHeight);
      ctx.fillStyle = '#1f3a4a';
      ctx.font = cellFont;
      ctx.fillText(tableRows[r][c], x + cellPaddingX, y + 24);
      x += w;
    }
    y += rowHeight;
  }

  triggerDownload(canvas.toDataURL('image/png'), normalizeFilename('anova_priorizada_regresion_multiple_hd'));
}

export default function MultipleRegressionPanel({ payload }: Props) {
  const [sortMode, setSortMode] = useState<SortMode>('coef_p');

  const sortedAnova = useMemo<MultipleRegressionAnovaRow[]>(() => {
    return [...payload.anova_rows].sort(compareAnovaPriority);
  }, [payload.anova_rows]);

  const sortedCoefficients = useMemo<MultipleRegressionCoefficient[]>(() => {
    if (sortMode === 'coef_abs_t') {
      return [...payload.coefficients].sort((a, b) => Math.abs(b.t_value ?? 0) - Math.abs(a.t_value ?? 0));
    }
    return [...payload.coefficients].sort((a, b) => {
      const av = a.p_value ?? Number.POSITIVE_INFINITY;
      const bv = b.p_value ?? Number.POSITIVE_INFINITY;
      return av - bv;
    });
  }, [payload.coefficients, sortMode]);

  const formulaSummary = useMemo(() => buildFormulaSummary(payload), [payload]);
  const formulaDefinitions = useMemo(() => buildFormulaDefinitions(formulaSummary), [formulaSummary]);

  const handleExportAnovaPng = () => {
    exportAnovaAsPng(sortedAnova);
  };

  const handleExportFormulaPng = () => {
    exportFormulaAsPng(formulaSummary, formulaDefinitions);
  };

  if (!payload.target_present) {
    return (
      <div className="panel">
        <h3>Regresión Múltiple (OUT -&gt; DaysInDeposit)</h3>
        <p className="muted">No disponible: el dataset actual no tiene fuente OUT con DaysInDeposit.</p>
        <WarningChips warnings={payload.warnings} />
      </div>
    );
  }

  if (!payload.model_built) {
    return (
      <div className="panel">
        <h3>Regresión Múltiple (OUT -&gt; DaysInDeposit)</h3>
        <p className="muted">No se pudo construir un modelo válido con los datos OUT actuales.</p>
        {payload.formula ? (
          <>
            <p className="muted">Fórmula intentada</p>
            <pre className="code-block">{prettifyFormula(payload.formula)}</pre>
          </>
        ) : null}
        <WarningChips warnings={payload.warnings} />
      </div>
    );
  }

  return (
    <section className="stack">
      <div className="panel">
        <div className="table-header">
          <h3>Regresión Múltiple (OUT -&gt; DaysInDeposit)</h3>
          <div className="segmented">
            <button className={sortMode === 'coef_p' ? 'active' : ''} onClick={() => setSortMode('coef_p')}>
              Sort Coef p
            </button>
            <button className={sortMode === 'coef_abs_t' ? 'active' : ''} onClick={() => setSortMode('coef_abs_t')}>
              Sort |t|
            </button>
          </div>
        </div>

        <div className="cards-grid regression-cards">
          <article className="metric-card">
            <p>n_obs</p>
            <strong>{payload.n_obs}</strong>
          </article>
          <article className="metric-card">
            <p>n_features</p>
            <strong>{payload.n_features}</strong>
          </article>
          <article className="metric-card">
            <p>R²</p>
            <strong>{formatNumber(payload.r_squared, 4)}</strong>
          </article>
          <article className="metric-card">
            <p>R² ajustado</p>
            <strong>{formatNumber(payload.adj_r_squared, 4)}</strong>
          </article>
          <article className="metric-card">
            <p>F p-value</p>
            <strong>{formatExp(payload.f_p_value)}</strong>
          </article>
          <article className="metric-card">
            <p>AIC / BIC</p>
            <strong>
              {formatNumber(payload.aic, 1)} / {formatNumber(payload.bic, 1)}
            </strong>
          </article>
        </div>
        <p className="muted">Fuente del modelo: OUT (raw_out.csv). No se usan columnas de IN para este ajuste.</p>
      </div>

      <div className="panel">
        <div className="table-header">
          <h4>Fórmula matemática del modelo</h4>
          <button type="button" className="chart-export-btn" onClick={handleExportFormulaPng}>
            Descargar fórmula PNG HD
          </button>
        </div>
        <p className="muted">Fórmula estimada (estructura del modelo)</p>
        <pre className="code-block">{prettifyFormula(payload.formula)}</pre>
        <p className="muted">Notación profesional para reporte</p>
        <div className="math-block">
          <p className="equation-title">Ecuación por términos</p>
          <div className="latex-equation">
            <BlockMath math={formulaSummary.equationLatex} />
          </div>
          <p className="equation-title">Leyenda de símbolos</p>
          <div className="equation-defs">
            {formulaDefinitions.map((item) => (
              <p key={item.key}>
                <InlineMath math={item.symbolLatex} />: {item.meaning}
              </p>
            ))}
            <p>
              <InlineMath math={'y'} /> objetivo = <InlineMath math={'\\mathrm{DaysInDeposit}'} />.
            </p>
          </div>
        </div>
      </div>

      <div className="panel">
        <h4>Conclusiones Automáticas</h4>
        <div className="chips">
          {payload.conclusions.map((conclusion, index) => (
            <div key={`conclusion-${index}`} className="chip info">
              <span>{conclusion}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="panel table-panel">
        <div className="table-actions">
          <button type="button" className="chart-export-btn" onClick={handleExportAnovaPng}>
            PNG HD
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Variable</th>
                <th>F</th>
                <th>p-value</th>
                <th>eta² parcial</th>
                <th>Conclusión</th>
              </tr>
            </thead>
            <tbody>
              {sortedAnova.map((row, index) => (
                <tr key={row.feature}>
                  <td>{index + 1}</td>
                  <td>{row.feature}</td>
                  <td>{formatNumber(row.f_value, 4)}</td>
                  <td>{formatExp(row.p_value)}</td>
                  <td>{formatNumber(row.partial_eta_squared, 4)}</td>
                  <td>{buildAnovaConclusion(row)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel table-panel">
        <h4>Coeficientes OLS</h4>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Término</th>
                <th>Estimate</th>
                <th>Std Err</th>
                <th>t</th>
                <th>p-value</th>
                <th>CI 95% low</th>
                <th>CI 95% high</th>
              </tr>
            </thead>
            <tbody>
              {sortedCoefficients.slice(0, 60).map((row) => (
                <tr key={row.term}>
                  <td>{prettifyTerm(row.term)}</td>
                  <td>{formatNumber(row.estimate, 5)}</td>
                  <td>{formatNumber(row.std_error, 5)}</td>
                  <td>{formatNumber(row.t_value, 4)}</td>
                  <td>{formatExp(row.p_value)}</td>
                  <td>{formatNumber(row.ci_low, 5)}</td>
                  <td>{formatNumber(row.ci_high, 5)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <WarningChips warnings={payload.warnings} />
    </section>
  );
}
