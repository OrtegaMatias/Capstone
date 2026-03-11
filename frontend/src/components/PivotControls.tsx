import type { ChangeEvent } from 'react';

import type { PivotAggFunc, PivotMetadataResponse, PivotQueryRequest } from '../api/types';

type Props = {
  title: string;
  metadata: PivotMetadataResponse;
  value: PivotQueryRequest;
  onChange: (next: PivotQueryRequest) => void;
  loading: boolean;
};

function requiresDays(agg: PivotAggFunc): boolean {
  return agg.startsWith('rate_gt_');
}

function allowedAggsForField(metadata: PivotMetadataResponse, field: string): PivotAggFunc[] {
  const allowed = metadata.field_agg_functions[field];
  if (!allowed || allowed.length === 0) return metadata.agg_functions;
  return allowed;
}

function selectedValuesFromEvent(event: ChangeEvent<HTMLSelectElement>): string[] {
  return Array.from(event.target.selectedOptions).map((option) => option.value);
}

export default function PivotControls({ title, metadata, value, onChange, loading }: Props) {
  const filters = value.filters ?? {};
  const currentAllowedAggs = allowedAggsForField(metadata, value.value_field);

  return (
    <div className="panel controls">
      <h3>{title}</h3>

      <div className="control-group">
        <label>Fila</label>
        <select value={value.row_dim} onChange={(e) => onChange({ ...value, row_dim: e.target.value })}>
          {metadata.dimensions.map((dimension) => (
            <option key={dimension} value={dimension}>
              {dimension}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label>Columna</label>
        <select value={value.col_dim} onChange={(e) => onChange({ ...value, col_dim: e.target.value })}>
          {metadata.dimensions.map((dimension) => (
            <option key={dimension} value={dimension}>
              {dimension}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label>Campo valor</label>
        <select
          value={value.value_field}
          onChange={(e) => {
            const nextField = e.target.value;
            const nextAllowedAggs = allowedAggsForField(metadata, nextField);
            const nextAgg = nextAllowedAggs.includes(value.agg_func) ? value.agg_func : nextAllowedAggs[0];
            onChange({ ...value, value_field: nextField, agg_func: nextAgg });
          }}
        >
          {metadata.value_fields.map((field) => (
            <option key={field} value={field}>
              {field}
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label>Agregación</label>
        <select
          value={value.agg_func}
          onChange={(e) => {
            const agg = e.target.value as PivotAggFunc;
            let nextField = value.value_field;
            if (requiresDays(agg) && value.value_field !== 'DaysInDeposit' && metadata.value_fields.includes('DaysInDeposit')) {
              nextField = 'DaysInDeposit';
            }
            const allowedForField = allowedAggsForField(metadata, nextField);
            const nextAgg = allowedForField.includes(agg) ? agg : allowedForField[0];
            onChange({ ...value, agg_func: nextAgg, value_field: nextField });
          }}
        >
          {currentAllowedAggs.map((agg) => (
            <option key={agg} value={agg}>
              {agg}
            </option>
          ))}
        </select>
        {!currentAllowedAggs.includes('sum') && ['sum', 'mean', 'median'].some((agg) => metadata.agg_functions.includes(agg as PivotAggFunc)) ? (
          <small className="muted">Este campo no es numérico para agregaciones como sum/mean/median.</small>
        ) : null}
      </div>

      <div className="control-group">
        <label>Top-K</label>
        <input
          type="number"
          min={1}
          max={200}
          value={value.top_k ?? 10}
          onChange={(e) => onChange({ ...value, top_k: Math.max(1, Number(e.target.value || 10)) })}
        />
      </div>

      <div className="control-group">
        <label>Umbral low sample</label>
        <input
          type="number"
          min={1}
          max={200}
          value={value.small_n_threshold ?? 5}
          onChange={(e) =>
            onChange({
              ...value,
              small_n_threshold: Math.max(1, Number(e.target.value || 5)),
            })
          }
        />
      </div>

      <label className="checkbox">
        <input
          type="checkbox"
          checked={value.include_blank ?? true}
          onChange={(e) => onChange({ ...value, include_blank: e.target.checked })}
        />
        <span>Incluir (en blanco)</span>
      </label>

      <details className="pivot-advanced">
        <summary>Filtros avanzados (opcional)</summary>
        <div className="filters-grid">
          {metadata.dimensions.map((dimension) => {
            const options = metadata.filter_options[dimension] ?? [];
            const selected = filters[dimension] ?? [];

            return (
              <div key={`filter-${dimension}`} className="control-group">
                <label>Filtro {dimension}</label>
                <select
                  className="filter-multi"
                  multiple
                  size={Math.min(Math.max(options.length, 4), 8)}
                  value={selected}
                  onChange={(event) => {
                    const selectedValues = selectedValuesFromEvent(event);
                    const nextFilters: Record<string, string[]> = { ...(value.filters ?? {}) };
                    if (selectedValues.length === 0) {
                      delete nextFilters[dimension];
                    } else {
                      nextFilters[dimension] = selectedValues;
                    }
                    onChange({ ...value, filters: nextFilters });
                  }}
                >
                  {options.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
                <small className="muted">
                  {selected.length > 0 ? `${selected.length} seleccionados` : 'sin filtro (todos)'}
                </small>
              </div>
            );
          })}
        </div>
      </details>

      <p className="muted">{loading ? 'Actualizando tabla dinámica...' : 'La tabla se actualiza automáticamente.'}</p>
    </div>
  );
}
