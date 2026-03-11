import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

import { fetchPivotMetadata, fetchPivotSources, runPivotQuery } from '../api/client';
import type { PivotMetadataResponse, PivotQueryRequest, PivotQueryResponse } from '../api/types';
import PivotControls from './PivotControls';
import PivotMatrixTable from './PivotMatrixTable';
import WarningChips from './WarningChips';

type Props = {
  datasetId: string;
  source: 'in' | 'out';
};

function buildInitialRequest(metadata: PivotMetadataResponse): PivotQueryRequest {
  return {
    source: metadata.source,
    row_dim: metadata.defaults.row_dim,
    col_dim: metadata.defaults.col_dim,
    value_field: metadata.defaults.value_field,
    agg_func: metadata.defaults.agg_func,
    filters: {},
    include_blank: metadata.defaults.include_blank,
    top_k: metadata.defaults.top_k,
    small_n_threshold: metadata.defaults.small_n_threshold,
  };
}

function buildPresetRequest(
  metadata: PivotMetadataResponse,
  preset: 'example_in' | 'example_out' | 'default',
): PivotQueryRequest {
  if (preset === 'example_in') {
    return {
      source: metadata.source,
      row_dim: metadata.dimensions.includes('Owner') ? 'Owner' : metadata.defaults.row_dim,
      col_dim: metadata.dimensions.includes('Quality') ? 'Quality' : metadata.defaults.col_dim,
      value_field: metadata.value_fields.includes('Size') ? 'Size' : metadata.defaults.value_field,
      agg_func: metadata.agg_functions.includes('sum') ? 'sum' : metadata.defaults.agg_func,
      filters: {},
      include_blank: true,
      top_k: 10,
      small_n_threshold: 5,
    };
  }

  if (preset === 'example_out') {
    return {
      source: metadata.source,
      row_dim: metadata.dimensions.includes('Owner') ? 'Owner' : metadata.defaults.row_dim,
      col_dim: metadata.dimensions.includes('Size') ? 'Size' : metadata.defaults.col_dim,
      value_field: metadata.value_fields.includes('DaysInDeposit') ? 'DaysInDeposit' : metadata.defaults.value_field,
      agg_func: metadata.agg_functions.includes('mean') ? 'mean' : metadata.defaults.agg_func,
      filters: {},
      include_blank: true,
      top_k: 10,
      small_n_threshold: 5,
    };
  }

  return buildInitialRequest(metadata);
}

export default function PivotTablePanel({ datasetId, source }: Props) {
  const sourcesQuery = useQuery({
    queryKey: ['pivot-sources', datasetId],
    queryFn: () => fetchPivotSources(datasetId),
  });

  const sourceAvailable = useMemo(() => {
    const sourceItem = sourcesQuery.data?.sources.find((item) => item.source === source);
    return sourceItem?.available ?? false;
  }, [sourcesQuery.data, source]);

  const metadataQuery = useQuery({
    queryKey: ['pivot-metadata', datasetId, source],
    queryFn: () => fetchPivotMetadata(datasetId, source),
    enabled: sourceAvailable,
  });

  const [draftRequest, setDraftRequest] = useState<PivotQueryRequest | null>(null);

  useEffect(() => {
    if (!metadataQuery.data) return;
    const base = buildInitialRequest(metadataQuery.data);
    setDraftRequest(base);
  }, [metadataQuery.data]);

  const matrixQuery = useQuery<PivotQueryResponse>({
    queryKey: ['pivot-query', datasetId, source, draftRequest],
    queryFn: () => runPivotQuery(datasetId, draftRequest as PivotQueryRequest),
    enabled: !!draftRequest && !!metadataQuery.data && draftRequest.source === source,
  });

  if (sourcesQuery.isLoading) {
    return <div className="panel">Cargando fuentes pivote...</div>;
  }

  if (!sourceAvailable) {
    return (
      <div className="panel">
        <h3>Pivot {source.toUpperCase()}</h3>
        {source === 'out' ? (
          <p className="muted">
            Esta fuente no está disponible para el dataset actual. Para habilitar <strong>Pivot OUT</strong> debes
            subir <code>Grupo1_out.csv</code> (o ambos archivos).
          </p>
        ) : (
          <p className="muted">
            Esta fuente no está disponible para el dataset actual. Para habilitar <strong>Pivot IN</strong> debes
            subir <code>Grupo1_in.csv</code>.
          </p>
        )}
      </div>
    );
  }

  if (!metadataQuery.data || !draftRequest) {
    return <div className="panel">Cargando metadata de pivote...</div>;
  }

  const apiError =
    matrixQuery.error && axios.isAxiosError(matrixQuery.error)
      ? matrixQuery.error.response?.data?.message ?? matrixQuery.error.message
      : matrixQuery.error
        ? String(matrixQuery.error)
        : null;

  return (
    <section className="stack">
      <div className="panel">
        <h4>Modo rápido</h4>
        <div className="pivot-presets">
          <button
            onClick={() =>
              setDraftRequest(
                buildPresetRequest(metadataQuery.data, source === 'in' ? 'example_in' : 'example_out'),
              )
            }
          >
            Cargar ejemplo {source.toUpperCase()}
          </button>
          <button onClick={() => setDraftRequest(buildPresetRequest(metadataQuery.data, 'default'))}>
            Reset configuración
          </button>
        </div>
        <p className="muted">Tip: cambia fila/columna/valor como en tabla dinámica de Excel; se recalcula al instante.</p>
      </div>

      <PivotControls
        title={`Pivot ${source.toUpperCase()}`}
        metadata={metadataQuery.data}
        value={draftRequest}
        onChange={setDraftRequest}
        loading={matrixQuery.isFetching}
      />

      <WarningChips warnings={[...metadataQuery.data.warnings, ...(matrixQuery.data?.warnings ?? [])]} />

      {matrixQuery.data ? <PivotMatrixTable payload={matrixQuery.data} /> : null}

      {matrixQuery.isError ? <p className="error">No se pudo calcular el pivote para {source.toUpperCase()}: {apiError}</p> : null}
    </section>
  );
}
