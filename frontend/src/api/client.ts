import axios from 'axios';

import type {
  AnovaResponse,
  EDAResponse,
  WeekAcademicEDAResponse,
  WeekClusteringResponse,
  FrameworkSummary,
  MlEvaluationSummary,
  MultipleRegressionResponse,
  NotesResponse,
  NotesSaveResponse,
  PivotMetadataResponse,
  PivotQueryRequest,
  PivotQueryResponse,
  PivotSourcesResponse,
  PreviewResponse,
  SupervisedOverviewResponse,
  UploadResponse,
  VariabilityResponse,
  WeekConfig,
  WeekReportSummary,
} from './types';

const defaultApiBase =
  typeof window !== 'undefined'
    ? `${window.location.protocol}//${window.location.hostname}:8000/api/v1`
    : 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? defaultApiBase,
  timeout: 30000,
});

export async function uploadDataset(inFile?: File | null, outFile?: File | null): Promise<UploadResponse> {
  const formData = new FormData();
  if (inFile) formData.append('in_file', inFile);
  if (outFile) formData.append('out_file', outFile);

  const { data } = await api.post<UploadResponse>('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function fetchPreview(datasetId: string, limit = 20): Promise<PreviewResponse> {
  const { data } = await api.get<PreviewResponse>(`/datasets/${datasetId}/preview`, { params: { limit } });
  return data;
}

export async function fetchFrameworkSummary(): Promise<FrameworkSummary> {
  const { data } = await api.get<FrameworkSummary>('/framework');
  return data;
}

export async function fetchWeekConfig(weekId: string): Promise<WeekConfig> {
  const { data } = await api.get<WeekConfig>(`/weeks/${weekId}`);
  return data;
}

export async function fetchWeekPreview(weekId: string, limit = 20): Promise<PreviewResponse> {
  const { data } = await api.get<PreviewResponse>(`/weeks/${weekId}/preview`, { params: { limit } });
  return data;
}

export async function fetchWeekEDA(weekId: string): Promise<WeekAcademicEDAResponse> {
  const { data } = await api.get<WeekAcademicEDAResponse>(`/weeks/${weekId}/eda`, { timeout: 120000 });
  return data;
}

export async function fetchWeekClustering(weekId: string): Promise<WeekClusteringResponse> {
  const { data } = await api.get<WeekClusteringResponse>(`/weeks/${weekId}/clustering`, { timeout: 120000 });
  return data;
}

export async function fetchWeekMlOverview(weekId: string): Promise<MlEvaluationSummary> {
  const { data } = await api.get<MlEvaluationSummary>(`/weeks/${weekId}/ml/overview`);
  return data;
}

export async function fetchWeekNotes(weekId: string): Promise<NotesResponse> {
  const { data } = await api.get<NotesResponse>(`/weeks/${weekId}/notes`);
  return data;
}

export async function saveWeekNotes(weekId: string, content: string): Promise<NotesSaveResponse> {
  const { data } = await api.put<NotesSaveResponse>(`/weeks/${weekId}/notes`, { content });
  return data;
}

export async function fetchWeekReport(weekId: string): Promise<WeekReportSummary> {
  const { data } = await api.get<WeekReportSummary>(`/weeks/${weekId}/report`);
  return data;
}

export async function refreshWeekReport(weekId: string): Promise<WeekReportSummary> {
  const { data } = await api.post<WeekReportSummary>(`/weeks/${weekId}/report/refresh`);
  return data;
}

export async function fetchEDA(datasetId: string): Promise<EDAResponse> {
  const { data } = await api.get<EDAResponse>(`/datasets/${datasetId}/eda`);
  return data;
}

export async function fetchVariability(
  datasetId: string,
  customMode: 'freq_only' | 'ordinal_map',
  ordinalStrategy: 'frequency' | 'alphabetical',
): Promise<VariabilityResponse> {
  const { data } = await api.get<VariabilityResponse>(`/datasets/${datasetId}/variability`, {
    params: {
      custom_mode: customMode,
      ordinal_strategy: ordinalStrategy,
    },
  });
  return data;
}

export async function fetchSupervisedOverview(datasetId: string): Promise<SupervisedOverviewResponse> {
  const { data } = await api.get<SupervisedOverviewResponse>(`/datasets/${datasetId}/supervised/overview`);
  return data;
}

export async function fetchAnova(datasetId: string): Promise<AnovaResponse> {
  const { data } = await api.get<AnovaResponse>(`/datasets/${datasetId}/anova`);
  return data;
}

export async function fetchMultipleRegression(datasetId: string): Promise<MultipleRegressionResponse> {
  const { data } = await api.get<MultipleRegressionResponse>(`/datasets/${datasetId}/supervised/multiple-regression`);
  return data;
}

export async function fetchNotes(datasetId: string): Promise<NotesResponse> {
  const { data } = await api.get<NotesResponse>(`/datasets/${datasetId}/notes`);
  return data;
}

export async function saveNotes(datasetId: string, content: string): Promise<NotesSaveResponse> {
  const { data } = await api.put<NotesSaveResponse>(`/datasets/${datasetId}/notes`, { content });
  return data;
}

export async function fetchPivotSources(datasetId: string): Promise<PivotSourcesResponse> {
  const { data } = await api.get<PivotSourcesResponse>(`/datasets/${datasetId}/pivot/sources`);
  return data;
}

export async function fetchPivotMetadata(datasetId: string, source: 'in' | 'out'): Promise<PivotMetadataResponse> {
  const { data } = await api.get<PivotMetadataResponse>(`/datasets/${datasetId}/pivot/metadata`, {
    params: { source },
  });
  return data;
}

export async function runPivotQuery(datasetId: string, payload: PivotQueryRequest): Promise<PivotQueryResponse> {
  const { data } = await api.post<PivotQueryResponse>(`/datasets/${datasetId}/pivot/query`, payload);
  return data;
}
