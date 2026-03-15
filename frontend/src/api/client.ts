import axios from 'axios';

import type {
  WeekAcademicEDAResponse,
  WeekClusteringResponse,
  FrameworkSummary,
  MlEvaluationSummary,
  NotesResponse,
  NotesSaveResponse,
  PreviewResponse,
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

export async function fetchWeekMlCached(weekId: string): Promise<MlEvaluationSummary | null> {
  const resp = await api.get(`/weeks/${weekId}/ml/cached`, {
    validateStatus: (s) => s === 200 || s === 204,
  });
  return resp.status === 204 ? null : resp.data;
}

export function getApiBaseUrl(): string {
  return (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? defaultApiBase;
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

