import axios from "axios";
import type {
  AnalysisType,
  BigQueryTableDataResponse,
  BigQueryLearningMetricsResponse,
  BigQueryDatasetSnapshot,
  DesktopUpdateStatus,
  InputMethod,
  JobStatusResponse,
  Provider,
  ServerConfig,
  ShortlistingMode,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:4010";

const client = axios.create({ baseURL: API_BASE });

export interface StartJobPayload {
  provider: Provider;
  concurrency: number;
  enableOcr: boolean;
  companyName: string;
  inputMethod: InputMethod;
  csvFile?: File | null;
  pastedText: string;
  userRequirements: string;
  analysisType: AnalysisType;
  shortlistingMode: ShortlistingMode;
}

export const getConfig = async (): Promise<ServerConfig> => {
  const { data } = await client.get<ServerConfig>("/api/config");
  return data;
};

export const getDesktopUpdateStatus = async (): Promise<DesktopUpdateStatus> => {
  const { data } = await client.get<DesktopUpdateStatus>("/api/desktop/update-status");
  return data;
};

export const requestDesktopUpdateInstall = async (availableVersion?: string): Promise<{ ok: true }> => {
  const { data } = await client.post<{ ok: true }>("/api/desktop/update-action", {
    action: "installNow",
    availableVersion,
  });
  return data;
};

export const startJob = async (payload: StartJobPayload): Promise<{ jobId: string; totalRows: number; message: string }> => {
  const formData = new FormData();
  formData.append("provider", payload.provider);
  formData.append("concurrency", String(payload.concurrency));
  formData.append("enableOcr", String(payload.enableOcr));
  formData.append("companyName", payload.companyName);
  formData.append("inputMethod", payload.inputMethod);
  formData.append("pastedText", payload.pastedText);
  formData.append("userRequirements", payload.userRequirements);
  formData.append("analysisType", payload.analysisType);
  formData.append("shortlistingMode", payload.shortlistingMode);

  if (payload.csvFile) {
    formData.append("csvFile", payload.csvFile);
  }

  const { data } = await client.post<{ jobId: string; totalRows: number; message: string }>("/api/jobs", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
};

export const getJobStatus = async (jobId: string): Promise<JobStatusResponse> => {
  const { data } = await client.get<JobStatusResponse>(`/api/jobs/${jobId}`);
  return data;
};

export const getBigQueryDataset = async (): Promise<BigQueryDatasetSnapshot> => {
  const { data } = await client.get<BigQueryDatasetSnapshot>("/api/bigquery");
  return data;
};

export const getBigQueryTableData = async (tableId: string): Promise<BigQueryTableDataResponse> => {
  const { data } = await client.get<BigQueryTableDataResponse>(`/api/bigquery/table/${encodeURIComponent(tableId)}`);
  return data;
};

export const getLearningMetricsByUserIds = async (userIds: string[]): Promise<BigQueryLearningMetricsResponse> => {
  const { data } = await client.post<BigQueryLearningMetricsResponse>("/api/bigquery/learning-metrics", { userIds });
  return data;
};

export const buildDownloadUrl = (jobId: string, params: Record<string, string | number | boolean | undefined>): string => {
  const qp = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      qp.set(key, String(value));
    }
  });
  return `${API_BASE}/api/jobs/${jobId}/download?${qp.toString()}`;
};
