import type { ANALYSIS_TYPES, SHORTLISTING_MODES } from "./constants.js";

export type Provider = "OpenAI" | "Mistral";

export type AnalysisType = (typeof ANALYSIS_TYPES)[number];
export type ShortlistingMode = (typeof SHORTLISTING_MODES)[number];

export type InputMethod = "csv" | "text";

export interface ResumeInputRow {
  user_id: string;
  "Resume link": string;
}

export interface JobCreatePayload {
  provider: Provider;
  concurrency: number;
  enableOcr: boolean;
  companyName: string;
  inputMethod: InputMethod;
  pastedText?: string;
  userRequirements: string;
  analysisType: AnalysisType;
  shortlistingMode: ShortlistingMode;
}

export type RowResult = Record<string, string | number | boolean | null>;

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface JobRecord {
  id: string;
  status: JobStatus;
  payload: JobCreatePayload;
  rows: ResumeInputRow[];
  total: number;
  completed: number;
  results: RowResult[];
  errors: string[];
  warnings: string[];
  startedAt?: string;
  finishedAt?: string;
  fileName: string;
}

export interface FilterQuery {
  priorityBands?: string[];
  searchTerm?: string;
  overallMin?: number;
  overallMax?: number;
  skillsMin?: number;
  skillsMax?: number;
  expMin?: number;
  expMax?: number;
  projectsMin?: number;
  projectsMax?: number;
  otherMin?: number;
  otherMax?: number;
  onlyInternal?: boolean;
  onlyExternal?: boolean;
  minTotalProjects?: number;
  minInternalProjects?: number;
  minExternalProjects?: number;
}

export interface ProviderConfig {
  hasOpenAiKey: boolean;
  hasMistralKeys: boolean;
  mistralKeyCount: number;
  hasGoogleServiceAccount: boolean;
  ocrAvailable: boolean;
}
