export type Provider = "OpenAI" | "Mistral";
export type InputMethod = "csv" | "text";
export type AnalysisType = "All Data" | "Personal Details" | "Skills & Projects" | "Internal Projects Matching";
export type ShortlistingMode = "Probability Wise (Default)" | "Priority Wise (P1 / P2 / P3 Bands)";

export interface ServerConfig {
  hasOpenAiKey: boolean;
  openAiKeyCount: number;
  hasMistralKeys: boolean;
  mistralKeyCount: number;
  hasGoogleServiceAccount: boolean;
  ocrAvailable: boolean;
}

export interface JobStatusResponse {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  total: number;
  completed: number;
  progress: number;
  payload: {
    provider: Provider;
    concurrency: number;
    enableOcr: boolean;
    companyName: string;
    inputMethod: InputMethod;
    pastedText?: string;
    userRequirements: string;
    analysisType: AnalysisType;
    shortlistingMode: ShortlistingMode;
  };
  warnings: string[];
  errors: string[];
  startedAt?: string;
  finishedAt?: string;
  liveResults: Array<Record<string, string | number | boolean | null>>;
  results: Array<Record<string, string | number | boolean | null>>;
  fileName: string;
}

export interface FilterState {
  priorityBands: string[];
  searchTerm: string;
  overallRange: [number, number];
  skillsRange: [number, number];
  experienceRange: [number, number];
  projectsRange: [number, number];
  otherRange: [number, number];
  onlyInternal: boolean;
  onlyExternal: boolean;
  minTotalProjects: number;
  minInternalProjects: number;
  minExternalProjects: number;
}
