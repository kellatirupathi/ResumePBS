import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import type { Provider, ProviderConfig } from "./types.js";

const appEnvPath = process.env.APP_ENV_PATH ? path.resolve(process.env.APP_ENV_PATH) : "";
const cwdEnvPath = path.resolve(process.cwd(), ".env");
const parentEnvPath = path.resolve(process.cwd(), "..", ".env");
const envCandidates = [appEnvPath, cwdEnvPath, parentEnvPath].filter(Boolean);
const resolvedEnvPath = envCandidates.find((candidate) => fs.existsSync(candidate));
if (resolvedEnvPath) {
  dotenv.config({ path: resolvedEnvPath });
} else {
  dotenv.config();
}
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const envDirectory = resolvedEnvPath ? path.dirname(resolvedEnvPath) : "";

const getEnv = (key: string): string => process.env[key]?.trim() ?? "";

const resolvePathFromCandidates = (filePath: string): string => {
  const normalized = filePath.trim();
  if (!normalized) return "";

  const candidatePaths = [
    path.isAbsolute(normalized) ? normalized : "",
    envDirectory ? path.resolve(envDirectory, normalized) : "",
    path.resolve(process.cwd(), normalized),
    path.resolve(process.cwd(), "..", normalized),
  ].filter(Boolean);

  const resolved = candidatePaths.find((candidate) => fs.existsSync(candidate));
  return resolved ?? "";
};

const getMistralKeys = (): string[] => {
  const numbered = Array.from({ length: 12 }, (_, idx) => getEnv(`MISTRAL_API_KEY_${idx + 1}`)).filter(Boolean);
  if (numbered.length > 0) {
    return numbered;
  }
  const single = getEnv("MISTRAL_API_KEY");
  return single ? [single] : [];
};

const getOpenAiKeys = (): string[] => {
  const numbered = Array.from({ length: 12 }, (_, idx) => getEnv(`OPENAI_API_KEY_${idx + 1}`)).filter(Boolean);
  if (numbered.length > 0) {
    return numbered;
  }
  const single = getEnv("OPENAI_API_KEY");
  return single ? [single] : [];
};

const openAiApiKeys = getOpenAiKeys();
const bigQueryProjectId = getEnv("BIGQUERY_PROJECT_ID") || "kossip-helpers";
const bigQueryDatasetId = getEnv("BIGQUERY_DATASET_ID") || "placement_support_analytics";
const bigQueryStoreDatasetId = getEnv("BIGQUERY_STORE_DATASET_ID") || getEnv("STORE_DATASET_ID") || "ps_interview_intel";
const bigQueryStoreTableId = getEnv("BIGQUERY_STORE_TABLE_ID") || getEnv("STORE_TABLE_ID") || "resume_job_matcher_tool_logs";
const bigQueryUserIdColumn = getEnv("BIGQUERY_USER_ID_COLUMN") || "user_id";
const configuredBigQueryServicePath = getEnv("BIGQUERY_SERVICE_ACCOUNT_PATH");
const defaultBigQueryServicePath = "kossip-helpers-fb18bd11f754.json";

export const config = {
  port: Number(getEnv("PORT") || 4000),
  openAiApiKeys,
  openAiApiKey: openAiApiKeys[0] ?? "",
  mistralApiKeys: getMistralKeys(),
  googleServiceAccountJson: getEnv("GOOGLE_SERVICE_ACCOUNT_JSON"),
  googleServiceAccountPath: getEnv("GOOGLE_SERVICE_ACCOUNT_PATH"),
  bigQueryProjectId,
  bigQueryDatasetId,
  bigQueryStoreDatasetId,
  bigQueryStoreTableId,
  bigQueryUserIdColumn,
  bigQueryServiceAccountPath:
    resolvePathFromCandidates(configuredBigQueryServicePath) ||
    resolvePathFromCandidates(defaultBigQueryServicePath),
  internalProjectListPath: path.resolve(__dirname, "..", "resources", "INTERNAL_PROJECT_LIST.txt"),
};

export const getRecommendedConcurrency = (provider: Provider): number => {
  if (provider === "OpenAI") {
    return 12;
  }
  return Math.max(1, Math.min(20, config.mistralApiKeys.length || 1));
};

export const getProviderConcurrencyCap = (provider: Provider): number => {
  if (provider === "OpenAI") {
    return 20;
  }
  return 20;
};

export const maskKey = (value: string): string => {
  if (!value) return "";
  return value.length > 4 ? `...${value.slice(-4)}` : "...";
};

export const loadInternalProjectsString = (): string => {
  const filePath = config.internalProjectListPath;
  if (!fs.existsSync(filePath)) {
    return "";
  }

  const rows = fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  return rows.join(", ");
};

export const loadGoogleServiceAccount = (): Record<string, unknown> | null => {
  try {
    if (config.googleServiceAccountJson) {
      return JSON.parse(config.googleServiceAccountJson);
    }

    if (config.googleServiceAccountPath) {
      const resolvedPath = resolvePathFromCandidates(config.googleServiceAccountPath);
      if (!resolvedPath) {
        return null;
      }
      const raw = fs.readFileSync(resolvedPath, "utf8");
      return JSON.parse(raw);
    }

    return null;
  } catch {
    return null;
  }
};

export const getProviderConfig = (ocrAvailable: boolean): ProviderConfig => {
  const serviceAccount = loadGoogleServiceAccount();
  return {
    hasOpenAiKey: config.openAiApiKeys.length > 0,
    openAiKeyCount: config.openAiApiKeys.length,
    hasMistralKeys: config.mistralApiKeys.length > 0,
    mistralKeyCount: config.mistralApiKeys.length,
    hasGoogleServiceAccount: Boolean(serviceAccount),
    ocrAvailable,
  };
};
