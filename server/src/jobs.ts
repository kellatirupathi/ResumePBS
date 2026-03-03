import { parse } from "csv-parse/sync";
import { v4 as uuidv4 } from "uuid";
import { config } from "./config.js";
import { processResumeComprehensively, processResumeForShortlisting } from "./analyzers.js";
import type { FilterQuery, JobCreatePayload, JobRecord, Provider, ResumeInputRow, RowResult } from "./types.js";
import { assignPriorityBand, applyFilters, nowTimestamp, orderAndSelectColumns, toCsvBuffer } from "./utils.js";
import { loadInternalProjectsString } from "./config.js";
import { writeResultsToGoogleSheets } from "./sheets.js";

const jobs = new Map<string, JobRecord>();

const normalizeRows = (rows: ResumeInputRow[]): ResumeInputRow[] => {
  return rows
    .map((row) => ({ user_id: String(row.user_id ?? "").trim(), "Resume link": String(row["Resume link"] ?? "").trim() }))
    .filter((row) => row.user_id && row["Resume link"]);
};

export const parseCsvRows = (buffer: Buffer): ResumeInputRow[] => {
  const records = parse(buffer, {
    columns: true,
    skip_empty_lines: true,
    bom: true,
    trim: true,
  }) as Record<string, string>[];

  const rows = records.map((record) => {
    const userId = record.user_id ?? record["User ID"] ?? "";
    const resumeLink = record["Resume link"] ?? record.resume_link ?? record["Resume Link"] ?? "";
    return {
      user_id: String(userId ?? ""),
      "Resume link": String(resumeLink ?? ""),
    };
  });

  return normalizeRows(rows);
};

export const parsePastedRows = (pastedText: string): ResumeInputRow[] => {
  const lines = pastedText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const rows: ResumeInputRow[] = lines.map((line) => {
    const [userId = "", resumeLink = ""] = line.split("\t");
    return {
      user_id: userId.trim(),
      "Resume link": resumeLink.trim(),
    };
  });

  return normalizeRows(rows);
};

const getActiveKeys = (provider: Provider): string[] => {
  if (provider === "OpenAI") {
    return config.openAiApiKey ? [config.openAiApiKey] : [];
  }
  return config.mistralApiKeys;
};

const runWithConcurrency = async <T>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<void>,
): Promise<void> => {
  const maxWorkers = Math.max(1, Math.min(concurrency, items.length || 1));
  let pointer = 0;

  const runners = Array.from({ length: maxWorkers }, async () => {
    while (true) {
      const index = pointer;
      pointer += 1;
      if (index >= items.length) return;
      await worker(items[index], index);
    }
  });

  await Promise.all(runners);
};

const processJob = async (job: JobRecord): Promise<void> => {
  job.status = "running";
  job.startedAt = new Date().toISOString();

  try {
    const keys = getActiveKeys(job.payload.provider);
    if (!keys.length) {
      throw new Error(
        job.payload.provider === "OpenAI"
          ? "OPENAI_API_KEY is missing."
          : "MISTRAL_API_KEY_1..12 (or MISTRAL_API_KEY) is missing.",
      );
    }

    const internalProjectsString = loadInternalProjectsString();
    const isShortlisting = job.payload.userRequirements.trim().length > 0;

    await runWithConcurrency(job.rows, job.payload.concurrency, async (row, index) => {
      const apiKey = keys[index % keys.length];
      let result: RowResult;

      if (isShortlisting) {
        result = await processResumeForShortlisting(row, index, {
          provider: job.payload.provider,
          apiKey,
          enableOcr: job.payload.enableOcr,
          companyName: job.payload.companyName,
          userRequirements: job.payload.userRequirements,
          analysisType: job.payload.analysisType,
          internalProjectsString,
        });

        if (job.payload.shortlistingMode === "Priority Wise (P1 / P2 / P3 Bands)") {
          result["Priority Band"] = assignPriorityBand(result["Overall Probability"]);
        }
      } else {
        result = await processResumeComprehensively(row, index, {
          provider: job.payload.provider,
          apiKey,
          enableOcr: job.payload.enableOcr,
          companyName: job.payload.companyName,
          userRequirements: job.payload.userRequirements,
          analysisType: job.payload.analysisType,
          internalProjectsString,
        });
      }

      result["Analysis Datetime"] = nowTimestamp();
      job.results.push(result);
      job.completed += 1;
    });

    const mode = isShortlisting ? "shortlisting" : job.payload.analysisType;
    const ordered = orderAndSelectColumns(job.results, mode, job.payload.shortlistingMode);
    job.results = ordered.rows;
    job.fileName = ordered.fileName;

    const sheetWrite = await writeResultsToGoogleSheets(mode, job.payload.shortlistingMode, job.results);
    if (!sheetWrite.ok && sheetWrite.warning) {
      job.warnings.push(sheetWrite.warning);
    }

    job.status = "completed";
    job.finishedAt = new Date().toISOString();
  } catch (error) {
    job.status = "failed";
    job.finishedAt = new Date().toISOString();
    job.errors.push(error instanceof Error ? error.message : String(error));
  }
};

export const createJob = (payload: JobCreatePayload, rows: ResumeInputRow[]): JobRecord => {
  const normalizedRows = normalizeRows(rows);
  const id = uuidv4();

  const job: JobRecord = {
    id,
    status: "queued",
    payload,
    rows: normalizedRows,
    total: normalizedRows.length,
    completed: 0,
    results: [],
    errors: [],
    warnings: [],
    fileName: "results.csv",
  };

  jobs.set(id, job);

  void processJob(job);
  return job;
};

export const getJob = (jobId: string): JobRecord | undefined => jobs.get(jobId);

export const serializeJob = (job: JobRecord) => ({
  id: job.id,
  status: job.status,
  total: job.total,
  completed: job.completed,
  progress: job.total > 0 ? job.completed / job.total : 0,
  payload: job.payload,
  warnings: job.warnings,
  errors: job.errors,
  startedAt: job.startedAt,
  finishedAt: job.finishedAt,
  results: job.results,
  fileName: job.fileName,
});

export const getJobResults = (jobId: string): RowResult[] => {
  const job = jobs.get(jobId);
  if (!job) return [];
  return job.results;
};

export const getFilteredCsv = (jobId: string, filters: FilterQuery): { buffer: Buffer; fileName: string } | null => {
  const job = jobs.get(jobId);
  if (!job) return null;

  const filteredRows = applyFilters(job.results, filters);
  const columns = filteredRows.length > 0 ? Object.keys(filteredRows[0]) : Object.keys(job.results[0] ?? {});
  const buffer = toCsvBuffer(filteredRows, columns);
  const fileName = job.fileName.startsWith("filtered_") ? job.fileName : `filtered_${job.fileName}`;

  return { buffer, fileName };
};
