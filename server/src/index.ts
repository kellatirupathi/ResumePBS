import fs from "node:fs";
import express from "express";
import cors from "cors";
import morgan from "morgan";
import multer from "multer";
import { config, getProviderConcurrencyCap, getProviderConfig, getRecommendedConcurrency } from "./config.js";
import { ANALYSIS_TYPES, SHORTLISTING_MODES } from "./constants.js";
import { createJob, getFilteredCsv, getJob, getJobResults, parseCsvRows, parsePastedRows, serializeJob } from "./jobs.js";
import { fetchBigQueryDatasetSnapshot, fetchBigQueryTableSnapshot, fetchLearningMetricsByUserIds } from "./bigquery.js";
import type { AnalysisType, DesktopUpdateStatus, FilterQuery, InputMethod, JobCreatePayload, Provider, ShortlistingMode } from "./types.js";
import { isOcrAvailable } from "./extraction.js";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: "2mb" }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan("dev"));

const parseBoolean = (value: unknown, fallback = false): boolean => {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const lower = value.toLowerCase().trim();
    if (["true", "1", "yes", "on"].includes(lower)) return true;
    if (["false", "0", "no", "off"].includes(lower)) return false;
  }
  return fallback;
};

const parseNumber = (value: unknown, fallback: number): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const parseFilterQuery = (query: Record<string, unknown>): FilterQuery => {
  const parseMaybeNumber = (key: string): number | undefined => {
    if (query[key] === undefined) return undefined;
    const n = Number(query[key]);
    return Number.isFinite(n) ? n : undefined;
  };

  const priorityBandsRaw = String(query.priorityBands ?? "").trim();

  return {
    priorityBands: priorityBandsRaw ? priorityBandsRaw.split(",").map((item) => item.trim()).filter(Boolean) : undefined,
    searchTerm: String(query.searchTerm ?? "").trim() || undefined,
    overallMin: parseMaybeNumber("overallMin"),
    overallMax: parseMaybeNumber("overallMax"),
    skillsMin: parseMaybeNumber("skillsMin"),
    skillsMax: parseMaybeNumber("skillsMax"),
    expMin: parseMaybeNumber("expMin"),
    expMax: parseMaybeNumber("expMax"),
    projectsMin: parseMaybeNumber("projectsMin"),
    projectsMax: parseMaybeNumber("projectsMax"),
    otherMin: parseMaybeNumber("otherMin"),
    otherMax: parseMaybeNumber("otherMax"),
    onlyInternal: parseBoolean(query.onlyInternal, false),
    onlyExternal: parseBoolean(query.onlyExternal, false),
    minTotalProjects: parseMaybeNumber("minTotalProjects"),
    minInternalProjects: parseMaybeNumber("minInternalProjects"),
    minExternalProjects: parseMaybeNumber("minExternalProjects"),
  };
};

const parseUserIds = (input: unknown): string[] => {
  if (Array.isArray(input)) {
    return input.map((value) => String(value ?? "").trim()).filter(Boolean);
  }

  if (typeof input === "string") {
    return input
      .split(/[\r\n,\t ]+/)
      .map((value) => value.trim())
      .filter(Boolean);
  }

  return [];
};

const readDesktopUpdateStatus = (): DesktopUpdateStatus => {
  const statusPath = String(process.env.DESKTOP_UPDATE_STATUS_PATH ?? "").trim();
  const currentVersion = String(process.env.DESKTOP_APP_VERSION ?? "0.0.0").trim() || "0.0.0";

  if (!statusPath) {
    return {
      isDesktopApp: false,
      currentVersion,
      phase: "unsupported",
      message: "",
    };
  }

  if (!fs.existsSync(statusPath)) {
    return {
      isDesktopApp: true,
      currentVersion,
      phase: "idle",
      message: "",
    };
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(statusPath, "utf8")) as Partial<DesktopUpdateStatus>;
    return {
      isDesktopApp: true,
      currentVersion,
      phase: parsed.phase ?? "idle",
      availableVersion: parsed.availableVersion,
      progressPercent: parsed.progressPercent,
      message: parsed.message ?? "",
      checkedAt: parsed.checkedAt,
    };
  } catch {
    return {
      isDesktopApp: true,
      currentVersion,
      phase: "error",
      message: "Failed to load desktop update status.",
    };
  }
};

const writeDesktopUpdateCommand = (action: "installNow" | "checkNow", availableVersion?: string): boolean => {
  const commandPath = String(process.env.DESKTOP_UPDATE_COMMAND_PATH ?? "").trim();
  if (!commandPath) return false;

  fs.writeFileSync(
    commandPath,
    JSON.stringify(
      {
        action,
        availableVersion,
        requestedAt: new Date().toISOString(),
      },
      null,
      2,
    ),
    "utf8",
  );

  return true;
};

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, timestamp: new Date().toISOString() });
});

app.get("/api/config", (_req, res) => {
  res.json(getProviderConfig(isOcrAvailable()));
});

app.get("/api/desktop/update-status", (_req, res) => {
  res.json(readDesktopUpdateStatus());
});

app.post("/api/desktop/update-action", (req, res) => {
  const action = String(req.body?.action ?? "").trim();
  if (action !== "installNow" && action !== "checkNow") {
    return res.status(400).json({ error: "action must be installNow or checkNow." });
  }

  const ok = writeDesktopUpdateCommand(action, String(req.body?.availableVersion ?? "").trim());
  if (!ok) {
    return res.status(400).json({ error: "Desktop update controls are unavailable in this environment." });
  }

  return res.json({ ok: true });
});

app.get("/api/bigquery", async (_req, res) => {
  try {
    const snapshot = await fetchBigQueryDatasetSnapshot();
    return res.json(snapshot);
  } catch (error) {
    return res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.get("/api/bigquery/table/:tableId", async (req, res) => {
  try {
    const tableId = String(req.params.tableId ?? "").trim();
    if (!tableId) {
      return res.status(400).json({ error: "tableId is required." });
    }

    const snapshot = await fetchBigQueryTableSnapshot(tableId);
    return res.json(snapshot);
  } catch (error) {
    return res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post("/api/bigquery/learning-metrics", async (req, res) => {
  try {
    const userIds = parseUserIds(req.body?.userIds);
    if (userIds.length === 0) {
      return res.status(400).json({ error: "userIds is required. Provide one or more user IDs." });
    }

    const result = await fetchLearningMetricsByUserIds(userIds);
    return res.json(result);
  } catch (error) {
    return res.status(500).json({
      error: error instanceof Error ? error.message : String(error),
    });
  }
});

app.post("/api/jobs", upload.single("csvFile"), (req, res) => {
  try {
    const provider = String(req.body.provider ?? "OpenAI") as Provider;
    if (!["OpenAI", "Mistral"].includes(provider)) {
      return res.status(400).json({ error: "provider must be OpenAI or Mistral" });
    }

    const inputMethod = String(req.body.inputMethod ?? "text") as InputMethod;
    if (!["csv", "text"].includes(inputMethod)) {
      return res.status(400).json({ error: "inputMethod must be csv or text" });
    }

    const companyName = String(req.body.companyName ?? "").trim();
    if (!companyName) {
      return res.status(400).json({ error: "Company Name is required." });
    }

    const userRequirements = String(req.body.userRequirements ?? "");
    const analysisType = String(req.body.analysisType ?? "All Data") as AnalysisType;
    const shortlistingMode = String(req.body.shortlistingMode ?? "Probability Wise (Default)") as ShortlistingMode;

    if (!ANALYSIS_TYPES.includes(analysisType)) {
      return res.status(400).json({ error: `analysisType must be one of: ${ANALYSIS_TYPES.join(", ")}` });
    }

    if (!SHORTLISTING_MODES.includes(shortlistingMode)) {
      return res.status(400).json({ error: `shortlistingMode must be one of: ${SHORTLISTING_MODES.join(", ")}` });
    }

    const providerConfig = getProviderConfig(isOcrAvailable());
    if (provider === "OpenAI" && !providerConfig.hasOpenAiKey) {
      return res.status(400).json({ error: "OPENAI_API_KEY not configured." });
    }
    if (provider === "Mistral" && !providerConfig.hasMistralKeys) {
      return res.status(400).json({ error: "MISTRAL_API_KEY_1..12 or MISTRAL_API_KEY not configured." });
    }

    const defaultConcurrency = getRecommendedConcurrency(provider);
    const concurrencyCap = getProviderConcurrencyCap(provider);
    const concurrency = Math.max(1, Math.min(concurrencyCap, parseNumber(req.body.concurrency, defaultConcurrency)));
    const enableOcr = parseBoolean(req.body.enableOcr, true);

    let rows;
    if (inputMethod === "csv") {
      if (!req.file?.buffer) {
        return res.status(400).json({ error: "csvFile is required for csv inputMethod." });
      }
      rows = parseCsvRows(req.file.buffer);
    } else {
      const pastedText = String(req.body.pastedText ?? "");
      rows = parsePastedRows(pastedText);
    }

    if (!rows.length) {
      return res.status(400).json({ error: "No valid resume rows found. Provide user_id and Resume link data." });
    }

    const payload: JobCreatePayload = {
      provider,
      concurrency,
      enableOcr,
      companyName,
      inputMethod,
      pastedText: String(req.body.pastedText ?? ""),
      userRequirements,
      analysisType,
      shortlistingMode,
    };

    const job = createJob(payload, rows);

    return res.status(202).json({
      jobId: job.id,
      totalRows: job.total,
      message: `Started ${companyName} with ${job.total} resume(s).`,
    });
  } catch (error) {
    return res.status(500).json({ error: error instanceof Error ? error.message : String(error) });
  }
});

app.get("/api/jobs/:jobId", (req, res) => {
  const job = getJob(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }

  return res.json(serializeJob(job));
});

app.get("/api/jobs/:jobId/results", (req, res) => {
  const job = getJob(req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: "Job not found" });
  }

  return res.json({
    jobId: job.id,
    status: job.status,
    total: job.total,
    completed: job.completed,
    fileName: job.fileName,
    results: getJobResults(job.id),
  });
});

app.get("/api/jobs/:jobId/download", (req, res) => {
  const parsedFilters = parseFilterQuery(req.query as Record<string, unknown>);
  const csv = getFilteredCsv(req.params.jobId, parsedFilters);

  if (!csv) {
    return res.status(404).json({ error: "Job not found" });
  }

  res.setHeader("Content-Type", "text/csv; charset=utf-8");
  res.setHeader("Content-Disposition", `attachment; filename=\"${csv.fileName}\"`);
  return res.send(csv.buffer);
});

app.listen(config.port, () => {
  console.log(`Server running at http://localhost:${config.port}`);
});
