import { useEffect, useMemo, useRef, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  Drawer,
  FormControl,
  FormControlLabel,
  FormLabel,
  Grid,
  InputLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Slider,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import axios from "axios";
import { buildDownloadUrl, getConfig, getDesktopUpdateStatus, getJobStatus, requestDesktopUpdateInstall, startJob, type StartJobPayload } from "./api";
import { applyFilters, defaultFilters, priorityOrder, sortAndOrderRows, toQueryParamsFromFilters, type TableRow } from "./helpers";
import type { AnalysisType, DesktopUpdateStatus, FilterState, InputMethod, JobStatusResponse, Provider, ServerConfig, ShortlistingMode } from "./types";

const drawerWidth = 320;
const rightPanelScale = 0.9;
const RUN_CANCELLED_MESSAGE = "__RUN_CANCELLED__";

const analysisOptions: AnalysisType[] = ["All Data", "Personal Details", "Skills & Projects", "Internal Projects Matching"];
const shortlistingOptions: ShortlistingMode[] = ["Probability Wise (Default)", "Priority Wise (P1 / P2 / P3 Bands)", "Sectionwise"];

type WorkflowMode = "single" | "multi";
type MultiCompanyRunStatus = "pending" | "starting" | "running" | "completed" | "failed" | "stopped";
type BatchPhase = "idle" | "running" | "waiting" | "completed" | "stopped";

interface DraftCompanyEntry {
  companyName: string;
  userRequirements: string;
  inputMethod: InputMethod;
  csvFile: File | null;
  pastedText: string;
}

interface MultiCompanyEntry extends DraftCompanyEntry {
  id: string;
  runStatus: MultiCompanyRunStatus;
  statusMessage: string;
  jobId?: string;
  resultFileName?: string;
}

interface BatchRunState {
  phase: BatchPhase;
  currentIndex: number;
  total: number;
  countdownSeconds: number;
}

const initialBatchRunState: BatchRunState = {
  phase: "idle",
  currentIndex: -1,
  total: 0,
  countdownSeconds: 0,
};

const getRecommendedConcurrency = (provider: Provider, config: ServerConfig | null): number => {
  if (provider === "OpenAI") {
    return 12;
  }
  return Math.max(1, Math.min(20, config?.mistralKeyCount || 1));
};

const getMaxConcurrency = (provider: Provider, _config: ServerConfig | null): number => {
  if (provider === "OpenAI") {
    return 20;
  }
  return 20;
};

const formatElapsedDuration = (milliseconds: number): string => {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
};

const toGridColumns = (columns: string[]): GridColDef[] => {
  return columns.map((column) => ({
    field: column,
    headerName: column,
    flex: 1,
    minWidth: 190,
    sortable: true,
    renderCell: (params) => {
      const value = String(params.value ?? "");
      return (
        <Box
          title={value}
          sx={{
            width: "100%",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {value}
        </Box>
      );
    },
  }));
};

const sleep = (milliseconds: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });

const countPastedResumeLines = (value: string): number =>
  value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean).length;

const createMultiCompanyEntryId = (): string => `company_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

const sanitizeFileSegment = (value: string, fallback: string): string => {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  return cleaned || fallback;
};

const getDraftHasContent = (draft: DraftCompanyEntry): boolean =>
  Boolean(draft.companyName.trim() || draft.userRequirements.trim() || draft.csvFile || draft.pastedText.trim());

const getDraftValidationError = (draft: DraftCompanyEntry): string | null => {
  if (!draft.companyName.trim()) {
    return "Company Name is required.";
  }

  if (draft.inputMethod === "csv" && !draft.csvFile) {
    return "Upload a CSV file for this company.";
  }

  if (draft.inputMethod === "text" && !draft.pastedText.trim()) {
    return "Paste user IDs and resume links for this company.";
  }

  return null;
};

const createQueuedCompanyEntry = (draft: DraftCompanyEntry): MultiCompanyEntry => ({
  id: createMultiCompanyEntryId(),
  companyName: draft.companyName.trim(),
  userRequirements: draft.userRequirements,
  inputMethod: draft.inputMethod,
  csvFile: draft.csvFile,
  pastedText: draft.pastedText,
  runStatus: "pending",
  statusMessage: "Ready to start",
});

const getCompanyInputSummary = (entry: DraftCompanyEntry): string => {
  if (entry.inputMethod === "csv") {
    return entry.csvFile?.name ? `CSV: ${entry.csvFile.name}` : "CSV upload";
  }

  return `${countPastedResumeLines(entry.pastedText)} resume line(s)`;
};

const getBatchStatusTone = (status: MultiCompanyRunStatus): "success" | "warning" | "error" | "info" => {
  if (status === "completed") return "success";
  if (status === "failed") return "error";
  if (status === "running" || status === "starting") return "info";
  return "warning";
};

const createBatchDownloadFileName = (companyName: string, originalFileName: string, index: number): string => {
  const dotIndex = originalFileName.lastIndexOf(".");
  const baseName = dotIndex >= 0 ? originalFileName.slice(0, dotIndex) : originalFileName;
  const extension = dotIndex >= 0 ? originalFileName.slice(dotIndex) : ".csv";
  const orderPrefix = String(index + 1).padStart(2, "0");
  const companySegment = sanitizeFileSegment(companyName, `company_${index + 1}`);
  const originalSegment = sanitizeFileSegment(baseName, "results");

  return `${orderPrefix}_${companySegment}_${originalSegment}${extension || ".csv"}`;
};

const extractErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.error ?? error.message;
  }

  return error instanceof Error ? error.message : String(error);
};

const App = () => {
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);
  const [desktopUpdateStatus, setDesktopUpdateStatus] = useState<DesktopUpdateStatus | null>(null);
  const [isInstallingUpdate, setIsInstallingUpdate] = useState(false);
  const [provider, setProvider] = useState<Provider>("OpenAI");
  const [concurrency, setConcurrency] = useState<number>(12);
  const [enableOcr, setEnableOcr] = useState<boolean>(true);

  const [workflowMode, setWorkflowMode] = useState<WorkflowMode>("single");
  const [multiCompanies, setMultiCompanies] = useState<MultiCompanyEntry[]>([]);
  const [batchRunState, setBatchRunState] = useState<BatchRunState>(initialBatchRunState);

  const [inputMethod, setInputMethod] = useState<InputMethod>("text");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [pastedText, setPastedText] = useState("");

  const [userRequirements, setUserRequirements] = useState("");
  const [companyName, setCompanyName] = useState("");
  const [analysisType, setAnalysisType] = useState<AnalysisType>("All Data");
  const [shortlistingMode, setShortlistingMode] = useState<ShortlistingMode>("Priority Wise (P1 / P2 / P3 Bands)");

  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [jobId, setJobId] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>("");
  const [info, setInfo] = useState<string>("");
  const [localJobStartMs, setLocalJobStartMs] = useState<number | null>(null);
  const [timerNow, setTimerNow] = useState<number>(Date.now());

  const [filters, setFilters] = useState<FilterState>(defaultFilters);
  const runTokenRef = useRef(0);

  const currentDraft = useMemo<DraftCompanyEntry>(
    () => ({
      companyName,
      userRequirements,
      inputMethod,
      csvFile,
      pastedText,
    }),
    [companyName, csvFile, inputMethod, pastedText, userRequirements],
  );

  const currentDraftHasContent = useMemo(() => getDraftHasContent(currentDraft), [currentDraft]);
  const currentDraftValidationError = useMemo(() => getDraftValidationError(currentDraft), [currentDraft]);
  const isBatchProcessing = batchRunState.phase === "running" || batchRunState.phase === "waiting";
  const formIsShortlistingActive = userRequirements.trim().length > 0;
  const queueHasShortlisting = useMemo(
    () => multiCompanies.some((entry) => entry.userRequirements.trim().length > 0),
    [multiCompanies],
  );
  const shouldShowShortlistingMode = workflowMode === "multi" ? queueHasShortlisting || formIsShortlistingActive : formIsShortlistingActive;

  useEffect(() => {
    void (async () => {
      try {
        const config = await getConfig();
        setServerConfig(config);
      } catch (loadError) {
        setError(`Failed to load server configuration: ${loadError instanceof Error ? loadError.message : String(loadError)}`);
      }
    })();
  }, []);

  useEffect(() => {
    let isMounted = true;

    const loadDesktopUpdateStatus = async () => {
      try {
        const status = await getDesktopUpdateStatus();
        if (isMounted) {
          setDesktopUpdateStatus(status);
        }
      } catch {
        if (isMounted) {
          setDesktopUpdateStatus(null);
        }
      }
    };

    void loadDesktopUpdateStatus();
    const intervalId = window.setInterval(() => {
      void loadDesktopUpdateStatus();
    }, 5000);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!serverConfig) return;
    const recommended = getRecommendedConcurrency(provider, serverConfig);
    setConcurrency(recommended);
  }, [provider, serverConfig]);

  useEffect(() => {
    const maxAllowed = getMaxConcurrency(provider, serverConfig);
    setConcurrency((prev) => Math.min(prev, maxAllowed));
  }, [provider, serverConfig]);

  useEffect(() => {
    if (!job) return;

    const startedAtMs = job.startedAt ? Date.parse(job.startedAt) : NaN;
    const startMs = Number.isNaN(startedAtMs) ? localJobStartMs : startedAtMs;
    if (!startMs) return;

    const isTerminal = job.status === "completed" || job.status === "failed";
    if (isTerminal) {
      const finishedAtMs = job.finishedAt ? Date.parse(job.finishedAt) : NaN;
      setTimerNow(Number.isNaN(finishedAtMs) ? Date.now() : finishedAtMs);
      return;
    }

    setTimerNow(Date.now());
    const timerId = window.setInterval(() => {
      setTimerNow(Date.now());
    }, 1000);

    return () => {
      window.clearInterval(timerId);
    };
  }, [job, localJobStartMs]);

  useEffect(() => {
    return () => {
      runTokenRef.current += 1;
    };
  }, []);

  const activeJobPayload = job?.payload;
  const displayShortlistingMode = activeJobPayload?.shortlistingMode ?? shortlistingMode;
  const modeForDisplay = activeJobPayload
    ? activeJobPayload.userRequirements.trim().length > 0
      ? "shortlisting"
      : activeJobPayload.analysisType
    : formIsShortlistingActive
      ? "shortlisting"
      : analysisType;

  const elapsedMs = useMemo(() => {
    if (!job) return 0;

    const startedAtMs = job.startedAt ? Date.parse(job.startedAt) : NaN;
    const startMs = Number.isNaN(startedAtMs) ? localJobStartMs : startedAtMs;
    if (!startMs) return 0;

    const isTerminal = job.status === "completed" || job.status === "failed";
    const finishedAtMs = isTerminal && job.finishedAt ? Date.parse(job.finishedAt) : NaN;
    const endMs = Number.isNaN(finishedAtMs) ? timerNow : finishedAtMs;

    return Math.max(0, endMs - startMs);
  }, [job, localJobStartMs, timerNow]);

  const elapsedLabel = useMemo(() => formatElapsedDuration(elapsedMs), [elapsedMs]);

  const sortedOutput = useMemo(() => {
    if (!job?.results) {
      return { rows: [] as TableRow[], columns: [] as string[] };
    }
    return sortAndOrderRows(job.results as TableRow[], modeForDisplay, displayShortlistingMode);
  }, [displayShortlistingMode, job?.results, modeForDisplay]);

  useEffect(() => {
    if (!sortedOutput.columns.includes("Priority Band")) return;
    if (filters.priorityBands.length > 0) return;

    const hasP1 = sortedOutput.rows.some((row) => String(row["Priority Band"] ?? "") === "P1");
    setFilters((prev) => ({ ...prev, priorityBands: hasP1 ? ["P1"] : [] }));
  }, [filters.priorityBands.length, sortedOutput.columns, sortedOutput.rows]);

  const filteredRows = useMemo(() => applyFilters(sortedOutput.rows, filters), [sortedOutput.rows, filters]);
  const gridColumns = useMemo(() => toGridColumns(sortedOutput.columns), [sortedOutput.columns]);

  const liveRows = useMemo(() => {
    if (!job) return [] as TableRow[];
    return (job.liveResults.length > 0 ? job.liveResults : job.results) as TableRow[];
  }, [job]);

  const liveGridRows = useMemo(
    () =>
      liveRows.map((row, index) => ({
        id: `${index + 1}-${String(row["User ID"] ?? "")}`,
        ...row,
      })),
    [liveRows],
  );

  const filteredGridRows = useMemo(
    () =>
      filteredRows.map((row, index) => ({
        id: `${index + 1}-${String(row["User ID"] ?? "")}`,
        ...row,
      })),
    [filteredRows],
  );

  const providerMissingKey = useMemo(() => {
    if (!serverConfig) return true;
    return provider === "OpenAI" ? !serverConfig.hasOpenAiKey : !serverConfig.hasMistralKeys;
  }, [provider, serverConfig]);

  const parsedLinesLabel = useMemo(() => {
    if (inputMethod !== "text") return "";
    const lines = countPastedResumeLines(pastedText);
    return lines > 0 ? `Parsed ${lines} line(s)` : "";
  }, [inputMethod, pastedText]);

  const ensureRunIsActive = (token: number) => {
    if (token !== runTokenRef.current) {
      throw new Error(RUN_CANCELLED_MESSAGE);
    }
  };

  const isRunCancelled = (runError: unknown): boolean =>
    runError instanceof Error && runError.message === RUN_CANCELLED_MESSAGE;

  const updateQueuedCompany = (entryId: string, patch: Partial<MultiCompanyEntry>) => {
    setMultiCompanies((prev) =>
      prev.map((entry) => (entry.id === entryId ? { ...entry, ...patch } : entry)),
    );
  };

  const clearCompanyDraft = () => {
    setCompanyName("");
    setUserRequirements("");
    setCsvFile(null);
    setPastedText("");
  };

  const beginJob = async (payload: StartJobPayload, token: number) => {
    ensureRunIsActive(token);
    setJob(null);
    setJobId("");
    setLocalJobStartMs(Date.now());
    setTimerNow(Date.now());

    const response = await startJob(payload);
    ensureRunIsActive(token);

    setJobId(response.jobId);
    setInfo(response.message || `Started ${payload.companyName} with ${response.totalRows} resume(s).`);
    setFilters(defaultFilters);

    return response;
  };

  const pollJobUntilTerminal = async (nextJobId: string, token: number): Promise<JobStatusResponse> => {
    while (true) {
      ensureRunIsActive(token);
      const status = await getJobStatus(nextJobId);
      ensureRunIsActive(token);

      setJob(status);
      if (status.status === "completed" || status.status === "failed") {
        return status;
      }

      await sleep(2000);
    }
  };

  const downloadCsvFile = async (downloadUrl: string, downloadFileName: string) => {
    const response = await fetch(downloadUrl);
    if (!response.ok) {
      throw new Error(`Download failed with status ${response.status}`);
    }

    const blob = await response.blob();
    const link = document.createElement("a");
    const href = URL.createObjectURL(blob);

    link.href = href;
    link.download = downloadFileName;
    document.body.appendChild(link);
    link.click();
    link.remove();

    window.setTimeout(() => {
      URL.revokeObjectURL(href);
    }, 1000);
  };

  const waitForNextCompany = async (seconds: number, token: number) => {
    for (let remaining = seconds; remaining > 0; remaining -= 1) {
      ensureRunIsActive(token);
      setBatchRunState((prev) => ({
        ...prev,
        phase: "waiting",
        countdownSeconds: remaining,
      }));
      await sleep(1000);
    }

    setBatchRunState((prev) => ({
      ...prev,
      phase: "running",
      countdownSeconds: 0,
    }));
  };

  const handleSingleStart = async () => {
    setError("");
    setInfo("");
    setIsSubmitting(true);

    const token = ++runTokenRef.current;
    const payload: StartJobPayload = {
      provider,
      concurrency,
      enableOcr,
      companyName,
      inputMethod,
      csvFile,
      pastedText,
      userRequirements,
      analysisType,
      shortlistingMode,
    };

    try {
      const response = await beginJob(payload, token);
      void pollJobUntilTerminal(response.jobId, token).catch((runError) => {
        if (isRunCancelled(runError)) return;
        setError(`Polling failed: ${extractErrorMessage(runError)}`);
      });
    } catch (startError) {
      if (!isRunCancelled(startError)) {
        setLocalJobStartMs(null);
        setError(extractErrorMessage(startError));
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleAddMultiCompany = () => {
    setError("");
    setInfo("");

    if (!currentDraftHasContent) {
      setError("Enter company data first, then click Done.");
      return;
    }

    if (currentDraftValidationError) {
      setError(currentDraftValidationError);
      return;
    }

    const nextEntry = createQueuedCompanyEntry(currentDraft);
    setMultiCompanies((prev) => [...prev, nextEntry]);
    clearCompanyDraft();
    setInfo(`Added ${nextEntry.companyName} to the multiple companies queue.`);
  };

  const handleEditQueuedCompany = (entryId: string) => {
    const entry = multiCompanies.find((item) => item.id === entryId);
    if (!entry) return;

    setError("");
    setInfo(`Loaded ${entry.companyName} back into the form for editing.`);
    setInputMethod(entry.inputMethod);
    setCsvFile(entry.csvFile);
    setPastedText(entry.pastedText);
    setUserRequirements(entry.userRequirements);
    setCompanyName(entry.companyName);
    setMultiCompanies((prev) => prev.filter((item) => item.id !== entryId));
  };

  const handleRemoveQueuedCompany = (entryId: string) => {
    const entry = multiCompanies.find((item) => item.id === entryId);
    setMultiCompanies((prev) => prev.filter((item) => item.id !== entryId));
    if (entry) {
      setInfo(`Removed ${entry.companyName} from the queue.`);
    }
  };

  const handleBatchStart = async () => {
    setError("");
    setInfo("");

    let queueForRun = [...multiCompanies];
    if (currentDraftHasContent) {
      if (currentDraftValidationError) {
        setError(`Finish the current company before starting: ${currentDraftValidationError}`);
        return;
      }

      queueForRun = [...queueForRun, createQueuedCompanyEntry(currentDraft)];
      clearCompanyDraft();
    }

    if (!queueForRun.length) {
      setError("Add at least one company before starting multiple companies.");
      return;
    }

    const preparedQueue = queueForRun.map((entry) => ({
      ...entry,
      runStatus: "pending" as const,
      statusMessage: "Queued for processing",
      jobId: undefined,
      resultFileName: undefined,
    }));

    const token = ++runTokenRef.current;
    const sharedSettings = {
      provider,
      concurrency,
      enableOcr,
      analysisType,
      shortlistingMode,
    };

    setJob(null);
    setJobId("");
    setFilters(defaultFilters);
    setMultiCompanies(preparedQueue);
    setBatchRunState({
      phase: "running",
      currentIndex: 0,
      total: preparedQueue.length,
      countdownSeconds: 0,
    });

    for (let index = 0; index < preparedQueue.length; index += 1) {
      const entry = preparedQueue[index];
      const payload: StartJobPayload = {
        ...sharedSettings,
        companyName: entry.companyName,
        inputMethod: entry.inputMethod,
        csvFile: entry.csvFile,
        pastedText: entry.pastedText,
        userRequirements: entry.userRequirements,
      };

      setBatchRunState({
        phase: "running",
        currentIndex: index,
        total: preparedQueue.length,
        countdownSeconds: 0,
      });
      updateQueuedCompany(entry.id, {
        runStatus: "starting",
        statusMessage: "Submitting job...",
      });

      try {
        const response = await beginJob(payload, token);
        updateQueuedCompany(entry.id, {
          runStatus: "running",
          statusMessage: `Running ${response.totalRows} resume(s)...`,
          jobId: response.jobId,
        });

        const finalStatus = await pollJobUntilTerminal(response.jobId, token);
        if (finalStatus.status === "completed") {
          const downloadFileName = createBatchDownloadFileName(entry.companyName, finalStatus.fileName || "results.csv", index);
          updateQueuedCompany(entry.id, {
            runStatus: "completed",
            statusMessage: "Completed. Downloading file...",
            resultFileName: downloadFileName,
          });

          try {
            await downloadCsvFile(buildDownloadUrl(response.jobId, {}), downloadFileName);
            updateQueuedCompany(entry.id, {
              runStatus: "completed",
              statusMessage: `Completed. Downloaded ${downloadFileName}`,
              resultFileName: downloadFileName,
            });
          } catch (downloadError) {
            const message = extractErrorMessage(downloadError);
            updateQueuedCompany(entry.id, {
              runStatus: "completed",
              statusMessage: `Completed. Auto-download failed: ${message}`,
              resultFileName: downloadFileName,
            });
            setError(`Auto-download failed for ${entry.companyName}: ${message}`);
          }
        } else {
          updateQueuedCompany(entry.id, {
            runStatus: "failed",
            statusMessage: finalStatus.errors[0] ?? "Job failed.",
          });
        }
      } catch (runError) {
        if (isRunCancelled(runError)) {
          return;
        }

        updateQueuedCompany(entry.id, {
          runStatus: "failed",
          statusMessage: extractErrorMessage(runError),
        });
      }

      if (index < preparedQueue.length - 1) {
        await waitForNextCompany(30, token);
      }
    }

    setBatchRunState({
      phase: "completed",
      currentIndex: preparedQueue.length - 1,
      total: preparedQueue.length,
      countdownSeconds: 0,
    });
    setInfo("Multiple companies processing finished. Completed files were downloaded automatically.");
  };

  const handleStopBatch = () => {
    if (!isBatchProcessing) return;

    const activeIndex = batchRunState.currentIndex;
    const wasWaiting = batchRunState.phase === "waiting";
    const activeEntry = activeIndex >= 0 ? multiCompanies[activeIndex] : null;

    runTokenRef.current += 1;
    setBatchRunState((prev) => ({
      ...prev,
      phase: "stopped",
      countdownSeconds: 0,
    }));
    setIsSubmitting(false);
    setError("");

    setMultiCompanies((prev) =>
      prev.map((entry, index) => {
        if (entry.runStatus === "completed" || entry.runStatus === "failed") {
          return entry;
        }

        if (index < activeIndex) {
          return entry;
        }

        if (wasWaiting && index === activeIndex) {
          return {
            ...entry,
            runStatus: "completed",
            statusMessage: entry.statusMessage,
          };
        }

        if (!wasWaiting && index === activeIndex) {
          return {
            ...entry,
            runStatus: "stopped",
            statusMessage: "Stopped. This company may still continue on the server if it was already submitted.",
          };
        }

        return {
          ...entry,
          runStatus: "stopped",
          statusMessage: "Stopped before start.",
        };
      }),
    );

    setInfo(
      wasWaiting
        ? "Multiple companies queue stopped. No further companies will start."
        : activeEntry
          ? `Stop requested. ${activeEntry.companyName} will not continue in the batch. If it was already submitted, it may still finish on the server.`
          : "Multiple companies queue stopped.",
    );
  };

  const handleStart = async () => {
    if (workflowMode === "multi") {
      await handleBatchStart();
      return;
    }

    await handleSingleStart();
  };

  const handleDownloadFiltered = async () => {
    if (!jobId) return;

    const query = toQueryParamsFromFilters(filters);
    const url = buildDownloadUrl(jobId, query);
    await downloadCsvFile(url, `filtered_${job?.fileName ?? "results.csv"}`);
  };

  const handleDownloadLiveFull = async () => {
    if (!jobId) return;

    const url = buildDownloadUrl(jobId, {});
    await downloadCsvFile(url, job?.fileName ?? "results.csv");
  };

  const handleInstallDesktopUpdate = async () => {
    if (!desktopUpdateStatus?.availableVersion || isInstallingUpdate) return;

    setIsInstallingUpdate(true);
    setDesktopUpdateStatus((current) =>
      current
        ? {
            ...current,
            phase: "installing",
            message: `Restarting to install version ${current.availableVersion}.`,
          }
        : current,
    );

    try {
      await requestDesktopUpdateInstall(desktopUpdateStatus.availableVersion);
    } catch (installError) {
      setIsInstallingUpdate(false);
      setDesktopUpdateStatus((current) =>
        current
          ? {
              ...current,
              phase: "downloaded",
              message: extractErrorMessage(installError),
            }
          : current,
      );
    }
  };

  const singleCanStart =
    Boolean(companyName.trim()) &&
    !providerMissingKey &&
    (inputMethod === "csv" ? Boolean(csvFile) : Boolean(pastedText.trim()));
  const multiCanStart =
    !providerMissingKey &&
    (multiCompanies.length > 0 || (currentDraftHasContent && !currentDraftValidationError));

  const canStart = workflowMode === "multi" ? multiCanStart : singleCanStart;
  const maxConcurrency = getMaxConcurrency(provider, serverConfig);
  const recommendedConcurrency = getRecommendedConcurrency(provider, serverConfig);

  const batchStatusAlert = useMemo(() => {
    if (workflowMode !== "multi") return null;
    if (batchRunState.phase === "running" && batchRunState.currentIndex >= 0) {
      const activeEntry = multiCompanies[batchRunState.currentIndex];
      return activeEntry ? `Running company ${batchRunState.currentIndex + 1} of ${batchRunState.total}: ${activeEntry.companyName}` : null;
    }

    if (batchRunState.phase === "waiting" && batchRunState.currentIndex >= 0) {
      const nextEntry = multiCompanies[batchRunState.currentIndex + 1];
      return nextEntry
        ? `Waiting ${batchRunState.countdownSeconds}s before starting company ${batchRunState.currentIndex + 2} of ${batchRunState.total}: ${nextEntry.companyName}`
        : null;
    }

    if (batchRunState.phase === "completed" && batchRunState.total > 0) {
      return `Multiple companies batch finished for ${batchRunState.total} company(s).`;
    }

    if (batchRunState.phase === "stopped" && batchRunState.total > 0) {
      return `Multiple companies batch stopped at company ${Math.max(batchRunState.currentIndex + 1, 1)} of ${batchRunState.total}.`;
    }

    return null;
  }, [batchRunState, multiCompanies, workflowMode]);

  const shouldShowDesktopUpdateCard = useMemo(() => {
    if (!desktopUpdateStatus?.isDesktopApp) return false;
    return ["available", "downloading", "downloaded", "installing"].includes(desktopUpdateStatus.phase);
  }, [desktopUpdateStatus]);

  const desktopUpdateButtonLabel = useMemo(() => {
    if (!desktopUpdateStatus) return "";
    if (desktopUpdateStatus.phase === "downloaded") return "Update App";
    if (desktopUpdateStatus.phase === "installing") return "Restarting...";
    if (desktopUpdateStatus.phase === "downloading") {
      const percent = Math.round(Number(desktopUpdateStatus.progressPercent ?? 0));
      return percent > 0 ? `Downloading ${percent}%` : "Downloading Update...";
    }
    return "Preparing Update...";
  }, [desktopUpdateStatus]);

  const desktopUpdateMessage = useMemo(() => {
    if (!desktopUpdateStatus) return "";
    if (desktopUpdateStatus.phase === "downloaded") {
      return `Version ${desktopUpdateStatus.availableVersion ?? "latest"} is ready to install.`;
    }
    if (desktopUpdateStatus.phase === "downloading" || desktopUpdateStatus.phase === "available") {
      return `Version ${desktopUpdateStatus.availableVersion ?? "latest"} is downloading in the background.`;
    }
    if (desktopUpdateStatus.phase === "installing") {
      return `Applying version ${desktopUpdateStatus.availableVersion ?? "latest"} now.`;
    }
    return desktopUpdateStatus.message ?? "";
  }, [desktopUpdateStatus]);

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "#f4f5f7" }}>
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
            p: 2,
            bgcolor: "#0f172a",
            color: "#e2e8f0",
          },
        }}
      >
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 700 }}>
          Configuration
        </Typography>

        <Stack spacing={2}>
          <FormControl fullWidth>
            <InputLabel id="provider-select" sx={{ color: "#cbd5e1" }}>
              AI Provider
            </InputLabel>
            <Select
              labelId="provider-select"
              label="AI Provider"
              value={provider}
              disabled={isBatchProcessing}
              onChange={(event) => setProvider(event.target.value as Provider)}
              sx={{ color: "#e2e8f0" }}
            >
              <MenuItem value="OpenAI">OpenAI</MenuItem>
              <MenuItem value="Mistral">Mistral</MenuItem>
            </Select>
          </FormControl>

          <Box>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Concurrency: {concurrency}
            </Typography>
            <Slider
              min={1}
              max={maxConcurrency}
              step={1}
              value={concurrency}
              disabled={isBatchProcessing}
              onChange={(_, value) => setConcurrency(Array.isArray(value) ? value[0] : value)}
              valueLabelDisplay="auto"
            />
            <Typography variant="caption" sx={{ color: "#94a3b8" }}>
              {provider === "OpenAI"
                ? `Loaded ${serverConfig?.openAiKeyCount ?? 0} OpenAI key(s). Recommended concurrency is ${recommendedConcurrency} for stable runs.`
                : `Loaded ${serverConfig?.mistralKeyCount ?? 0} Mistral key(s). Recommended concurrency follows loaded key count.`}
            </Typography>
          </Box>

          <FormControlLabel
            control={
              <Checkbox
                checked={enableOcr}
                disabled={isBatchProcessing}
                onChange={(event) => setEnableOcr(event.target.checked)}
                sx={{ color: "#e2e8f0" }}
              />
            }
            label="Enable OCR for PDFs & Images"
          />

          {serverConfig && !serverConfig.ocrAvailable && enableOcr ? (
            <Alert severity="warning">OCR dependency for PDF page rendering is unavailable. OCR fallback is best-effort.</Alert>
          ) : null}

          {shouldShowDesktopUpdateCard ? (
            <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: "rgba(15, 23, 42, 0.35)", border: "1px solid rgba(56, 189, 248, 0.28)" }}>
              <Typography variant="subtitle2" sx={{ mb: 0.75, color: "#bae6fd" }}>
                App Update
              </Typography>
              <Typography variant="body2" sx={{ mb: 1.25, color: "#e2e8f0", lineHeight: 1.5 }}>
                {desktopUpdateMessage}
              </Typography>
              <Button
                variant="contained"
                fullWidth
                onClick={() => {
                  void handleInstallDesktopUpdate();
                }}
                disabled={desktopUpdateStatus?.phase !== "downloaded" || isInstallingUpdate}
              >
                {desktopUpdateButtonLabel}
              </Button>
            </Box>
          ) : null}

          <Box sx={{ pt: 1, borderTop: "1px solid rgba(148, 163, 184, 0.25)" }}>
            <Typography variant="subtitle2" sx={{ mb: 1.5, color: "#cbd5e1" }}>
              Workflow
            </Typography>
            <Stack spacing={1.25}>
              <Button
                variant={workflowMode === "single" ? "contained" : "outlined"}
                disabled={isBatchProcessing}
                onClick={() => setWorkflowMode("single")}
                sx={{
                  color: workflowMode === "single" ? undefined : "#e2e8f0",
                  borderColor: workflowMode === "single" ? undefined : "#475569",
                }}
              >
                Single Company
              </Button>
              <Button
                variant={workflowMode === "multi" ? "contained" : "outlined"}
                disabled={isBatchProcessing}
                onClick={() => setWorkflowMode("multi")}
                sx={{
                  color: workflowMode === "multi" ? undefined : "#e2e8f0",
                  borderColor: workflowMode === "multi" ? undefined : "#475569",
                }}
              >
                Multiple Companies
              </Button>
              <Button
                variant="contained"
                color="success"
                disabled={workflowMode !== "multi" || isBatchProcessing || !currentDraftHasContent}
                onClick={handleAddMultiCompany}
              >
                Done
              </Button>
              <Typography variant="caption" sx={{ color: "#94a3b8" }}>
                {workflowMode === "multi"
                  ? `Saved companies: ${multiCompanies.length}. Click Done to add the current company to the queue.`
                  : "Switch to Multiple Companies to save companies one by one and process them sequentially."}
              </Typography>
            </Stack>
          </Box>
        </Stack>
      </Drawer>

      <Box component="main" sx={{ flexGrow: 1, minWidth: 0, py: 3, px: 3, overflowX: "hidden" }}>
        <Box
          sx={{
            transform: `scale(${rightPanelScale})`,
            transformOrigin: "top left",
            width: `calc(100% / ${rightPanelScale})`,
          }}
        >
          {error ? (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          ) : null}
          {info ? (
            <Alert severity="success" sx={{ mb: 2 }}>
              {info}
            </Alert>
          ) : null}
          {batchStatusAlert ? (
            <Alert severity={batchRunState.phase === "completed" ? "success" : batchRunState.phase === "stopped" ? "warning" : "info"} sx={{ mb: 2 }}>
              {batchStatusAlert}
            </Alert>
          ) : null}
          {job?.warnings?.map((warning) => (
            <Alert severity="warning" sx={{ mb: 2 }} key={warning}>
              {warning}
            </Alert>
          ))}

          {workflowMode === "multi" ? (
            <Paper sx={{ p: 2, mb: 2 }}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
                <Typography variant="h6">Multiple Companies Queue</Typography>
                <Typography variant="body2" sx={{ color: "text.secondary" }}>
                  {multiCompanies.length} saved company(s)
                </Typography>
              </Stack>

              <Alert severity="info" sx={{ mb: 2 }}>
                Step 3 and Step 4 settings below are shared across this queue. Each company keeps its own Company Name, JD, and resume input.
              </Alert>

              {currentDraftHasContent ? (
                <Alert severity={currentDraftValidationError ? "warning" : "info"} sx={{ mb: 2 }}>
                  {currentDraftValidationError
                    ? `Current form is not yet ready for the queue: ${currentDraftValidationError}`
                    : "Current company is ready but not yet saved. Click Done in the sidebar to add it to the queue, or Start to auto-include it."}
                </Alert>
              ) : null}

              {multiCompanies.length === 0 ? (
                <Typography variant="body2" sx={{ color: "text.secondary" }}>
                  No saved companies yet. Fill Step 1 and Step 2, then click Done in the sidebar to add each company.
                </Typography>
              ) : (
                <Stack spacing={1.5}>
                  {multiCompanies.map((entry, index) => (
                    <Paper variant="outlined" sx={{ p: 1.5 }} key={entry.id}>
                      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={2}>
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                            {index + 1}. {entry.companyName}
                          </Typography>
                          <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.5 }}>
                            {getCompanyInputSummary(entry)}
                          </Typography>
                          <Typography variant="body2" sx={{ color: "text.secondary", mt: 0.5 }}>
                            {entry.userRequirements.trim() ? "JD added for shortlisting" : "No JD. This company will use comprehensive extraction only."}
                          </Typography>
                          <Alert severity={getBatchStatusTone(entry.runStatus)} sx={{ mt: 1.25 }}>
                            {entry.statusMessage}
                          </Alert>
                          {entry.resultFileName ? (
                            <Typography variant="caption" sx={{ display: "block", mt: 1, color: "text.secondary" }}>
                              Download file: {entry.resultFileName}
                            </Typography>
                          ) : null}
                        </Box>

                        <Stack direction="row" spacing={1}>
                          <Button
                            variant="outlined"
                            size="small"
                            disabled={isBatchProcessing}
                            onClick={() => handleEditQueuedCompany(entry.id)}
                          >
                            Edit
                          </Button>
                          <Button
                            variant="outlined"
                            color="error"
                            size="small"
                            disabled={isBatchProcessing}
                            onClick={() => handleRemoveQueuedCompany(entry.id)}
                          >
                            Remove
                          </Button>
                        </Stack>
                      </Stack>
                    </Paper>
                  ))}
                </Stack>
              )}
            </Paper>
          ) : null}

          <Paper sx={{ p: 2, mb: 2 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
              <Typography variant="h6">
                Step 1: Provide Resume Data{workflowMode === "multi" ? " for the Next Company" : ""}
              </Typography>
              {parsedLinesLabel ? (
                <Typography variant="body2" sx={{ color: "success.main", fontWeight: 600 }}>
                  {parsedLinesLabel}
                </Typography>
              ) : null}
            </Stack>

            {workflowMode === "multi" ? (
              <Alert severity="info" sx={{ mb: 2 }}>
                Use this form for one company at a time. Click Done in the sidebar to save it, then enter the next company.
              </Alert>
            ) : null}

            <FormControl>
              <FormLabel>Choose input method:</FormLabel>
              <RadioGroup row value={inputMethod} onChange={(event) => setInputMethod(event.target.value as InputMethod)}>
                <FormControlLabel value="csv" control={<Radio />} label="Upload CSV" />
                <FormControlLabel value="text" control={<Radio />} label="Paste Text" />
              </RadioGroup>
            </FormControl>

            {inputMethod === "csv" ? (
              <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
                <Button component="label" variant="outlined">
                  Upload CSV with user_id and Resume link
                  <input
                    hidden
                    type="file"
                    accept=".csv"
                    onChange={(event) => {
                      const file = event.target.files?.[0] ?? null;
                      setCsvFile(file);
                    }}
                  />
                </Button>
                <Typography variant="body2">{csvFile?.name ?? "No file selected"}</Typography>
              </Stack>
            ) : (
              <TextField
                multiline
                minRows={6}
                maxRows={6}
                fullWidth
                sx={{
                  mt: 1,
                  "& .MuiInputBase-inputMultiline": {
                    overflowY: "auto",
                  },
                }}
                placeholder={"user1\thttp://example.com/resume.pdf\nuser2\thttps://example.com/resume.png"}
                value={pastedText}
                onChange={(event) => setPastedText(event.target.value)}
              />
            )}
          </Paper>

          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid size={{ xs: 12, md: 6 }}>
              <Paper sx={{ p: 2, height: "100%" }}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Step 2: Priority-Based Shortlisting{workflowMode === "multi" ? " per Company" : ""}
                </Typography>
                <TextField
                  multiline
                  minRows={8}
                  fullWidth
                  label="Enter Job Description or Requirements for Shortlisting"
                  value={userRequirements}
                  onChange={(event) => setUserRequirements(event.target.value)}
                  placeholder="e.g., Seeking a senior Python developer with 5+ years of experience"
                />

                <TextField
                  sx={{ mt: 2 }}
                  fullWidth
                  label="Enter Company Name (Required)"
                  value={companyName}
                  onChange={(event) => setCompanyName(event.target.value)}
                  placeholder="e.g., Acme Corporation"
                />

                <Alert severity="info" sx={{ mt: 2 }}>
                  {formIsShortlistingActive
                    ? "Mode: Priority-Based Shortlisting (Focused Analysis)"
                    : "Mode: Comprehensive Extraction"}
                </Alert>
              </Paper>
            </Grid>

            <Grid size={{ xs: 12, md: 6 }}>
              <Paper sx={{ p: 2, height: "100%" }}>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Step 3: Comprehensive Data Extraction
                </Typography>

                <FormControl fullWidth>
                  <InputLabel id="analysis-select">Choose data to extract</InputLabel>
                  <Select
                    labelId="analysis-select"
                    label="Choose data to extract"
                    value={analysisType}
                    disabled={isBatchProcessing}
                    onChange={(event) => setAnalysisType(event.target.value as AnalysisType)}
                  >
                    {analysisOptions.map((option) => (
                      <MenuItem value={option} key={option}>
                        {option}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>
                  Step 4: Start Analysis
                </Typography>

                {shouldShowShortlistingMode ? (
                  <FormControl fullWidth>
                    <InputLabel id="shortlist-mode-select">Choose Shortlisting Mode</InputLabel>
                    <Select
                      labelId="shortlist-mode-select"
                      value={shortlistingMode}
                      label="Choose Shortlisting Mode"
                      disabled={isBatchProcessing}
                      onChange={(event) => setShortlistingMode(event.target.value as ShortlistingMode)}
                    >
                      {shortlistingOptions.map((option) => (
                        <MenuItem key={option} value={option}>
                          {option}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                ) : null}

                <Typography variant="body2" sx={{ mt: 2, mb: 2 }}>
                  Provider for this run: {provider}
                </Typography>

                {workflowMode === "multi" ? (
                  <Stack spacing={1.5}>
                    <Button
                      variant="contained"
                      size="large"
                      fullWidth
                      disabled={!canStart || isSubmitting || isBatchProcessing}
                      onClick={() => {
                        void handleStart();
                      }}
                    >
                      {isSubmitting ? (
                        <Stack direction="row" spacing={1} alignItems="center">
                          <CircularProgress size={18} color="inherit" />
                          <span>Starting...</span>
                        </Stack>
                      ) : (
                        "Start Multiple Companies"
                      )}
                    </Button>

                    {isBatchProcessing ? (
                      <Button variant="outlined" color="error" size="large" fullWidth onClick={handleStopBatch}>
                        Stop Multiple Companies
                      </Button>
                    ) : null}
                  </Stack>
                ) : (
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    disabled={!canStart || isSubmitting || isBatchProcessing}
                    onClick={() => {
                      void handleStart();
                    }}
                  >
                    {isSubmitting ? (
                      <Stack direction="row" spacing={1} alignItems="center">
                        <CircularProgress size={18} color="inherit" />
                        <span>Starting...</span>
                      </Stack>
                    ) : formIsShortlistingActive ? (
                      "Start Shortlisting"
                    ) : (
                      `Start '${analysisType}' Extraction`
                    )}
                  </Button>
                )}

                {workflowMode === "multi" ? (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Completed company files will download automatically. The next company starts after a 30-second gap.
                  </Alert>
                ) : null}

                {!companyName.trim() && workflowMode === "single" ? (
                  <Alert severity="warning" sx={{ mt: 2 }}>
                    Company Name is a required field.
                  </Alert>
                ) : null}
              </Paper>
            </Grid>
          </Grid>

          {job ? (
            <Paper sx={{ p: 2, mb: 2 }}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="h6">
                  Live Progress{workflowMode === "multi" && job.payload.companyName ? `: ${job.payload.companyName}` : ""}
                </Typography>
                <Stack direction="row" spacing={1.5} alignItems="center">
                  <Typography variant="body2" sx={{ color: "text.secondary", fontVariantNumeric: "tabular-nums" }}>
                    Elapsed: {elapsedLabel}
                  </Typography>
                  {job.status === "completed" ? (
                    <Button variant="outlined" onClick={() => void handleDownloadLiveFull()}>
                      Download Full Live Data
                    </Button>
                  ) : null}
                </Stack>
              </Stack>
              <Typography variant="body1" sx={{ mb: 1 }}>
                Processing... {job.completed}/{job.total} resumes completed.
              </Typography>
              <LinearProgress variant="determinate" value={(job.progress || 0) * 100} sx={{ mb: 2 }} />

              <Box sx={{ height: 420 }}>
                <DataGrid
                  rows={liveGridRows}
                  columns={gridColumns}
                  pageSizeOptions={[100]}
                  initialState={{ pagination: { paginationModel: { pageSize: 100, page: 0 } } }}
                  disableRowSelectionOnClick
                  rowHeight={44}
                  sx={{
                    "& .MuiDataGrid-cell": {
                      alignItems: "center",
                      overflow: "hidden",
                    },
                    "& .MuiDataGrid-columnHeaderTitle": {
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    },
                  }}
                />
              </Box>
            </Paper>
          ) : null}
          {job?.status === "completed" ? (
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Step 5: Filter, Review & Download Results
              </Typography>

              {workflowMode === "multi" ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Step 5 shows the most recently finished company. Batch mode already auto-downloads each completed company file.
                </Alert>
              ) : null}

              <Accordion>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>Show Interactive Filters</AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    <Grid size={{ xs: 12, md: 3 }}>
                      {sortedOutput.columns.includes("Priority Band") ? (
                        <FormControl fullWidth>
                          <InputLabel id="priority-band-filter">Filter by Priority Band</InputLabel>
                          <Select
                            labelId="priority-band-filter"
                            label="Filter by Priority Band"
                            multiple
                            value={filters.priorityBands}
                            onChange={(event) => setFilters((prev) => ({ ...prev, priorityBands: event.target.value as string[] }))}
                          >
                            {priorityOrder.map((band) => (
                              <MenuItem key={band} value={band}>
                                {band}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      ) : null}

                      <TextField
                        sx={{ mt: 2 }}
                        fullWidth
                        label="Search text in key fields"
                        value={filters.searchTerm}
                        onChange={(event) => setFilters((prev) => ({ ...prev, searchTerm: event.target.value }))}
                      />
                    </Grid>

                    <Grid size={{ xs: 12, md: 3 }}>
                      {sortedOutput.columns.includes("Overall Probability") ? (
                        <Box>
                          <Typography variant="body2">Overall Probability</Typography>
                          <Slider
                            min={0}
                            max={100}
                            value={filters.overallRange}
                            onChange={(_, value) => setFilters((prev) => ({ ...prev, overallRange: value as [number, number] }))}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      ) : null}

                      {sortedOutput.columns.includes("Skills Probability") ? (
                        <Box>
                          <Typography variant="body2">Skills Probability</Typography>
                          <Slider
                            min={0}
                            max={100}
                            value={filters.skillsRange}
                            onChange={(_, value) => setFilters((prev) => ({ ...prev, skillsRange: value as [number, number] }))}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      ) : null}
                    </Grid>

                    <Grid size={{ xs: 12, md: 3 }}>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        Project Filters
                      </Typography>
                      {sortedOutput.columns.includes("Internal Projects Count") ? (
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={filters.onlyInternal}
                              onChange={(event) => setFilters((prev) => ({ ...prev, onlyInternal: event.target.checked }))}
                            />
                          }
                          label="Show only with Internal Projects"
                        />
                      ) : null}

                      {sortedOutput.columns.includes("External Projects Count") ? (
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={filters.onlyExternal}
                              onChange={(event) => setFilters((prev) => ({ ...prev, onlyExternal: event.target.checked }))}
                            />
                          }
                          label="Show only with External Projects"
                        />
                      ) : null}

                      {sortedOutput.columns.includes("Experience Probability") ? (
                        <Box>
                          <Typography variant="body2">Experience Probability</Typography>
                          <Slider
                            min={0}
                            max={100}
                            value={filters.experienceRange}
                            onChange={(_, value) => setFilters((prev) => ({ ...prev, experienceRange: value as [number, number] }))}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      ) : null}
                    </Grid>

                    <Grid size={{ xs: 12, md: 3 }}>
                      {sortedOutput.columns.includes("Projects Probability") ? (
                        <Box>
                          <Typography variant="body2">Projects Probability</Typography>
                          <Slider
                            min={0}
                            max={100}
                            value={filters.projectsRange}
                            onChange={(_, value) => setFilters((prev) => ({ ...prev, projectsRange: value as [number, number] }))}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      ) : null}

                      {sortedOutput.columns.includes("Other Probability") ? (
                        <Box>
                          <Typography variant="body2">Other Probability</Typography>
                          <Slider
                            min={0}
                            max={100}
                            value={filters.otherRange}
                            onChange={(_, value) => setFilters((prev) => ({ ...prev, otherRange: value as [number, number] }))}
                            valueLabelDisplay="auto"
                          />
                        </Box>
                      ) : null}
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                      {sortedOutput.columns.includes("Total Projects Count") ? (
                        <TextField
                          fullWidth
                          type="number"
                          label="Minimum Total Projects"
                          value={filters.minTotalProjects}
                          onChange={(event) => setFilters((prev) => ({ ...prev, minTotalProjects: Number(event.target.value || 0) }))}
                        />
                      ) : null}
                    </Grid>
                    <Grid size={{ xs: 12, md: 4 }}>
                      {sortedOutput.columns.includes("Internal Projects Count") ? (
                        <TextField
                          fullWidth
                          type="number"
                          label="Minimum Internal Projects"
                          value={filters.minInternalProjects}
                          onChange={(event) => setFilters((prev) => ({ ...prev, minInternalProjects: Number(event.target.value || 0) }))}
                        />
                      ) : null}
                    </Grid>
                    <Grid size={{ xs: 12, md: 4 }}>
                      {sortedOutput.columns.includes("External Projects Count") ? (
                        <TextField
                          fullWidth
                          type="number"
                          label="Minimum External Projects"
                          value={filters.minExternalProjects}
                          onChange={(event) => setFilters((prev) => ({ ...prev, minExternalProjects: Number(event.target.value || 0) }))}
                        />
                      ) : null}
                    </Grid>
                  </Grid>
                </AccordionDetails>
              </Accordion>

              <Alert severity="info" sx={{ mt: 2, mb: 2 }}>
                Displaying {filteredRows.length} of {sortedOutput.rows.length} candidates.
              </Alert>

              <Box sx={{ height: 500, mb: 2 }}>
                <DataGrid
                  rows={filteredGridRows}
                  columns={gridColumns}
                  pageSizeOptions={[10, 25, 50, 100]}
                  initialState={{ pagination: { paginationModel: { pageSize: 25, page: 0 } } }}
                  disableRowSelectionOnClick
                  rowHeight={44}
                  sx={{
                    "& .MuiDataGrid-cell": {
                      alignItems: "center",
                      overflow: "hidden",
                    },
                    "& .MuiDataGrid-columnHeaderTitle": {
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    },
                  }}
                />
              </Box>

              <Button variant="contained" onClick={() => void handleDownloadFiltered()}>
                Download {filteredRows.length} Filtered Results as CSV
              </Button>
            </Paper>
          ) : null}
        </Box>
      </Box>
    </Box>
  );
};

export default App;
