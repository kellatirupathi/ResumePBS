import { useEffect, useMemo, useRef, useState } from "react";
import {
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
  Accordion,
  AccordionDetails,
  AccordionSummary,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import axios from "axios";
import { buildDownloadUrl, getConfig, getJobStatus, startJob } from "./api";
import { applyFilters, defaultFilters, sortAndOrderRows, toQueryParamsFromFilters, type TableRow, priorityOrder } from "./helpers";
import type { AnalysisType, FilterState, JobStatusResponse, Provider, ServerConfig, ShortlistingMode } from "./types";

const drawerWidth = 320;
const rightPanelScale = 0.9;

const analysisOptions: AnalysisType[] = ["All Data", "Personal Details", "Skills & Projects", "Internal Projects Matching"];
const shortlistingOptions: ShortlistingMode[] = ["Probability Wise (Default)", "Priority Wise (P1 / P2 / P3 Bands)", "Sectionwise"];

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

const App = () => {
  const [serverConfig, setServerConfig] = useState<ServerConfig | null>(null);
  const [provider, setProvider] = useState<Provider>("OpenAI");
  const [concurrency, setConcurrency] = useState<number>(12);
  const [enableOcr, setEnableOcr] = useState<boolean>(true);

  const [inputMethod, setInputMethod] = useState<"csv" | "text">("text");
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
  const pollingRef = useRef<number | null>(null);

  const isShortlistingActive = userRequirements.trim().length > 0;
  const modeForDisplay = isShortlistingActive ? "shortlisting" : analysisType;

  useEffect(() => {
    void (async () => {
      try {
        const config = await getConfig();
        setServerConfig(config);
      } catch (e) {
        setError(`Failed to load server configuration: ${e instanceof Error ? e.message : String(e)}`);
      }
    })();
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
    if (!jobId) return;

    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        setJob(status);
        if (status.status === "completed" || status.status === "failed") {
          if (pollingRef.current) {
            window.clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        }
      } catch (e) {
        setError(`Polling failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    };

    void poll();
    pollingRef.current = window.setInterval(() => {
      void poll();
    }, 2000);

    return () => {
      if (pollingRef.current) {
        window.clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [jobId]);

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
    return sortAndOrderRows(job.results as TableRow[], modeForDisplay, shortlistingMode);
  }, [job?.results, modeForDisplay, shortlistingMode]);

  useEffect(() => {
    if (!sortedOutput.columns.includes("Priority Band")) return;
    if (filters.priorityBands.length > 0) return;

    const hasP1 = sortedOutput.rows.some((row) => String(row["Priority Band"] ?? "") === "P1");
    setFilters((prev) => ({ ...prev, priorityBands: hasP1 ? ["P1"] : [] }));
  }, [sortedOutput.columns, sortedOutput.rows, filters.priorityBands.length]);

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
    const lines = pastedText.split(/\r?\n/).filter((line) => line.trim()).length;
    return lines > 0 ? `Parsed ${lines} line(s)` : "";
  }, [inputMethod, pastedText]);

  const handleStart = async () => {
    setError("");
    setInfo("");
    setIsSubmitting(true);
    setJob(null);
    setLocalJobStartMs(Date.now());
    setTimerNow(Date.now());

    try {
      const response = await startJob({
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
      });

      setJobId(response.jobId);
      setInfo(response.message);
      setFilters(defaultFilters);
    } catch (e) {
      setLocalJobStartMs(null);
      if (axios.isAxiosError(e)) {
        setError(e.response?.data?.error ?? e.message);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDownloadFiltered = async () => {
    if (!jobId) return;

    const query = toQueryParamsFromFilters(filters);
    const url = buildDownloadUrl(jobId, query);
    const response = await fetch(url);
    const blob = await response.blob();

    const link = document.createElement("a");
    const href = URL.createObjectURL(blob);
    link.href = href;
    link.download = `filtered_${job?.fileName ?? "results.csv"}`;
    link.click();
    URL.revokeObjectURL(href);
  };

  const handleDownloadLiveFull = async () => {
    if (!jobId) return;

    const url = buildDownloadUrl(jobId, {});
    const response = await fetch(url);
    const blob = await response.blob();

    const link = document.createElement("a");
    const href = URL.createObjectURL(blob);
    link.href = href;
    link.download = job?.fileName ?? "results.csv";
    link.click();
    URL.revokeObjectURL(href);
  };

  const canStart = Boolean(companyName.trim()) && !providerMissingKey && (inputMethod === "csv" ? Boolean(csvFile) : Boolean(pastedText.trim()));
  const maxConcurrency = getMaxConcurrency(provider, serverConfig);
  const recommendedConcurrency = getRecommendedConcurrency(provider, serverConfig);

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
            control={<Checkbox checked={enableOcr} onChange={(event) => setEnableOcr(event.target.checked)} sx={{ color: "#e2e8f0" }} />}
            label="Enable OCR for PDFs & Images"
          />

          {serverConfig && !serverConfig.ocrAvailable && enableOcr ? (
            <Alert severity="warning">OCR dependency for PDF page rendering is unavailable. OCR fallback is best-effort.</Alert>
          ) : null}

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

        {error ? <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert> : null}
        {info ? <Alert severity="success" sx={{ mb: 2 }}>{info}</Alert> : null}
        {job?.warnings?.map((warning) => (
          <Alert severity="warning" sx={{ mb: 2 }} key={warning}>
            {warning}
          </Alert>
        ))}

        <Paper sx={{ p: 2, mb: 2 }}>
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
            <Typography variant="h6">Step 1: Provide Resume Data</Typography>
            {parsedLinesLabel ? (
              <Typography variant="body2" sx={{ color: "success.main", fontWeight: 600 }}>
                {parsedLinesLabel}
              </Typography>
            ) : null}
          </Stack>

          <FormControl>
            <FormLabel>Choose input method:</FormLabel>
            <RadioGroup row value={inputMethod} onChange={(event) => setInputMethod(event.target.value as "csv" | "text")}> 
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
              <Typography variant="h6" sx={{ mb: 2 }}>Step 2: Priority-Based Shortlisting</Typography>
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
                {isShortlistingActive
                  ? "Mode: Priority-Based Shortlisting (Focused Analysis)"
                  : "Mode: Comprehensive Extraction"}
              </Alert>
            </Paper>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }}>
            <Paper sx={{ p: 2, height: "100%" }}>
              <Typography variant="h6" sx={{ mb: 2 }}>Step 3: Comprehensive Data Extraction</Typography>

              <FormControl fullWidth>
                <InputLabel id="analysis-select">Choose data to extract</InputLabel>
                <Select
                  labelId="analysis-select"
                  label="Choose data to extract"
                  value={analysisType}
                  onChange={(event) => setAnalysisType(event.target.value as AnalysisType)}
                >
                  {analysisOptions.map((option) => (
                    <MenuItem value={option} key={option}>{option}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>Step 4: Start Analysis</Typography>

              {isShortlistingActive ? (
                <FormControl fullWidth>
                  <InputLabel id="shortlist-mode-select">Choose Shortlisting Mode</InputLabel>
                  <Select
                    labelId="shortlist-mode-select"
                    value={shortlistingMode}
                    label="Choose Shortlisting Mode"
                    onChange={(event) => setShortlistingMode(event.target.value as ShortlistingMode)}
                  >
                    {shortlistingOptions.map((option) => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              ) : null}

              <Typography variant="body2" sx={{ mt: 2, mb: 2 }}>
                Provider for this run: {provider}
              </Typography>

              <Button
                variant="contained"
                size="large"
                fullWidth
                disabled={!canStart || isSubmitting}
                onClick={handleStart}
              >
                {isSubmitting ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={18} color="inherit" />
                    <span>Starting...</span>
                  </Stack>
                ) : isShortlistingActive ? (
                  "Start Shortlisting"
                ) : (
                  `Start '${analysisType}' Extraction`
                )}
              </Button>

              {!companyName.trim() ? <Alert severity="warning" sx={{ mt: 2 }}>Company Name is a required field.</Alert> : null}
            </Paper>
          </Grid>
        </Grid>

        {job ? (
          <Paper sx={{ p: 2, mb: 2 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
              <Typography variant="h6">Live Progress</Typography>
              <Stack direction="row" spacing={1.5} alignItems="center">
                <Typography variant="body2" sx={{ color: "text.secondary", fontVariantNumeric: "tabular-nums" }}>
                  Elapsed: {elapsedLabel}
                </Typography>
                {job.status === "completed" ? (
                  <Button variant="outlined" onClick={handleDownloadLiveFull}>
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
            <Typography variant="h6" sx={{ mb: 2 }}>Step 5: Filter, Review & Download Results</Typography>

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
                            <MenuItem key={band} value={band}>{band}</MenuItem>
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
                    <Typography variant="body2" sx={{ mb: 1 }}>Project Filters</Typography>
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

            <Button variant="contained" onClick={handleDownloadFiltered}>
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
