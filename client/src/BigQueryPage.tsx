import { useEffect, useMemo, useState } from "react";
import { Alert, Box, Button, Chip, CircularProgress, Paper, Stack, Tab, Tabs, TextField, Typography } from "@mui/material";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { getBigQueryDataset, getBigQueryTableData, getLearningMetricsByUserIds } from "./api";
import type {
  BigQueryDatasetSnapshot,
  BigQueryLearningMetricsResponse,
  BigQueryTableDataResponse,
  BigQueryTableSummary,
} from "./types";

const parseUserIds = (raw: string): string[] => {
  const unique = new Set<string>();
  for (const part of raw.split(/[\r\n,\t ]+/)) {
    const normalized = part.trim();
    if (normalized) unique.add(normalized);
  }
  return Array.from(unique);
};

const toDisplayValue = (value: unknown): string | number | boolean | null => {
  if (value === undefined || value === null) return null;
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const toGridColumns = (columns: string[]): GridColDef[] =>
  columns.map((column) => ({
    field: column,
    headerName: column,
    minWidth: 180,
    flex: 1,
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

const toCsv = (rows: Array<Record<string, unknown>>, columns: string[]): string => {
  const escape = (value: unknown): string => {
    const raw = value === null || value === undefined ? "" : String(value);
    const needsQuote = /[",\n]/.test(raw);
    const escaped = raw.replace(/"/g, '""');
    return needsQuote ? `"${escaped}"` : escaped;
  };

  const lines = [columns.map(escape).join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => escape(row[column])).join(","));
  }
  return lines.join("\n");
};

const BigQueryPage = () => {
  const [activeTab, setActiveTab] = useState(0);

  const [userIdInput, setUserIdInput] = useState("");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");
  const [metricsResult, setMetricsResult] = useState<BigQueryLearningMetricsResponse | null>(null);

  const [tableListLoading, setTableListLoading] = useState(false);
  const [tableListError, setTableListError] = useState("");
  const [tableListSnapshot, setTableListSnapshot] = useState<BigQueryDatasetSnapshot | null>(null);
  const [tableListAttempted, setTableListAttempted] = useState(false);

  const [selectedTableId, setSelectedTableId] = useState("");
  const [selectedTableLoading, setSelectedTableLoading] = useState(false);
  const [selectedTableError, setSelectedTableError] = useState("");
  const [selectedTableData, setSelectedTableData] = useState<BigQueryTableDataResponse | null>(null);
  const [tableDataCache, setTableDataCache] = useState<Record<string, BigQueryTableDataResponse>>({});

  const parsedUserIds = useMemo(() => parseUserIds(userIdInput), [userIdInput]);

  const metricsColumns = useMemo(() => {
    if (!metricsResult || metricsResult.rows.length === 0) return [] as string[];
    return Object.keys(metricsResult.rows[0] ?? {});
  }, [metricsResult]);

  const metricsGridColumns = useMemo(() => toGridColumns(metricsColumns), [metricsColumns]);

  const metricsGridRows = useMemo(
    () =>
      (metricsResult?.rows ?? []).map((row, index) => {
        const formattedRow: Record<string, string | number | boolean | null> = {
          id: `${index + 1}`,
        };
        for (const column of metricsColumns) {
          formattedRow[column] = toDisplayValue(row[column]);
        }
        return formattedRow;
      }),
    [metricsResult, metricsColumns],
  );

  const selectedTableColumns = useMemo(() => {
    if (!selectedTableData) return [] as string[];
    const table = selectedTableData.table;
    if (table.columns.length > 0) return table.columns;
    return Object.keys(table.rows[0] ?? {});
  }, [selectedTableData]);

  const selectedTableGridColumns = useMemo(() => toGridColumns(selectedTableColumns), [selectedTableColumns]);

  const selectedTableGridRows = useMemo(
    () =>
      (selectedTableData?.table.rows ?? []).map((row, index) => {
        const formattedRow: Record<string, string | number | boolean | null> = {
          id: `${index + 1}`,
        };
        for (const column of selectedTableColumns) {
          formattedRow[column] = toDisplayValue(row[column]);
        }
        return formattedRow;
      }),
    [selectedTableData, selectedTableColumns],
  );

  const handleSubmitMetrics = async () => {
    setMetricsError("");
    setMetricsLoading(true);
    setMetricsResult(null);

    try {
      const response = await getLearningMetricsByUserIds(parsedUserIds);
      setMetricsResult(response);
    } catch (err) {
      setMetricsError(err instanceof Error ? err.message : String(err));
    } finally {
      setMetricsLoading(false);
    }
  };

  const handleLoadTableNames = async () => {
    setTableListAttempted(true);
    setTableListError("");
    setTableListLoading(true);

    try {
      const response = await getBigQueryDataset();
      setTableListSnapshot(response);
    } catch (err) {
      setTableListError(err instanceof Error ? err.message : String(err));
    } finally {
      setTableListLoading(false);
    }
  };

  const handleSelectTable = async (table: BigQueryTableSummary) => {
    const tableId = table.tableId;
    setSelectedTableId(tableId);
    setSelectedTableError("");

    const cached = tableDataCache[tableId];
    if (cached) {
      setSelectedTableData(cached);
      return;
    }

    setSelectedTableLoading(true);
    setSelectedTableData(null);

    try {
      const response = await getBigQueryTableData(tableId);
      setSelectedTableData(response);
      setTableDataCache((prev) => ({ ...prev, [tableId]: response }));
    } catch (err) {
      setSelectedTableError(err instanceof Error ? err.message : String(err));
    } finally {
      setSelectedTableLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab !== 1) return;
    if (tableListAttempted || tableListLoading) return;
    void handleLoadTableNames();
  }, [activeTab, tableListAttempted, tableListLoading]);

  const handleDownloadMetricsCsv = () => {
    if (!metricsResult || metricsResult.rows.length === 0 || metricsColumns.length === 0) return;
    const csv = toCsv(metricsResult.rows, metricsColumns);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const href = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = href;
    link.download = `learning_metrics_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "_")}.csv`;
    link.click();
    URL.revokeObjectURL(href);
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#f4f5f7", p: 3 }}>
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
        <Stack spacing={0.5}>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            BigQuery Learning Metrics
          </Typography>
          <Typography variant="body2" sx={{ color: "text.secondary" }}>
            Learning metrics by user IDs and BigQuery table explorer.
          </Typography>
        </Stack>
        <Button variant="outlined" href="/">
          Back To Analyzer
        </Button>
      </Stack>

      <Paper sx={{ p: 1.5, mb: 2 }}>
        <Tabs value={activeTab} onChange={(_event, value) => setActiveTab(value)}>
          <Tab label="Learning Metrics By User IDs" />
          <Tab label="Fetch All Tables" />
        </Tabs>
      </Paper>

      {activeTab === 0 ? (
        <>
          <Paper sx={{ p: 2, mb: 2 }}>
            <TextField
              label="User IDs"
              placeholder={"user_id_1\nuser_id_2\nuser_id_3"}
              multiline
              minRows={6}
              maxRows={10}
              fullWidth
              value={userIdInput}
              onChange={(event) => setUserIdInput(event.target.value)}
            />

            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mt: 2 }}>
              <Typography variant="body2" sx={{ color: "text.secondary" }}>
                Parsed {parsedUserIds.length} user ID(s)
              </Typography>
              <Button
                variant="contained"
                onClick={() => void handleSubmitMetrics()}
                disabled={metricsLoading || parsedUserIds.length === 0}
              >
                {metricsLoading ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={16} color="inherit" />
                    <span>Submitting...</span>
                  </Stack>
                ) : (
                  "Submit"
                )}
              </Button>
            </Stack>
          </Paper>

          {metricsError ? <Alert severity="error" sx={{ mb: 2 }}>{metricsError}</Alert> : null}

          {metricsResult ? (
            <Paper sx={{ p: 2 }}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="h6">Learning Metrics Result</Typography>
                <Button variant="outlined" disabled={metricsResult.rows.length === 0} onClick={handleDownloadMetricsCsv}>
                  Download CSV
                </Button>
              </Stack>

              <Typography variant="body2" sx={{ color: "text.secondary", mb: 2 }}>
                Project: {metricsResult.projectId} | Dataset: {metricsResult.datasetId} | Table: {metricsResult.tableId} |
                {" "}Input IDs: {metricsResult.inputUserCount} | Matched Rows: {metricsResult.matchedRowCount}
              </Typography>

              {metricsGridRows.length > 0 ? (
                <Box sx={{ height: 520 }}>
                  <DataGrid
                    rows={metricsGridRows}
                    columns={metricsGridColumns}
                    pageSizeOptions={[25, 50, 100]}
                    initialState={{ pagination: { paginationModel: { pageSize: 25, page: 0 } } }}
                    disableRowSelectionOnClick
                    rowHeight={42}
                  />
                </Box>
              ) : (
                <Alert severity="info">No matching records found for the submitted user IDs.</Alert>
              )}
            </Paper>
          ) : null}
        </>
      ) : null}

      {activeTab === 1 ? (
        <>
          <Paper sx={{ p: 2, mb: 2 }}>
            <Stack direction="row" alignItems="center" justifyContent="space-between">
              <Typography variant="body2" sx={{ color: "text.secondary" }}>
                First load table names, then click a table name to fetch that table data.
              </Typography>
              <Button variant="contained" onClick={() => void handleLoadTableNames()} disabled={tableListLoading}>
                {tableListLoading ? (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <CircularProgress size={16} color="inherit" />
                    <span>Loading Names...</span>
                  </Stack>
                ) : (
                  "Load Table Names"
                )}
              </Button>
            </Stack>
          </Paper>

          {tableListError ? <Alert severity="error" sx={{ mb: 2 }}>{tableListError}</Alert> : null}

          {tableListSnapshot ? (
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Tables ({tableListSnapshot.tableCount})
              </Typography>
              <Typography variant="body2" sx={{ color: "text.secondary", mb: 2 }}>
                Project: {tableListSnapshot.projectId} | Dataset: {tableListSnapshot.datasetId}
              </Typography>

              {tableListSnapshot.tables.length > 0 ? (
                <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                  {tableListSnapshot.tables.map((table) => (
                    <Button
                      key={table.tableId}
                      variant={selectedTableId === table.tableId ? "contained" : "outlined"}
                      onClick={() => void handleSelectTable(table)}
                    >
                      {table.tableId}
                    </Button>
                  ))}
                </Stack>
              ) : (
                <Alert severity="info">No tables found.</Alert>
              )}
            </Paper>
          ) : null}

          {selectedTableLoading ? (
            <Paper sx={{ p: 2, mb: 2 }}>
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={18} />
                <Typography>Loading table data for {selectedTableId}...</Typography>
              </Stack>
            </Paper>
          ) : null}

          {selectedTableError ? <Alert severity="error" sx={{ mb: 2 }}>{selectedTableError}</Alert> : null}

          {selectedTableData ? (
            <Paper sx={{ p: 2 }}>
              <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="h6">Table: {selectedTableData.table.tableId}</Typography>
                <Chip label={selectedTableData.table.tableType} color="default" />
              </Stack>

              <Typography variant="body2" sx={{ color: "text.secondary", mb: 2 }}>
                Project: {selectedTableData.projectId} | Dataset: {selectedTableData.datasetId} | Rows: {selectedTableData.table.rowCount}
              </Typography>

              {selectedTableData.table.fetchError ? (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  {selectedTableData.table.fetchError}
                </Alert>
              ) : null}

              {selectedTableGridRows.length > 0 ? (
                <Box sx={{ height: 520 }}>
                  <DataGrid
                    rows={selectedTableGridRows}
                    columns={selectedTableGridColumns}
                    pageSizeOptions={[25, 50, 100]}
                    initialState={{ pagination: { paginationModel: { pageSize: 25, page: 0 } } }}
                    disableRowSelectionOnClick
                    rowHeight={42}
                  />
                </Box>
              ) : (
                <Alert severity="info">No rows available for this table.</Alert>
              )}
            </Paper>
          ) : null}
        </>
      ) : null}
    </Box>
  );
};

export default BigQueryPage;
