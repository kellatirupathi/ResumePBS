import fs from "node:fs";
import { google } from "googleapis";
import { config } from "./config.js";
import { safeStr } from "./utils.js";

const BIGQUERY_SCOPE = "https://www.googleapis.com/auth/bigquery.readonly";
const PAGE_SIZE = 10_000;

interface BigQueryField {
  name: string;
  type: string;
  mode: string;
  fields: BigQueryField[];
}

interface BigQueryRowCell {
  v?: unknown;
}

interface BigQueryTableRow {
  f?: BigQueryRowCell[];
}

export interface BigQueryTableSnapshot {
  tableId: string;
  tableType: string;
  rowCount: number;
  columns: string[];
  rows: Array<Record<string, unknown>>;
  fetchError?: string;
}

export interface BigQueryTableSummary {
  tableId: string;
  tableType: string;
  rowCount: number;
}

export interface BigQueryDatasetSnapshot {
  projectId: string;
  datasetId: string;
  tableCount: number;
  tables: BigQueryTableSummary[];
}

export interface BigQueryTableDataSnapshot {
  projectId: string;
  datasetId: string;
  table: BigQueryTableSnapshot;
}

export interface BigQueryLearningMetricsResult {
  projectId: string;
  datasetId: string;
  tableId: string;
  userIdColumn: string;
  inputUserCount: number;
  matchedRowCount: number;
  rows: Array<Record<string, unknown>>;
}

const toRecord = (value: unknown): Record<string, unknown> =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};

const normalizeField = (value: unknown): BigQueryField => {
  const source = toRecord(value);
  const nestedFields = Array.isArray(source.fields) ? source.fields.map((field) => normalizeField(field)) : [];
  return {
    name: safeStr(source.name) || "column",
    type: safeStr(source.type) || "STRING",
    mode: safeStr(source.mode) || "NULLABLE",
    fields: nestedFields,
  };
};

const normalizeSchema = (tableSchema: unknown): BigQueryField[] => {
  const source = toRecord(tableSchema);
  if (!Array.isArray(source.fields)) return [];
  return source.fields.map((field) => normalizeField(field));
};

const decodeFieldValue = (value: unknown, field: BigQueryField): unknown => {
  if (field.mode === "REPEATED") {
    const items = Array.isArray(value) ? value : [];
    return items.map((item) => {
      const rawItem = toRecord(item).v ?? item;
      return decodeFieldValue(rawItem, { ...field, mode: "NULLABLE" });
    });
  }

  if (field.type === "RECORD") {
    const source = toRecord(value);
    const cells = Array.isArray(source.f) ? (source.f as BigQueryRowCell[]) : [];
    const record: Record<string, unknown> = {};

    field.fields.forEach((subField, index) => {
      const raw = cells[index]?.v ?? null;
      record[subField.name || `field_${index + 1}`] = decodeFieldValue(raw, subField);
    });

    return record;
  }

  if (value === undefined || value === null) {
    return null;
  }

  return value;
};

const mapRowToObject = (row: BigQueryTableRow, schema: BigQueryField[]): Record<string, unknown> => {
  const result: Record<string, unknown> = {};
  const cells = Array.isArray(row.f) ? row.f : [];

  schema.forEach((field, index) => {
    const key = field.name || `column_${index + 1}`;
    const raw = cells[index]?.v ?? null;
    result[key] = decodeFieldValue(raw, field);
  });

  return result;
};

const createBigQueryClient = () => {
  const projectId = safeStr(config.bigQueryProjectId);
  const defaultDatasetId = safeStr(config.bigQueryDatasetId);
  const keyFile = safeStr(config.bigQueryServiceAccountPath);

  if (!projectId) {
    throw new Error("BIGQUERY_PROJECT_ID must be configured.");
  }
  if (!defaultDatasetId) {
    throw new Error("BIGQUERY_DATASET_ID must be configured.");
  }
  if (!keyFile || !fs.existsSync(keyFile)) {
    throw new Error("BIGQUERY_SERVICE_ACCOUNT_PATH is missing or file does not exist.");
  }

  const auth = new google.auth.GoogleAuth({
    keyFile,
    scopes: [BIGQUERY_SCOPE],
  });

  const client = google.bigquery({ version: "v2", auth });
  return { client, projectId, defaultDatasetId };
};

const isValidIdentifier = (value: string): boolean => /^[A-Za-z_][A-Za-z0-9_]*$/.test(value);

const getTableMetadata = async (
  client: ReturnType<typeof google.bigquery>,
  projectId: string,
  datasetId: string,
  tableId: string,
) => {
  const tableResponse = await client.tables.get({
    projectId,
    datasetId,
    tableId,
  });

  const tableType = safeStr(tableResponse.data.type || "TABLE");
  const schema = normalizeSchema(tableResponse.data.schema);
  const columns = schema.map((field) => field.name);
  const rowCountValue = Number(safeStr(tableResponse.data.numRows));
  const rowCount = Number.isFinite(rowCountValue) ? rowCountValue : 0;

  return {
    tableType,
    schema,
    columns,
    rowCount,
  };
};

const fetchAllRowsForTable = async (
  client: ReturnType<typeof google.bigquery>,
  projectId: string,
  datasetId: string,
  tableId: string,
  schema: BigQueryField[],
): Promise<Array<Record<string, unknown>>> => {
  const rows: Array<Record<string, unknown>> = [];
  let pageToken: string | undefined;

  do {
    const response = await client.tabledata.list({
      projectId,
      datasetId,
      tableId,
      maxResults: PAGE_SIZE,
      pageToken,
    });

    const batch = Array.isArray(response.data.rows) ? (response.data.rows as BigQueryTableRow[]) : [];
    for (const row of batch) {
      rows.push(mapRowToObject(row, schema));
    }

    const nextToken = safeStr(response.data.pageToken);
    pageToken = nextToken || undefined;
  } while (pageToken);

  return rows;
};

export const fetchBigQueryDatasetSnapshot = async (): Promise<BigQueryDatasetSnapshot> => {
  const { client, projectId, defaultDatasetId } = createBigQueryClient();
  const datasetId = defaultDatasetId;

  const listResponse = await client.tables.list({
    projectId,
    datasetId,
    maxResults: 1_000,
  });

  const listedTables = Array.isArray(listResponse.data.tables) ? listResponse.data.tables : [];
  const tables: BigQueryTableSummary[] = [];

  for (const listedTable of listedTables) {
    const tableId = safeStr(listedTable.tableReference?.tableId);
    if (!tableId) continue;

    try {
      const metadata = await getTableMetadata(client, projectId, datasetId, tableId);
      tables.push({
        tableId,
        tableType: metadata.tableType,
        rowCount: metadata.rowCount,
      });
    } catch {
      tables.push({
        tableId,
        tableType: safeStr(listedTable.type || "TABLE"),
        rowCount: 0,
      });
    }
  }

  return {
    projectId,
    datasetId,
    tableCount: tables.length,
    tables,
  };
};

export const fetchBigQueryTableSnapshot = async (tableIdInput: string): Promise<BigQueryTableDataSnapshot> => {
  const tableId = safeStr(tableIdInput);
  if (!isValidIdentifier(tableId)) {
    throw new Error("Invalid tableId. Use letters, numbers, and underscores only.");
  }

  const { client, projectId, defaultDatasetId } = createBigQueryClient();
  const datasetId = defaultDatasetId;
  const metadata = await getTableMetadata(client, projectId, datasetId, tableId);

  if (metadata.tableType.toUpperCase() === "EXTERNAL") {
    return {
      projectId,
      datasetId,
      table: {
        tableId,
        tableType: metadata.tableType,
        rowCount: metadata.rowCount,
        columns: metadata.columns,
        rows: [],
        fetchError: "External table preview is skipped by API limitations.",
      },
    };
  }

  const rows = await fetchAllRowsForTable(client, projectId, datasetId, tableId, metadata.schema);

  return {
    projectId,
    datasetId,
    table: {
      tableId,
      tableType: metadata.tableType,
      rowCount: metadata.rowCount || rows.length,
      columns: metadata.columns,
      rows,
    },
  };
};

const normalizeInputUserIds = (userIds: string[]): string[] => {
  const unique = new Set<string>();
  for (const userId of userIds) {
    const normalized = safeStr(userId).trim();
    if (normalized) {
      unique.add(normalized);
    }
  }
  return Array.from(unique);
};

const fetchQueryPages = async (
  client: ReturnType<typeof google.bigquery>,
  projectId: string,
  initialRows: BigQueryTableRow[],
  schema: BigQueryField[],
  jobId: string,
  initialPageToken: string,
): Promise<Array<Record<string, unknown>>> => {
  const mappedRows: Array<Record<string, unknown>> = initialRows.map((row) => mapRowToObject(row, schema));
  let pageToken: string | undefined = initialPageToken || undefined;

  while (pageToken) {
    const nextPage = await client.jobs.getQueryResults({
      projectId,
      jobId,
      pageToken,
      maxResults: PAGE_SIZE,
    });

    const batch = Array.isArray(nextPage.data.rows) ? (nextPage.data.rows as BigQueryTableRow[]) : [];
    for (const row of batch) {
      mappedRows.push(mapRowToObject(row, schema));
    }

    const nextToken = safeStr(nextPage.data.pageToken);
    pageToken = nextToken || undefined;
  }

  return mappedRows;
};

export const fetchLearningMetricsByUserIds = async (userIds: string[]): Promise<BigQueryLearningMetricsResult> => {
  const normalizedUserIds = normalizeInputUserIds(userIds);
  if (normalizedUserIds.length === 0) {
    throw new Error("At least one user_id is required.");
  }

  const { client, projectId } = createBigQueryClient();
  const datasetId = safeStr(config.bigQueryStoreDatasetId);
  const tableId = safeStr(config.bigQueryStoreTableId);
  const userIdColumn = safeStr(config.bigQueryUserIdColumn) || "user_id";

  if (!datasetId || !tableId) {
    throw new Error("BIGQUERY_STORE_DATASET_ID and BIGQUERY_STORE_TABLE_ID must be configured.");
  }
  if (!isValidIdentifier(datasetId) || !isValidIdentifier(tableId) || !isValidIdentifier(userIdColumn)) {
    throw new Error("BigQuery dataset/table/user-id column contains invalid characters.");
  }

  const query = `
SELECT *
FROM \`${projectId}.${datasetId}.${tableId}\`
WHERE CAST(\`${userIdColumn}\` AS STRING) IN UNNEST(@userIds)
`;

  const queryResponse = await client.jobs.query({
    projectId,
    requestBody: {
      useLegacySql: false,
      query,
      parameterMode: "NAMED",
      queryParameters: [
        {
          name: "userIds",
          parameterType: { type: "ARRAY", arrayType: { type: "STRING" } },
          parameterValue: {
            arrayValues: normalizedUserIds.map((id) => ({ value: id })),
          },
        },
      ],
      maxResults: PAGE_SIZE,
    },
  });

  const schema = normalizeSchema(queryResponse.data.schema);
  const firstBatch = Array.isArray(queryResponse.data.rows) ? (queryResponse.data.rows as BigQueryTableRow[]) : [];
  const jobId = safeStr(queryResponse.data.jobReference?.jobId);
  const pageToken = safeStr(queryResponse.data.pageToken);

  const rows =
    jobId && pageToken
      ? await fetchQueryPages(client, projectId, firstBatch, schema, jobId, pageToken)
      : firstBatch.map((row) => mapRowToObject(row, schema));

  return {
    projectId,
    datasetId,
    tableId,
    userIdColumn,
    inputUserCount: normalizedUserIds.length,
    matchedRowCount: rows.length,
    rows,
  };
};
