import { google } from "googleapis";
import { GSHEET_NAME } from "./constants.js";
import { loadGoogleServiceAccount } from "./config.js";
import type { RowResult } from "./types.js";

interface SpreadsheetTarget {
  spreadsheetId: string;
}

const getAuthorizedClients = () => {
  const credentials = loadGoogleServiceAccount();
  if (!credentials) {
    return null;
  }

  const auth = new google.auth.GoogleAuth({
    credentials,
    scopes: [
      "https://www.googleapis.com/auth/spreadsheets",
      "https://www.googleapis.com/auth/drive.readonly",
    ],
  });

  return {
    sheets: google.sheets({ version: "v4", auth }),
    drive: google.drive({ version: "v3", auth }),
  };
};

const findSpreadsheetByName = async (name: string): Promise<SpreadsheetTarget | null> => {
  const clients = getAuthorizedClients();
  if (!clients) return null;

  const query = `name='${name.replace(/'/g, "\\'")}' and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false`;
  const response = await clients.drive.files.list({
    q: query,
    fields: "files(id,name)",
    pageSize: 1,
    spaces: "drive",
  });

  const file = response.data.files?.[0];
  if (!file?.id) return null;

  return { spreadsheetId: file.id };
};

const getSheetMetadata = async (spreadsheetId: string) => {
  const clients = getAuthorizedClients();
  if (!clients) return null;

  const response = await clients.sheets.spreadsheets.get({ spreadsheetId });
  return response.data;
};

const ensureWorksheet = async (spreadsheetId: string, sheetName: string): Promise<void> => {
  const clients = getAuthorizedClients();
  if (!clients) return;

  const metadata = await getSheetMetadata(spreadsheetId);
  const existing = metadata?.sheets?.find((sheet) => sheet.properties?.title === sheetName);
  if (existing) return;

  await clients.sheets.spreadsheets.batchUpdate({
    spreadsheetId,
    requestBody: {
      requests: [
        {
          addSheet: {
            properties: {
              title: sheetName,
              gridProperties: { rowCount: 100, columnCount: 50 },
            },
          },
        },
      ],
    },
  });
};

const getHeader = async (spreadsheetId: string, worksheet: string): Promise<string[]> => {
  const clients = getAuthorizedClients();
  if (!clients) return [];

  const response = await clients.sheets.spreadsheets.values.get({
    spreadsheetId,
    range: `${worksheet}!1:1`,
  });

  return (response.data.values?.[0] ?? []).map((value) => String(value));
};

const writeHeader = async (spreadsheetId: string, worksheet: string, header: string[]): Promise<void> => {
  const clients = getAuthorizedClients();
  if (!clients) return;

  await clients.sheets.spreadsheets.values.update({
    spreadsheetId,
    range: `${worksheet}!A1`,
    valueInputOption: "USER_ENTERED",
    requestBody: {
      values: [header],
    },
  });
};

const appendRows = async (spreadsheetId: string, worksheet: string, rows: Array<Array<string | number | boolean | null>>) => {
  const clients = getAuthorizedClients();
  if (!clients || rows.length === 0) return;

  await clients.sheets.spreadsheets.values.append({
    spreadsheetId,
    range: `${worksheet}!A:A`,
    valueInputOption: "USER_ENTERED",
    requestBody: {
      values: rows,
    },
  });
};

export const resolveSubsheetName = (analysisMode: string, shortlistingMode: string): string => {
  if (analysisMode === "shortlisting") {
    return shortlistingMode === "Priority Wise (P1 / P2 / P3 Bands)"
      ? "Priority_Wise_Results"
      : "Probability_Wise_Results";
  }
  return analysisMode.replaceAll(" ", "_");
};

export const writeResultsToGoogleSheets = async (
  analysisMode: string,
  shortlistingMode: string,
  rows: RowResult[],
): Promise<{ ok: boolean; warning?: string }> => {
  if (!rows.length) {
    return { ok: true };
  }

  try {
    const spreadsheet = await findSpreadsheetByName(GSHEET_NAME);
    if (!spreadsheet) {
      return {
        ok: false,
      };
    }

    const worksheet = resolveSubsheetName(analysisMode, shortlistingMode);
    await ensureWorksheet(spreadsheet.spreadsheetId, worksheet);

    const existingHeader = await getHeader(spreadsheet.spreadsheetId, worksheet);

    const allKeys = new Set<string>();
    for (const row of rows) {
      Object.keys(row).forEach((key) => allKeys.add(key));
    }

    let finalHeader = [...existingHeader];
    if (!existingHeader.length) {
      finalHeader = Array.from(allKeys).sort();
      await writeHeader(spreadsheet.spreadsheetId, worksheet, finalHeader);
    } else {
      const newColumns = Array.from(allKeys).filter((key) => !finalHeader.includes(key));
      if (newColumns.length > 0) {
        finalHeader = [...finalHeader, ...newColumns];
        await writeHeader(spreadsheet.spreadsheetId, worksheet, finalHeader);
      }
    }

    const values = rows.map((row) => finalHeader.map((column) => (row[column] ?? "") as string | number | boolean | null));
    await appendRows(spreadsheet.spreadsheetId, worksheet, values);

    return { ok: true };
  } catch (error) {
    return {
      ok: false,
      warning: `Could not write to Google Sheets: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
};
