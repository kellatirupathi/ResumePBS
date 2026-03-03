import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileTypeFromBuffer } from "file-type";
import axios from "axios";
import pdf from "pdf-parse";
import Tesseract from "tesseract.js";
import { createCanvas } from "@napi-rs/canvas";
import { safeStr } from "./utils.js";

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36";

const REQUEST_TIMEOUT_MS = 60_000;

export interface DownloadResult {
  success: boolean;
  pathOrError: string;
  fileType: "pdf" | "png" | "jpeg" | "unsupported" | "error";
}

export const isOcrAvailable = (): boolean => {
  try {
    void createCanvas(10, 10);
    return true;
  } catch {
    return false;
  }
};

const isGoogleDocUrl = (url: URL): boolean => url.hostname.includes("docs.google.com") && url.pathname.includes("/document/");

const isGoogleDriveUrl = (url: URL): boolean =>
  url.hostname.includes("drive.google.com") &&
  (url.pathname.includes("open") || url.pathname.includes("file") || url.pathname.includes("/d/"));

const extractGoogleId = (url: URL): string | null => {
  const idFromQuery = url.searchParams.get("id");
  if (idFromQuery) return idFromQuery;

  const match = /\/d\/([^/]+)/.exec(url.pathname);
  return match?.[1] ?? null;
};

const writeBufferToFile = async (buffer: Buffer, outputPath: string): Promise<void> => {
  await fs.writeFile(outputPath, buffer);
};

export const downloadAndIdentifyFile = async (fileUrl: string, outputPath: string): Promise<DownloadResult> => {
  try {
    const parsed = new URL(fileUrl);
    let dataBuffer: Buffer;

    if (isGoogleDocUrl(parsed)) {
      const docId = extractGoogleId(parsed);
      if (!docId) {
        throw new Error("Could not extract Google Docs document ID");
      }

      const exportUrl = `https://docs.google.com/document/d/${docId}/export?format=pdf`;
      const response = await axios.get<ArrayBuffer>(exportUrl, {
        responseType: "arraybuffer",
        timeout: REQUEST_TIMEOUT_MS,
        headers: { "User-Agent": USER_AGENT },
        validateStatus: () => true,
      });

      if (response.status >= 400) {
        throw new Error(`Google Docs export failed with status ${response.status}`);
      }

      dataBuffer = Buffer.from(response.data);
    } else if (isGoogleDriveUrl(parsed)) {
      const fileId = extractGoogleId(parsed);
      if (!fileId) {
        throw new Error("Could not extract Google Drive file ID");
      }

      const exportUrl = `https://docs.google.com/document/d/${fileId}/export?format=pdf`;
      let response = await axios.get<ArrayBuffer>(exportUrl, {
        responseType: "arraybuffer",
        timeout: REQUEST_TIMEOUT_MS,
        headers: { "User-Agent": USER_AGENT },
        validateStatus: () => true,
      });

      let buffer = Buffer.from(response.data);
      if (response.status >= 400 || !buffer.subarray(0, 5).equals(Buffer.from("%PDF-"))) {
        const fallbackUrl = `https://docs.google.com/uc?export=download&id=${fileId}`;
        response = await axios.get<ArrayBuffer>(fallbackUrl, {
          responseType: "arraybuffer",
          timeout: REQUEST_TIMEOUT_MS,
          headers: { "User-Agent": USER_AGENT },
          validateStatus: () => true,
        });

        if (response.status >= 400) {
          throw new Error(`Google Drive fallback download failed with status ${response.status}`);
        }
        buffer = Buffer.from(response.data);
      }

      dataBuffer = buffer;
    } else {
      const response = await axios.get<ArrayBuffer>(fileUrl, {
        responseType: "arraybuffer",
        timeout: REQUEST_TIMEOUT_MS,
        headers: { "User-Agent": USER_AGENT },
        maxRedirects: 5,
        validateStatus: () => true,
      });

      if (response.status >= 400) {
        throw new Error(`Download failed with status ${response.status}`);
      }
      dataBuffer = Buffer.from(response.data);
    }

    if (!dataBuffer.length) {
      throw new Error("Downloaded file is empty");
    }

    await writeBufferToFile(dataBuffer, outputPath);

    const detected = await fileTypeFromBuffer(dataBuffer);
    if (dataBuffer.subarray(0, 5).equals(Buffer.from("%PDF-"))) {
      return { success: true, pathOrError: outputPath, fileType: "pdf" };
    }
    if (detected?.mime === "image/png") {
      return { success: true, pathOrError: outputPath, fileType: "png" };
    }
    if (detected?.mime === "image/jpeg") {
      return { success: true, pathOrError: outputPath, fileType: "jpeg" };
    }

    return { success: true, pathOrError: outputPath, fileType: "unsupported" };
  } catch (error) {
    return {
      success: false,
      pathOrError: `Download failed: ${error instanceof Error ? error.message : String(error)}`,
      fileType: "error",
    };
  }
};

const urlRegex = /https?:\/\/[\w.-]+(?:\/[\w\-./?%&=:#]*)?/gi;

const extractUrlsFromText = (text: string): string[] => {
  const matches = text.match(urlRegex) ?? [];
  const cleaned = matches.map((u) => u.replace(/[\]\[)>(<"',;]+$/g, ""));
  return Array.from(new Set(cleaned));
};

export const extractUrlsFromPdfAnnotations = async (pdfPath: string): Promise<string[]> => {
  try {
    const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
    const data = await fs.readFile(pdfPath);
    const doc = await pdfjs.getDocument({ data }).promise;

    const links = new Set<string>();

    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
      const page = await doc.getPage(pageNumber);
      const annotations = await page.getAnnotations();
      for (const annotation of annotations) {
        const url = safeStr((annotation as { url?: string }).url);
        if (url) links.add(url);
      }
    }

    return Array.from(links);
  } catch {
    return [];
  }
};

const ocrImageBuffer = async (image: Buffer): Promise<string> => {
  const result = await Tesseract.recognize(image, "eng", { logger: () => undefined });
  return safeStr(result.data?.text ?? "");
};

const extractTextFromPdfWithPdfJs = async (pdfPath: string): Promise<string> => {
  try {
    const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
    const data = await fs.readFile(pdfPath);
    const doc = await pdfjs.getDocument({ data }).promise;

    const chunks: string[] = [];
    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
      const page = await doc.getPage(pageNumber);
      const content = await page.getTextContent();
      const items = content.items as Array<{ str?: string }>;
      chunks.push(items.map((item) => safeStr(item.str)).join(" "));
    }

    return chunks.join("\n");
  } catch {
    return "";
  }
};

const ocrPdfWithRendering = async (pdfPath: string): Promise<string> => {
  try {
    const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
    const data = await fs.readFile(pdfPath);
    const doc = await pdfjs.getDocument({ data }).promise;
    const chunks: string[] = [];

    for (let pageNumber = 1; pageNumber <= doc.numPages; pageNumber += 1) {
      const page = await doc.getPage(pageNumber);
      const viewport = page.getViewport({ scale: 2.0 });

      const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
      const context = canvas.getContext("2d");

      await page.render({
        canvas: canvas as unknown as HTMLCanvasElement,
        canvasContext: context as unknown as CanvasRenderingContext2D,
        viewport,
      }).promise;

      const pngBuffer = canvas.toBuffer("image/png");
      const ocrText = await ocrImageBuffer(pngBuffer);
      chunks.push(ocrText);
    }

    return chunks.join("\n");
  } catch {
    return "";
  }
};

export const extractTextAndUrlsFromPdf = async (
  pdfPath: string,
  enableOcr: boolean,
): Promise<{ text: string; urls: string[] }> => {
  let text = "";

  try {
    const data = await fs.readFile(pdfPath);
    const parsed = await pdf(data);
    text = safeStr(parsed.text);
  } catch {
    text = "";
  }

  if (!text.trim()) {
    text = await extractTextFromPdfWithPdfJs(pdfPath);
  }

  if (!text.trim() && enableOcr) {
    text = await ocrPdfWithRendering(pdfPath);
  }

  const annotationUrls = await extractUrlsFromPdfAnnotations(pdfPath);
  const textUrls = extractUrlsFromText(text);
  const urls = Array.from(new Set([...annotationUrls, ...textUrls]));

  return { text, urls };
};

export const extractTextFromImage = async (
  imagePath: string,
  enableOcr: boolean,
): Promise<{ text: string; urls: string[] }> => {
  if (!enableOcr) {
    return { text: "OCR is disabled in settings.", urls: [] };
  }

  try {
    const data = await fs.readFile(imagePath);
    const text = await ocrImageBuffer(data);
    return { text, urls: [] };
  } catch (error) {
    return {
      text: `Error during image processing: ${error instanceof Error ? error.message : String(error)}`,
      urls: [],
    };
  }
};

export const createTempFilePath = (extension = ".tmp"): string => {
  const name = `resume-${Date.now()}-${Math.random().toString(36).slice(2)}${extension}`;
  return path.join(os.tmpdir(), name);
};

export const cleanupTempFile = async (filePath: string | null | undefined): Promise<void> => {
  if (!filePath) return;
  try {
    await fs.unlink(filePath);
  } catch {
    // ignored
  }
};
