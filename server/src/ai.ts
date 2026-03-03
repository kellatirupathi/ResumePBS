import axios from "axios";
import { MISTRAL_ENDPOINT, MISTRAL_MODEL, OPENAI_ENDPOINT, OPENAI_MODEL } from "./constants.js";
import { maskKey } from "./config.js";
import { safeStr } from "./utils.js";

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
const jitterMs = (): number => 500 + Math.floor(Math.random() * 2000);

const systemPrompt =
  "You are an expert resume parser and data analyst. Always return STRICT JSON only-" +
  'no markdown fences, no commentary. Fill missing values with empty strings "" or [] where appropriate.';

const requestWithRetries = async (
  endpoint: string,
  apiKey: string,
  model: string,
  prompt: string,
  providerLabel: "OpenAI" | "Mistral",
): Promise<string> => {
  if (!apiKey) {
    return JSON.stringify({ error: `Missing ${providerLabel} API key for this request.` });
  }

  const attempts = 5;
  const initialBackoffSeconds = 5;

  const headers = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };

  const payload = {
    model,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt },
    ],
    temperature: 0.1,
    max_tokens: 4096,
  };

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      await sleep(jitterMs());

      const response = await axios.post(endpoint, payload, {
        headers,
        timeout: 120_000,
        validateStatus: () => true,
      });

      if (response.status === 200) {
        const content = response.data?.choices?.[0]?.message?.content;

        if (Array.isArray(content)) {
          return content
            .map((part) => {
              if (typeof part === "string") return safeStr(part);
              if (part && typeof part === "object") {
                return safeStr((part as Record<string, unknown>).text);
              }
              return "";
            })
            .join("")
            .trim();
        }

        return safeStr(content).trim();
      }

      if (response.status === 429 || [500, 502, 503, 504].includes(response.status)) {
        const waitSeconds = initialBackoffSeconds * 2 ** attempt + (1 + Math.random() * 2);
        console.warn(
          `${providerLabel} retryable error ${response.status} on key ${maskKey(apiKey)} (attempt ${attempt + 1}/${attempts}). Waiting ${waitSeconds.toFixed(2)}s`,
        );
        await sleep(waitSeconds * 1000);
        continue;
      }

      return JSON.stringify({
        error: `${providerLabel} API Error Status ${response.status}: ${safeStr(response.data)}`,
      });
    } catch (error) {
      const waitSeconds = initialBackoffSeconds * 2 ** attempt + (1 + Math.random() * 2);
      console.warn(
        `${providerLabel} request exception on key ${maskKey(apiKey)} (attempt ${attempt + 1}/${attempts}). Waiting ${waitSeconds.toFixed(2)}s`,
      );
      await sleep(waitSeconds * 1000);
      if (attempt === attempts - 1) {
        return JSON.stringify({
          error: `${providerLabel} request failed: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
    }
  }

  return JSON.stringify({ error: "API Rate Limit Exceeded. Failed after all retries." });
};

export const analyzeTextWithMistral = async (prompt: string, apiKey: string): Promise<string> => {
  return requestWithRetries(MISTRAL_ENDPOINT, apiKey, MISTRAL_MODEL, prompt, "Mistral");
};

export const analyzeTextWithOpenAi = async (prompt: string, apiKey: string): Promise<string> => {
  return requestWithRetries(OPENAI_ENDPOINT, apiKey, OPENAI_MODEL, prompt, "OpenAI");
};

export const analyzeTextWithProvider = async (
  prompt: string,
  provider: string,
  apiKey: string,
): Promise<string> => {
  if (safeStr(provider).toLowerCase() === "openai") {
    return analyzeTextWithOpenAi(prompt, apiKey);
  }
  return analyzeTextWithMistral(prompt, apiKey);
};
