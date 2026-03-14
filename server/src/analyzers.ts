import axios from "axios";
import { analyzeTextWithProvider, repairJsonWithProvider } from "./ai.js";
import { createTempFilePath, cleanupTempFile, downloadAndIdentifyFile, extractTextAndUrlsFromPdf, extractTextFromImage } from "./extraction.js";
import { SKILL_COLUMNS } from "./constants.js";
import { calculateSkillProbabilities, classifyAndFormatProjectsFromAi, extractGithubUsername, formatMobileNumber, getHighestEducationInstitute, getLatestExperience, isPresentStr, relaxedJsonLoads, safeStr, sortLinks } from "./utils.js";
import type { AnalysisType, Provider, ResumeInputRow, RowResult, ShortlistingMode } from "./types.js";

export interface WorkerContext {
  provider: Provider;
  apiKey: string;
  enableOcr: boolean;
  companyName: string;
  userRequirements: string;
  analysisType: AnalysisType;
  shortlistingMode: ShortlistingMode;
  internalProjectsString: string;
}

const defaultShortlistResult = (row: ResumeInputRow, companyName: string): RowResult => ({
  "User ID": row.user_id,
  "Resume Link": row["Resume link"],
  "Company Name": companyName,
  "Overall Probability": 0,
  "Overall Remarks": "Error processing",
  "Priority Band": "Not Shortlisted",
  "Projects Probability": 0,
  "Projects Remarks": "",
  "Skills Probability": 0,
  "Skills Remarks": "",
  "Experience Probability": 0,
  "Experience Remarks": "",
  "Other Probability": 0,
  "Other Remarks": "",
  "Internal Project Title": "",
  "Internal Projects Techstacks": "",
  "External Project Title": "",
  "External Projects Techstacks": "",
  "Total Projects Count": 0,
  "Internal Projects Count": 0,
  "External Projects Count": 0,
});

const defaultSectionwiseShortlistResult = (row: ResumeInputRow, companyName: string): RowResult => ({
  "User ID": row.user_id,
  "Resume Link": row["Resume link"],
  "Company Name": companyName,
  "Skills": "",
  "Skills Pro": "0%",
  "Projects": "",
  "Projects Pro": "0%",
  "Internal Project Title": "",
  "Internal Projects Techstacks": "",
  "External Project Title": "",
  "External Projects Techstacks": "",
  "Experience": "",
  "Experience Pro": "0%",
  "Latest Experience Company Name": "",
  "Latest Experience Job Title": "",
  "Latest Experience Start Date": "",
  "Latest Experience End Date": "",
  "Certifications": "",
  "Certification Pro": "0%",
  "Education": "",
  "Education Pro": "0%",
  "Summary or Overview": "",
  "Summary Pro": "0%",
  "Overall Probability": 0,
  "Overall Remarks": "Error processing",
});

const normalizeTechToken = (value: string): string =>
  safeStr(value)
    .toLowerCase()
    .replace(/[`'".]/g, "")
    .replace(/\s+/g, " ")
    .trim();

const splitTechStackText = (text: string): string[] =>
  safeStr(text)
    .split(/[\r\n,;|]+/)
    .map((item) => item.trim())
    .filter(Boolean);

const dedupeByNormalization = (items: string[]): string[] => {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of items) {
    const normalized = normalizeTechToken(item);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    result.push(item.trim());
  }
  return result;
};

const extractRequiredTechStacks = (requirements: string): string[] => {
  const direct = dedupeByNormalization(splitTechStackText(requirements)).filter((token) => {
    const words = token.split(/\s+/).filter(Boolean);
    return words.length > 0 && words.length <= 4;
  });
  return direct;
};

const roundToTwoDecimals = (value: number): number => Math.round(value * 100) / 100;

const toPercentLabel = (value: number): string => {
  const normalized = Math.max(0, Math.min(100, roundToTwoDecimals(value)));
  return `${normalized.toFixed(2).replace(/\.?0+$/, "")}%`;
};

type SectionwiseKey = "skills" | "projects" | "experience" | "certifications" | "education" | "summary";

const sectionHeadingAliases: Record<SectionwiseKey, string[]> = {
  skills: ["skills", "technical skills", "skill set", "core competencies", "technologies", "technical proficiencies"],
  projects: ["projects", "project", "project experience", "academic projects", "personal projects", "relevant projects"],
  experience: ["experience", "work experience", "professional experience", "employment history", "internship", "internships"],
  certifications: ["certifications", "certification", "certificates", "courses", "trainings"],
  education: ["education", "academics", "academic details", "qualifications", "educational qualifications"],
  summary: ["summary", "professional summary", "profile summary", "profile", "objective", "career objective", "overview"],
};

const headingSuffixes = new Set(["section", "details", "detail", "history", "profile", "information", "info"]);

const toStringArray = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value.map((item) => safeStr(item).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return splitTechStackText(value);
  }
  return [];
};

const toCompactToken = (value: string): string => safeStr(value).toLowerCase().replace(/[^a-z0-9]/g, "");

const normalizeHeadingText = (value: string): string =>
  safeStr(value)
    .toLowerCase()
    .replace(/[^a-z0-9/&.\s-]/g, " ")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const normalizeSearchText = (value: string): string => safeStr(value).toLowerCase().replace(/\s+/g, " ").trim();

const escapeRegExp = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const findSectionHeading = (line: string): SectionwiseKey | null => {
  const normalized = normalizeHeadingText(line);
  if (!normalized || normalized.split(" ").length > 6) {
    return null;
  }

  for (const [sectionKey, aliases] of Object.entries(sectionHeadingAliases) as Array<[SectionwiseKey, string[]]>) {
    for (const alias of aliases) {
      if (normalized === alias) {
        return sectionKey;
      }

      if (!normalized.startsWith(`${alias} `)) {
        continue;
      }

      const suffix = normalized.slice(alias.length).trim();
      if (headingSuffixes.has(suffix)) {
        return sectionKey;
      }
    }
  }

  return null;
};

const extractSectionTextByHeading = (resumeText: string): Record<SectionwiseKey, string> => {
  const sections = Object.fromEntries(
    (Object.keys(sectionHeadingAliases) as SectionwiseKey[]).map((key) => [key, [] as string[]]),
  ) as Record<SectionwiseKey, string[]>;

  const lines = safeStr(resumeText)
    .split(/\r?\n+/)
    .map((line) => line.trim())
    .filter(Boolean);

  let currentSection: SectionwiseKey | null = null;
  const preamble: string[] = [];

  for (const line of lines) {
    const heading = findSectionHeading(line);
    if (heading) {
      currentSection = heading;
      continue;
    }

    if (currentSection) {
      sections[currentSection].push(line);
    } else {
      preamble.push(line);
    }
  }

  if (!sections.summary.length && preamble.length > 0) {
    sections.summary = preamble.slice(0, 8);
  }

  return Object.fromEntries(
    (Object.keys(sections) as SectionwiseKey[]).map((key) => [key, sections[key].join("\n").trim()]),
  ) as Record<SectionwiseKey, string>;
};

const getTechMatchPatterns = (tech: string): RegExp[] => {
  const compact = toCompactToken(tech);

  switch (compact) {
    case "node":
    case "nodejs":
      return [/\bnode(?:\s*\.?\s*js)?\b/i];
    case "react":
    case "reactjs":
      return [/\breact(?:\s*\.?\s*js)?\b/i];
    case "next":
    case "nextjs":
      return [/\bnext(?:\s*\.?\s*js)?\b/i];
    case "angular":
    case "angularjs":
      return [/\bangular(?:\s*\.?\s*js)?\b/i];
    case "javascript":
      return [/\bjavascript\b/i, /\becmascript\b/i];
    case "typescript":
      return [/\btypescript\b/i];
    case "java":
      return [/\bjava\b/i];
    case "springboot":
      return [/\bspring\s*boot\b/i, /\bspringboot\b/i];
    case "dotnet":
      return [/(?<![a-z0-9])\.net(?![a-z0-9])/i, /\bdot\s*net\b/i];
    case "aiml":
      return [/\bai\s*\/\s*ml\b/i, /\bartificial intelligence\b/i, /\bmachine learning\b/i];
    default: {
      const normalized = safeStr(tech).trim();
      if (!normalized) return [];
      const pattern = normalized
        .split(/\s+/)
        .map((part) => escapeRegExp(part))
        .join("\\s+");
      return [new RegExp(`\\b${pattern}\\b`, "i")];
    }
  }
};

const matchesExplicitTechInText = (text: string, tech: string): boolean => {
  if (!text.trim()) return false;
  return getTechMatchPatterns(tech).some((pattern) => pattern.test(text));
};

const joinFieldValues = (value: unknown): string => {
  if (Array.isArray(value)) {
    return value.map((item) => safeStr(item).trim()).filter(Boolean).join("\n");
  }
  return safeStr(value).trim();
};

const buildProjectEntriesEvidenceText = (projectEntries: unknown[]): string => {
  const chunks: string[] = [];

  for (const entry of projectEntries) {
    if (typeof entry === "string") {
      const value = safeStr(entry).trim();
      if (value) chunks.push(value);
      continue;
    }

    if (!entry || typeof entry !== "object") continue;
    const project = entry as Record<string, unknown>;
    chunks.push(
      joinFieldValues([
        project.title,
        project.projectTitle,
        project.project_title,
        project.name,
        project.projectName,
        project.description,
        project.summary,
        project.details,
        project.techStack,
        project.techstack,
        project.tech_stack,
        project.technologies,
        project.tools,
      ]),
    );
  }

  return chunks.map((chunk) => chunk.trim()).filter(Boolean).join("\n");
};

const buildExperienceEntriesEvidenceText = (experienceEntries: unknown[]): string => {
  const chunks: string[] = [];

  for (const entry of experienceEntries) {
    if (typeof entry === "string") {
      const value = safeStr(entry).trim();
      if (value) chunks.push(value);
      continue;
    }

    if (!entry || typeof entry !== "object") continue;
    const experience = entry as Record<string, unknown>;
    chunks.push(
      joinFieldValues([
        experience.companyName,
        experience.jobTitle,
        experience.startDate,
        experience.endDate,
        experience.description,
        experience.responsibilities,
        experience.techStack,
        experience.technologies,
      ]),
    );
  }

  return chunks.map((chunk) => chunk.trim()).filter(Boolean).join("\n");
};

const collectEvidenceQuotesForSection = (sectionBlock: unknown): string[] => {
  if (!sectionBlock || typeof sectionBlock !== "object" || Array.isArray(sectionBlock)) {
    return [];
  }

  const sectionObj = sectionBlock as Record<string, unknown>;
  const rawEvidence = [
    sectionObj.evidence,
    sectionObj.evidence_quotes,
    sectionObj.evidenceQuotes,
    sectionObj.quotes,
    sectionObj.quote,
    sectionObj.evidence_text,
    sectionObj.evidenceText,
  ];

  const quotes: string[] = [];

  for (const candidate of rawEvidence) {
    if (Array.isArray(candidate)) {
      for (const item of candidate) {
        if (typeof item === "string") {
          const quote = safeStr(item).trim();
          if (quote) quotes.push(quote);
          continue;
        }

        if (!item || typeof item !== "object") continue;
        const quote = safeStr(
          (item as Record<string, unknown>).quote ??
            (item as Record<string, unknown>).text ??
            (item as Record<string, unknown>).evidence ??
            (item as Record<string, unknown>).snippet,
        ).trim();
        if (quote) quotes.push(quote);
      }
      continue;
    }

    const quote = safeStr(candidate).trim();
    if (quote) quotes.push(quote);
  }

  return quotes;
};

const quoteExistsInResume = (resumeText: string, quote: string): boolean => {
  const normalizedResume = normalizeSearchText(resumeText);
  const normalizedQuote = normalizeSearchText(quote);
  if (!normalizedQuote) return false;
  return normalizedResume.includes(normalizedQuote);
};

const buildSectionEvidenceTexts = (
  resumeText: string,
  projectEntries: unknown[],
  experienceEntries: unknown[],
): Record<SectionwiseKey, string> => {
  const extractedSections = extractSectionTextByHeading(resumeText);
  return {
    skills: extractedSections.skills,
    projects: [extractedSections.projects, buildProjectEntriesEvidenceText(projectEntries)].filter(Boolean).join("\n"),
    experience: [extractedSections.experience, buildExperienceEntriesEvidenceText(experienceEntries)].filter(Boolean).join("\n"),
    certifications: extractedSections.certifications,
    education: extractedSections.education,
    summary: extractedSections.summary,
  };
};

const getVerifiedSectionMatches = (
  sectionBlock: unknown,
  sectionEvidenceText: string,
  resumeText: string,
  requiredTechStacks: string[],
): string[] => {
  const verified: string[] = [];
  const evidenceQuotes = collectEvidenceQuotesForSection(sectionBlock);

  for (const tech of requiredTechStacks) {
    if (matchesExplicitTechInText(sectionEvidenceText, tech)) {
      verified.push(tech);
      continue;
    }

    const hasVerifiedQuote = evidenceQuotes.some(
      (quote) =>
        quoteExistsInResume(resumeText, quote) &&
        (!sectionEvidenceText.trim() || quoteExistsInResume(sectionEvidenceText, quote)) &&
        matchesExplicitTechInText(quote, tech),
    );

    if (hasVerifiedQuote) {
      verified.push(tech);
    }
  }

  return verified;
};

const getRequiredTokenAliases = (tech: string): string[] => {
  const compact = toCompactToken(tech);
  const aliases = new Set<string>([compact]);

  if (compact === "nodejs") {
    aliases.add("node");
  }
  if (compact === "reactjs") {
    aliases.add("react");
  }
  if (compact === "nextjs") {
    aliases.add("next");
  }
  if (compact === "aiml") {
    aliases.add("ai");
    aliases.add("ml");
  }

  return Array.from(aliases);
};

const buildRequiredTechLookup = (requiredTechStacks: string[]): Map<string, string> => {
  const lookup = new Map<string, string>();
  for (const tech of requiredTechStacks) {
    for (const alias of getRequiredTokenAliases(tech)) {
      if (!lookup.has(alias)) {
        lookup.set(alias, tech);
      }
    }
  }
  return lookup;
};

const extractSectionBlockFromAi = (data: Record<string, unknown>, keys: string[]): unknown => {
  for (const key of keys) {
    if (data[key] !== undefined) {
      return data[key];
    }
  }
  return undefined;
};

const extractSectionMatchedFromAi = (sectionBlock: unknown, requiredLookup: Map<string, string>): string[] => {
  const sectionObj =
    sectionBlock && typeof sectionBlock === "object" && !Array.isArray(sectionBlock)
      ? (sectionBlock as Record<string, unknown>)
      : {};

  const candidates = [
    sectionObj.matched_techstacks,
    sectionObj.matchedTechstacks,
    sectionObj.matched,
    sectionObj.technologies,
    sectionObj.techStacks,
    sectionObj.tech_stack,
    sectionObj.skills,
    sectionBlock,
  ];

  const matched: string[] = [];
  const seen = new Set<string>();

  for (const candidate of candidates) {
    for (const item of toStringArray(candidate)) {
      const canonical = requiredLookup.get(toCompactToken(item));
      if (!canonical) continue;
      const key = toCompactToken(canonical);
      if (seen.has(key)) continue;
      seen.add(key);
      matched.push(canonical);
    }
  }

  return matched;
};

const parsePercentageNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.min(100, Math.round(value)));
  }

  const text = safeStr(value).replace("%", "").trim();
  if (!text) return null;
  const numeric = Number(text);
  if (!Number.isFinite(numeric)) return null;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const getSectionProbabilityFromFormula = (matchedCount: number, requiredCount: number): number =>
  requiredCount > 0 ? roundToTwoDecimals((matchedCount / requiredCount) * 100) : 0;

const sectionwiseSectionKeyAliases: Record<SectionwiseKey, string[]> = {
  skills: ["skills", "skill"],
  projects: ["projects", "project"],
  experience: ["experience", "work_experience", "internship"],
  certifications: ["certifications", "certification", "certificates"],
  education: ["education", "academics"],
  summary: ["summary", "overview", "profile_summary"],
};

const buildSectionwisePrompt = (context: WorkerContext, requiredTechStacks: string[], resumeText: string): string => {
  const projectClassificationBlock = getProjectClassificationBlock(context.internalProjectsString, context.analysisType);

  return `
You are an expert resume section analyzer.
Return STRICT JSON only.

Job Requirements (raw):
${context.userRequirements}

Required Tech Stacks (canonical list):
${requiredTechStacks.join(", ")}

Task:
1) Analyze the resume text section-wise for:
   - skills
   - projects
   - experience
   - certifications
   - education
   - summary
2) For each section, return:
   - matched_techstacks: only from the canonical required tech stack list
3) In the same response, extract actual project entries into project_entries.
4) In the same response, extract actual work experience / internship entries into experience_entries.
5) Do not include tech stacks outside the required list in matched_techstacks.
6) Keep Java and JavaScript strictly different.
7) Resume formatting may be noisy/flattened; still infer the section context and map correctly.
8) IMPORTANT: Match tech stacks to the correct section only, not global resume-wide mentions.
9) IMPORTANT: Do not treat summary/profile/objective text, "experienced in" statements, skills, certifications, education, or projects as work experience.
10) If the resume has no real work experience or internship entry, set experience_entries to [].
11) project_entries must contain only actual projects from the resume.
12) project_entries must ALWAYS be present as an array. Use [] when none exist.
13) experience_entries must ALWAYS be present as an array. Use [] when none exist.
14) Do NOT omit any top-level key from the required JSON.
15) Never infer Angular from JavaScript/Bootstrap or Node JS from REST API/backend wording. Match only when the technology name or a common alias is explicitly present in that same section.
16) For every matched tech stack, include exact evidence copied from that same section of the resume.

Project Classification Rules:
${projectClassificationBlock}

Required JSON:
{
  "skills": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "projects": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "experience": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "certifications": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "education": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "summary": { "matched_techstacks": ["string"], "evidence": [ { "techstack": "string", "quote": "exact text from this section" } ] },
  "project_entries": [
    { "title": "string", "description": "exact project text from resume when available", "techStack": ["string"], "classification": "Internal" or "External" }
  ],
  "experience_entries": [
    { "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string or list of strings" }
  ],
  "overall_remarks": "string"
}

Resume Text:
---
${resumeText}
---
`;
};

const getComprehensiveDefaultResult = (row: ResumeInputRow, analysisType: AnalysisType, companyName: string): RowResult => {
  if (analysisType === "Internal Projects Matching") {
    const result: RowResult = {
      "User ID": row.user_id,
      "Resume Link": row["Resume link"],
      "Total Projects Count": "",
      "Internal Projects Count": "",
      "External Projects Count": "",
      "Internal Project Titles": "",
      "Internal Project Techstacks": "",
      "External Project Titles": "",
      "External Project Techstacks": "",
      "Company Name": companyName,
    };
    return result;
  }

  const baseColumns = [
    "User ID",
    "Resume Link",
    "Full Name",
    "Mobile Number",
    "Email ID",
    "LinkedIn Link",
    "GitHub Link",
    "Other Links",
    "Skills",
    "Internal Project Title",
    "Internal Projects Techstacks",
    "External Project Title",
    "External Projects Techstacks",
    "Latest Experience Company Name",
    "Latest Experience Job Title",
    "Latest Experience Start Date",
    "Latest Experience End Date",
    "Currently Working? (Yes/No)",
    "Years of IT Experience",
    "Years of Non-IT Experience",
    "City",
    "State",
    "Certifications",
    "Awards",
    "Achievements",
    "GitHub Repo Count",
    "Highest Education Institute Name",
    "Company Name",
  ];

  const educationColumns = [
    "Masters/Doctorate Course Name",
    "Masters/Doctorate Branch",
    "Masters/Doctorate College Name",
    "Masters/Doctorate Year of Completion",
    "Masters/Doctorate Percentage",
    "Bachelors Course Name",
    "Bachelors Branch",
    "Bachelors College Name",
    "Bachelors Year of Completion",
    "Bachelors Percentage",
    "Diploma Course Name",
    "Diploma Branch",
    "Diploma College Name",
    "Diploma Year of Completion",
    "Diploma Percentage",
    "Intermediate / PUC / 12th Board",
    "Intermediate / PUC / 12th Stream/Branch",
    "Intermediate / PUC / 12th School/College Name",
    "Intermediate / PUC / 12th Year of Completion",
    "Intermediate / PUC / 12th Percentage",
    "SSC / 10th Board",
    "SSC / 10th School Name",
    "SSC / 10th Year of Completion",
    "SSC / 10th Percentage",
  ];

  const result: RowResult = {};
  for (const col of [...baseColumns, ...SKILL_COLUMNS, ...educationColumns]) {
    result[col] = "";
  }

  result["User ID"] = row.user_id;
  result["Resume Link"] = row["Resume link"];
  result["Company Name"] = companyName;

  return result;
};

const getProjectInstructionBlock = (): string => {
  return `
"projects": [ { "title": "string", "techStack": ["list of tech keywords"], "classification": "Internal" or "External" } ]
`;
};

const isWeakExtractedText = (text: string): boolean => {
  const normalized = safeStr(text);
  const compactLength = normalized.replace(/\s+/g, "").length;
  const wordCount = (normalized.match(/[a-zA-Z]{3,}/g) ?? []).length;
  return compactLength < 600 || wordCount < 40;
};

const shortlistKeywordBank = [
  "html",
  "css",
  "javascript",
  "react",
  "react js",
  "node",
  "node js",
  "sql",
  "mongodb",
  "aws",
  "gcp",
  "python",
  "java",
  "springboot",
  "django",
  "cloud",
  "cloud computing",
];

const collectRequiredKeywords = (requirements: string): string[] => {
  const reqLower = safeStr(requirements).toLowerCase();
  return shortlistKeywordBank.filter((token, idx, arr) => reqLower.includes(token) && arr.indexOf(token) === idx);
};

const keywordMatch = (text: string, token: string): boolean => {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  return new RegExp(`\\b${escaped}\\b`, "i").test(text);
};

const applyLatestExperienceToResult = (result: RowResult, latestExperience: Record<string, unknown> | null): void => {
  if (!latestExperience) return;

  result["Latest Experience Company Name"] = safeStr(latestExperience.companyName);
  result["Latest Experience Job Title"] = safeStr(latestExperience.jobTitle);
  result["Latest Experience Start Date"] = safeStr(latestExperience.startDate);
  result["Latest Experience End Date"] = safeStr(latestExperience.endDate);
};

const getRequiredSectionwiseArray = (data: Record<string, unknown>, keys: string[], label: string): unknown[] => {
  for (const key of keys) {
    if (!(key in data)) {
      continue;
    }

    const value = data[key];
    if (!Array.isArray(value)) {
      throw new Error(`Sectionwise AI response field '${key}' must be an array for ${label}.`);
    }

    return value;
  }

  throw new Error(`Sectionwise AI response is missing required ${label} field.`);
};

const validateSectionwisePayload = (data: Record<string, unknown>): void => {
  for (const [sectionKey, aliases] of Object.entries(sectionwiseSectionKeyAliases) as Array<[SectionwiseKey, string[]]>) {
    const sectionBlock = extractSectionBlockFromAi(data, aliases);
    if (sectionBlock === undefined) {
      throw new Error(`Sectionwise AI response is missing required '${sectionKey}' section.`);
    }
  }

  getRequiredSectionwiseArray(data, ["project_entries", "projectEntries"], "project_entries");
  getRequiredSectionwiseArray(data, ["experience_entries", "experienceEntries"], "experience_entries");
};

const buildAllDataExtractionPrompt = (resumeText: string, context: WorkerContext): string => {
  const projectInstructionBlock = getProjectInstructionBlock();
  const projectClassificationBlock = getProjectClassificationBlock(context.internalProjectsString, context.analysisType);

  return `
You are a machine that strictly outputs a single, valid JSON object. Analyze the resume text provided below to populate the specified JSON structure.

**JSON STRUCTURE AND INSTRUCTIONS:**
{
  "fullName": "string", "mobileNumber": "string", "email": "string",
  "address": {"city": "string", "state": "string"}, "textLinks": ["list of all URLs found"],
  "skills": ["list of strings"], "certifications": ["list of strings"], "awards": ["list of strings"],
  "achievements": ["list of strings"], "yearsITExperience": "float or string", "yearsNonITExperience": "float or string",
  ${projectInstructionBlock}
  "education": {
    "masters_doctorate": {
        "courseName": "e.g. M.Tech, MBA, PhD",
        "branch": "e.g. Computer Science, VLSI, Marketing",
        "collegeName": "string",
        "completionYear": "string",
        "percentage": "string (e.g. 85% or 8.5 CGPA)"
    },
    "bachelors": {
        "courseName": "e.g. B.Tech, B.Sc, B.Com",
        "branch": "e.g. Computer Science, Mechanical, Civil",
        "collegeName": "string",
        "completionYear": "string",
        "percentage": "string"
    },
    "diploma": {
        "courseName": "e.g. Diploma",
        "branch": "e.g. ECE, CSE",
        "collegeName": "string",
        "completionYear": "string",
        "percentage": "string"
    },
    "intermediate_puc_12th": {
        "board": "e.g. CBSE, State Board",
        "stream": "e.g. MPC, BiPC, Science, Commerce",
        "school_college_name": "string",
        "completionYear": "string",
        "percentage": "string"
    },
    "ssc_10th": {
        "board": "e.g. CBSE, SSC",
        "schoolName": "string",
        "completionYear": "string",
        "percentage": "string"
    }
  },
  "experience": [ { "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string or list of strings" } ]
}

**PROJECT CLASSIFICATION RULES:**
${projectClassificationBlock}

Resume Text:
---
${resumeText}
---
`;
};

const extractProjectsWithFocusedPrompt = async (
  resumeText: string,
  context: WorkerContext,
): Promise<Record<string, string> | null> => {
  const projectClassificationBlock = getProjectClassificationBlock(context.internalProjectsString, context.analysisType);
  const projectPrompt = `
Extract projects from the resume text and return ONLY valid JSON.
Return format:
{
  "projects": [
    {
      "title": "string",
      "description": "exact project text from resume when available",
      "techStack": ["string"],
      "classification": "Internal" or "External"
    }
  ]
}

Rules:
- Extract only actual project entries from the resume.
- Each project must include its title. Include tech stacks only when explicitly present.
- ${projectClassificationBlock.trim()}
- Do not add explanation text outside JSON.

Resume Text:
---
${resumeText}
---
`;

  try {
    const aiResponse = await analyzeTextWithProvider(projectPrompt, context.provider, context.apiKey);
    const data = await parseAiJsonResponse(aiResponse, context.provider, context.apiKey);
    const classified = classifyAndFormatProjectsFromAi(extractProjectsPayload(data));
    const internalCount = safeStr(classified["Internal Project Title"]).split(/\r?\n/).filter(Boolean).length;
    const externalCount = safeStr(classified["External Project Title"]).split(/\r?\n/).filter(Boolean).length;

    return internalCount + externalCount > 0 ? classified : null;
  } catch {
    return null;
  }
};

const getProjectClassificationBlock = (internalProjectsString: string, analysisType?: AnalysisType): string => {
  if (!internalProjectsString || analysisType === "Personal Details") {
    return "If project classification is unclear, default classification to \"External\".";
  }

  return `
Use the OFFICIAL INTERNAL PROJECTS LIST below to classify each extracted project:
- Mark as "Internal" only when the title/description semantically matches a listed internal project.
- Use flexible matching (punctuation/spacing/wording variants should still match).
- Mark all non-matching projects as "External".
OFFICIAL INTERNAL PROJECTS LIST: ${internalProjectsString}
`;
};

const extractProjectsPayload = (data: Record<string, unknown>): unknown[] => {
  const candidates: unknown[] = [
    data.projects,
    data.project,
    data.project_entries,
    data.projectEntries,
    data.projectsList,
    data.project_list,
    data.projectDetails,
    data.project_details,
  ];

  for (const candidate of candidates) {
    if (Array.isArray(candidate)) {
      return candidate;
    }

    if (!candidate || typeof candidate !== "object") {
      continue;
    }

    const obj = candidate as Record<string, unknown>;
    if (Array.isArray(obj.projects)) {
      return obj.projects;
    }
    if (Array.isArray(obj.items)) {
      return obj.items;
    }

    const maybeTitle = safeStr(obj.title ?? obj.projectTitle ?? obj.project_title ?? obj.name ?? obj.projectName).trim();
    if (maybeTitle) {
      return [obj];
    }
  }

  return [];
};

const parseAiJsonResponse = async (
  aiResponse: string,
  provider: Provider,
  apiKey: string,
): Promise<Record<string, unknown>> => {
  try {
    return relaxedJsonLoads(aiResponse);
  } catch (error) {
    if (!(error instanceof SyntaxError)) {
      throw error;
    }

    const repaired = await repairJsonWithProvider(aiResponse, provider, apiKey);
    try {
      return relaxedJsonLoads(repaired);
    } catch (repairError) {
      const firstError = error instanceof Error ? error.message : String(error);
      const secondError = repairError instanceof Error ? repairError.message : String(repairError);
      throw new Error(`Could not parse AI JSON. First error: ${firstError}. Repair error: ${secondError}`);
    }
  }
};

const getGithubRepoCount = async (username: string): Promise<string> => {
  if (!username) return "";
  try {
    const response = await axios.get(`https://api.github.com/users/${username}`, {
      timeout: 10_000,
      validateStatus: () => true,
    });
    if (response.status === 200) {
      return safeStr(response.data?.public_repos);
    }
    return "";
  } catch {
    return "";
  }
};

const loadResumeText = async (
  resumeLink: string,
  enableOcr: boolean,
): Promise<{ text: string; clickableLinks: string[] }> => {
  const tempFilePath = createTempFilePath();
  try {
    const downloaded = await downloadAndIdentifyFile(resumeLink, tempFilePath);
    if (!downloaded.success) {
      throw new Error(downloaded.pathOrError);
    }

    if (downloaded.fileType === "pdf") {
      let extracted = await extractTextAndUrlsFromPdf(tempFilePath, enableOcr);

      // Second pass: force OCR when the first PDF pass yields empty or very weak text.
      if (enableOcr && (!extracted.text.trim() || isWeakExtractedText(extracted.text))) {
        extracted = await extractTextAndUrlsFromPdf(tempFilePath, true, { forceOcr: true });
      }

      return { text: extracted.text, clickableLinks: extracted.urls };
    }

    if (downloaded.fileType === "png" || downloaded.fileType === "jpeg") {
      const { text, urls } = await extractTextFromImage(tempFilePath, enableOcr);
      return { text, clickableLinks: urls };
    }

    throw new Error("Unsupported file type.");
  } finally {
    await cleanupTempFile(tempFilePath);
  }
};

export const processResumeForShortlisting = async (
  row: ResumeInputRow,
  resumeIndex: number,
  context: WorkerContext,
): Promise<RowResult> => {
  const result =
    context.shortlistingMode === "Sectionwise"
      ? defaultSectionwiseShortlistResult(row, context.companyName)
      : defaultShortlistResult(row, context.companyName);
  let resumeText = "";

  try {
    const loaded = await loadResumeText(row["Resume link"], context.enableOcr);
    resumeText = loaded.text;

    if (!resumeText.trim()) {
      throw new Error("Could not extract any text from the file.");
    }

    if (context.shortlistingMode === "Sectionwise") {
      const requiredTechStacks = extractRequiredTechStacks(context.userRequirements);
      if (requiredTechStacks.length === 0) {
        throw new Error("Could not parse required tech stacks from Step 2 input. Enter tech stacks as comma/newline separated values.");
      }
      const prompt = buildSectionwisePrompt(context, requiredTechStacks, resumeText);
      const aiResponse = await analyzeTextWithProvider(prompt, context.provider, context.apiKey);
      const data = await parseAiJsonResponse(aiResponse, context.provider, context.apiKey);

      if (typeof data !== "object" || Array.isArray(data)) {
        throw new Error(`AI returned data that is not a JSON object. Type: ${typeof data}`);
      }

      if (data.error) {
        throw new Error(safeStr(data.error));
      }

      validateSectionwisePayload(data);

      const sectionConfig: Array<{
        key: SectionwiseKey;
        outputColumn: string;
        probabilityColumn: string;
        aiKeys: string[];
      }> = [
        { key: "skills", outputColumn: "Skills", probabilityColumn: "Skills Pro", aiKeys: ["skills", "skill"] },
        { key: "projects", outputColumn: "Projects", probabilityColumn: "Projects Pro", aiKeys: ["projects", "project"] },
        { key: "experience", outputColumn: "Experience", probabilityColumn: "Experience Pro", aiKeys: ["experience", "work_experience", "internship"] },
        {
          key: "certifications",
          outputColumn: "Certifications",
          probabilityColumn: "Certification Pro",
          aiKeys: ["certifications", "certification", "certificates"],
        },
        { key: "education", outputColumn: "Education", probabilityColumn: "Education Pro", aiKeys: ["education", "academics"] },
        { key: "summary", outputColumn: "Summary or Overview", probabilityColumn: "Summary Pro", aiKeys: ["summary", "overview", "profile_summary"] },
      ];

      const sectionScores: number[] = [];
      const requiredCount = requiredTechStacks.length;
      const sectionwiseProjectEntries = getRequiredSectionwiseArray(data, ["project_entries", "projectEntries"], "project_entries");
      const sectionwiseExperienceEntries = getRequiredSectionwiseArray(data, ["experience_entries", "experienceEntries"], "experience_entries");
      const sectionEvidenceTexts = buildSectionEvidenceTexts(resumeText, sectionwiseProjectEntries, sectionwiseExperienceEntries);

      for (const section of sectionConfig) {
        const sectionBlock = extractSectionBlockFromAi(data, section.aiKeys);
        const matchedTechs = getVerifiedSectionMatches(sectionBlock, sectionEvidenceTexts[section.key], resumeText, requiredTechStacks);
        const probability = getSectionProbabilityFromFormula(matchedTechs.length, requiredCount);

        result[section.outputColumn] = matchedTechs.join(", ");
        result[section.probabilityColumn] = toPercentLabel(probability);
        sectionScores.push(probability);
      }

      const overallProbability =
        sectionScores.length > 0 ? roundToTwoDecimals(sectionScores.reduce((sum, score) => sum + score, 0) / sectionScores.length) : 0;
      result["Overall Probability"] = overallProbability;
      result["Overall Remarks"] =
        safeStr(data.overall_remarks ?? data.overallRemarks) ||
        `${requiredTechStacks.length} required tech stack(s); section-wise matching verified against extracted section text.`;

      Object.assign(result, classifyAndFormatProjectsFromAi(sectionwiseProjectEntries));
      applyLatestExperienceToResult(result, getLatestExperience(sectionwiseExperienceEntries));

      return result;
    }

    const textLower = resumeText.toLowerCase();
    const reqsLower = context.userRequirements.toLowerCase();
    let systemWarning = "";

    if (/\bjava\b/.test(reqsLower) && !/\bjava\b/.test(textLower)) {
      systemWarning +=
        "\n\n[SYSTEM WARNING]: The user explicitly requires 'Java' (the backend language). I have scanned the text and 'Java' appears to be MISSING as a standalone word. The text might contain 'JavaScript', but THAT IS NOT JAVA. Treat 'Java' as MISSING.";
    }

    const projectInstructionBlock = getProjectInstructionBlock();
    const projectClassificationBlock = getProjectClassificationBlock(context.internalProjectsString);

    const prompt = `
You are a Nuanced Technical Recruiter and Logic Engine.
Your goal is to categorize the candidate into Priority Bands (P1, P2, P3) based on strict keyword matching.

${systemWarning}

**CRITICAL ANTI-HALLUCINATION RULES:**
1. **JAVA IS NOT JAVASCRIPT.**
   - If the resume contains "JavaScript", "ECMAScript", or "React.js", DO NOT count this as "Java".
   - "Java" is a standalone backend language. "JavaScript" is a frontend language.
   - If the candidate lists "JavaScript Essentials" certification, that is **NOT** Java.
   - If the resume text does not explicitly say "Java" as a separate word, count it as MISSING.

**SCORING GUIDELINES (STRICTLY FOLLOW THIS):**

**BAND P1 (Score 90 - 100): THE PERFECT MATCH**
- The candidate has **ALL** the specific technologies listed in the Required Criteria.
- Example: If user asks for "Java, Springboot, React", the resume MUST have ALL THREE to get > 90.
- If even ONE core skill (especially Java) is missing, DO NOT give a score above 90.

**BAND P2 (Score 75 - 89): THE STRONG CONTENDER (Missing 1-2 Skills)**
- The candidate matches **MOST** of the criteria but is missing a specific technology.
- Example: User asks for "Java, Springboot, React". Candidate has "React and Node" but NO "Java".
- **Action:** This is still a good profile. Do NOT give 0. Give a score between 75 and 89.
- **Remarks:** You must explicitly state: "Candidate fits P2. Good frontend skills, but missing required Java."

**BAND P3 (Score 60 - 74): THE PARTIAL MATCH**
- The candidate has relevant skills but is missing **MAJOR** parts of the stack.
- Example: User asks for "Full Stack Java". Candidate only knows "HTML and CSS".

**BAND F (Score 0 - 59): NO MATCH**
- The resume is completely unrelated to the job description.

Return your answer as a **single, pure JSON object**.

**REQUIRED JSON STRUCTURE:**
{
  "projects_probability": "integer (0-100)",
  "projects_remarks": "string",
  "skills_probability": "integer (0-100)",
  "skills_remarks": "string",
  "experience_probability": "integer (0-100)",
  "experience_remarks": "string",
  "other_probability": "integer (0-100)",
  "other_remarks": "string",
  "overall_probability": "integer (0-100)",
  "overall_remarks": "string",
  ${projectInstructionBlock}
}

**PROJECT CLASSIFICATION RULES:**
${projectClassificationBlock}

---
**Required Criteria:**
${context.userRequirements}
---
**Resume Text:**
${resumeText}
---
`;

    const aiResponse = await analyzeTextWithProvider(prompt, context.provider, context.apiKey);
    const data = await parseAiJsonResponse(aiResponse, context.provider, context.apiKey);

    if (typeof data !== "object" || Array.isArray(data)) {
      throw new Error(`AI returned data that is not a JSON object. Type: ${typeof data}`);
    }

    if (data.error) {
      throw new Error(safeStr(data.error));
    }

    result["Overall Probability"] = Number(data.overall_probability ?? 0);
    result["Projects Probability"] = Number(data.projects_probability ?? 0);
    result["Skills Probability"] = Number(data.skills_probability ?? 0);
    result["Experience Probability"] = Number(data.experience_probability ?? 0);
    result["Other Probability"] = Number(data.other_probability ?? 0);
    result["Overall Remarks"] = safeStr(data.overall_remarks ?? "N/A");
    result["Projects Remarks"] = safeStr(data.projects_remarks ?? "N/A");
    result["Skills Remarks"] = safeStr(data.skills_remarks ?? "N/A");
    result["Experience Remarks"] = safeStr(data.experience_remarks ?? "N/A");
    result["Other Remarks"] = safeStr(data.other_remarks ?? "N/A");

    let classified = classifyAndFormatProjectsFromAi(extractProjectsPayload(data));
    let internalTitles = safeStr(classified["Internal Project Title"]);
    let externalTitles = safeStr(classified["External Project Title"]);
    let internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
    let externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;

    if (internalCount + externalCount === 0) {
      const fallbackClassified = await extractProjectsWithFocusedPrompt(resumeText, context);
      if (fallbackClassified) {
        classified = fallbackClassified;
        internalTitles = safeStr(classified["Internal Project Title"]);
        externalTitles = safeStr(classified["External Project Title"]);
        internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
        externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;
      }
    }

    Object.assign(result, classified);

    result["Internal Projects Count"] = internalCount;
    result["External Projects Count"] = externalCount;
    result["Total Projects Count"] = internalCount + externalCount;

    const isErrorRemark = safeStr(result["Overall Remarks"]).toLowerCase().startsWith("error:");
    const overallProbability = Number(result["Overall Probability"] ?? 0);
    if (!isErrorRemark && overallProbability <= 0) {
      const requiredKeywords = collectRequiredKeywords(context.userRequirements);
      if (requiredKeywords.length > 0) {
        const matchedKeywords = requiredKeywords.filter((token) => keywordMatch(resumeText, token));
        if (matchedKeywords.length > 0) {
          const coverage = matchedKeywords.length / requiredKeywords.length;
          const adjustedOverall = Math.max(35, Math.min(74, Math.round(coverage * 100)));
          result["Overall Probability"] = adjustedOverall;
          if (Number(result["Skills Probability"] ?? 0) <= 0) {
            result["Skills Probability"] = Math.max(30, Math.round(adjustedOverall * 0.8));
          }
          result["Overall Remarks"] = `${safeStr(result["Overall Remarks"])} | System adjustment: ${matchedKeywords.length}/${requiredKeywords.length} required keywords detected in resume text (${matchedKeywords.join(", ")}).`;
        }
      }
    }
  } catch (error) {
    const errorText = `Error: ${error instanceof Error ? error.message : String(error)}`;
    result["Overall Remarks"] = errorText;
    if (context.shortlistingMode === "Sectionwise") {
      result["Summary or Overview"] = errorText;
      result["Skills Pro"] = "0%";
      result["Projects Pro"] = "0%";
      result["Experience Pro"] = "0%";
      result["Certification Pro"] = "0%";
      result["Education Pro"] = "0%";
      result["Summary Pro"] = "0%";
      result["Overall Probability"] = 0;
    } else {
      result["Projects Remarks"] = errorText;
      result["Skills Remarks"] = errorText;
      result["Experience Remarks"] = errorText;
      result["Other Remarks"] = errorText;
    }
  }

  return result;
};

export const processResumeComprehensively = async (
  row: ResumeInputRow,
  resumeIndex: number,
  context: WorkerContext,
): Promise<RowResult> => {
  const result = getComprehensiveDefaultResult(row, context.analysisType, context.companyName);
  let aiResponseForDebug = "";
  let resumeText = "";
  let clickableLinks: string[] = [];

  try {
    const loaded = await loadResumeText(row["Resume link"], context.enableOcr);
    resumeText = loaded.text;
    clickableLinks = loaded.clickableLinks;

    if (!resumeText.trim()) {
      throw new Error("Could not extract any text from the file.");
    }

    const projectInstructionBlock = getProjectInstructionBlock();
    const projectClassificationBlock = getProjectClassificationBlock(context.internalProjectsString, context.analysisType);

    let prompt = "";

    if (context.analysisType === "Internal Projects Matching") {
      prompt = `
You are a project classification expert. Analyze the provided resume text and perform this CRITICAL task:
1. Extract all projects mentioned in the resume.
2. For each project, determine if it is an "Internal" or "External" project by comparing it against the provided OFFICIAL INTERNAL PROJECTS LIST. Your matching should be smart and flexible.
3. Return ONLY a pure JSON object with the results.

OFFICIAL INTERNAL PROJECTS LIST:
---
${context.internalProjectsString}
---

REQUIRED JSON Structure:
{
  "projects": [
    {
      "title": "string",
      "techStack": ["list", "of", "technologies"],
      "classification": "Internal" or "External"
    }
  ]
}

Resume Text:
---
${resumeText}
---
`;
    } else if (context.analysisType === "All Data") {
      prompt = buildAllDataExtractionPrompt(resumeText, context);
    } else if (context.analysisType === "Personal Details") {
      prompt = `
Analyze the provided resume text and extract ONLY the personal details into a pure JSON object.
The entire response MUST be ONLY the JSON object.
JSON Structure: {"fullName": "string", "mobileNumber": "string", "email": "string", "address": {"city": "string", "state": "string"}, "textLinks": ["list of strings"]}
Resume Text: --- ${resumeText} ---
`;
    } else {
      prompt = `
You are an expert data extractor. Analyze the resume and produce a single JSON object.

**JSON STRUCTURE AND INSTRUCTIONS:**
{
  "skills": ["list of strings"],
  "certifications": ["list of strings"],
  "awards": ["list of strings"],
  "achievements": ["list of strings"],
  ${projectInstructionBlock}
  "experience": [{ "companyName": "string", "jobTitle": "string", "startDate": "string", "endDate": "string", "description": "string"}]
}

**PROJECT CLASSIFICATION RULES:**
${projectClassificationBlock}

Resume Text:
---
${resumeText}
---
`;
    }

    const aiResponse = await analyzeTextWithProvider(prompt, context.provider, context.apiKey);
    aiResponseForDebug = aiResponse;
    const data = await parseAiJsonResponse(aiResponse, context.provider, context.apiKey);

    if (typeof data !== "object" || Array.isArray(data)) {
      throw new Error(`AI returned non-dict data. Type: ${typeof data}`);
    }
    if (data.error) {
      throw new Error(safeStr(data.error));
    }

    let classifiedProjects = classifyAndFormatProjectsFromAi(extractProjectsPayload(data));
    let internalTitles = safeStr(classifiedProjects["Internal Project Title"]);
    let externalTitles = safeStr(classifiedProjects["External Project Title"]);
    let internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
    let externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;

    if (internalCount + externalCount === 0 && context.analysisType !== "Personal Details") {
      const fallbackClassified = await extractProjectsWithFocusedPrompt(resumeText, context);
      if (fallbackClassified) {
        classifiedProjects = fallbackClassified;
        internalTitles = safeStr(classifiedProjects["Internal Project Title"]);
        externalTitles = safeStr(classifiedProjects["External Project Title"]);
        internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
        externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;
      }
    }

    if (context.analysisType === "Internal Projects Matching") {
      result["Total Projects Count"] = internalCount + externalCount;
      result["Internal Projects Count"] = internalCount;
      result["External Projects Count"] = externalCount;
      result["Internal Project Titles"] = internalTitles;
      result["Internal Project Techstacks"] = safeStr(classifiedProjects["Internal Projects Techstacks"]);
      result["External Project Titles"] = externalTitles;
      result["External Project Techstacks"] = safeStr(classifiedProjects["External Projects Techstacks"]);
    } else {
      Object.assign(result, classifiedProjects);
      result["Internal Projects Count"] = internalCount;
      result["External Projects Count"] = externalCount;
      result["Total Projects Count"] = internalCount + externalCount;

      if (["All Data", "Personal Details"].includes(context.analysisType)) {
        const address = typeof data.address === "object" && data.address !== null ? (data.address as Record<string, unknown>) : {};

        result["Full Name"] = safeStr(data.fullName);
        result["Mobile Number"] = formatMobileNumber(safeStr(data.mobileNumber));
        result["Email ID"] = safeStr(data.email);
        result["City"] = safeStr(address.city);
        result["State"] = safeStr(address.state);

        const textLinksRaw = data.textLinks;
        const textLinks = Array.isArray(textLinksRaw)
          ? textLinksRaw.map((link) => safeStr(link))
          : textLinksRaw
            ? [safeStr(textLinksRaw)]
            : [];

        const allLinks = Array.from(new Set([...textLinks, ...clickableLinks].map((link) => safeStr(link)).filter(Boolean))).sort();
        const { linkedin, github, otherLinks } = sortLinks(allLinks);

        result["LinkedIn Link"] = linkedin;
        result["GitHub Link"] = github;
        result["Other Links"] = otherLinks.join("\n");

        if (github) {
          const username = extractGithubUsername(github);
          if (username) {
            result["GitHub Repo Count"] = await getGithubRepoCount(username);
          }
        }
      }

      if (["All Data", "Skills & Projects"].includes(context.analysisType)) {
        Object.assign(result, calculateSkillProbabilities(data));

        if (Array.isArray(data.skills)) {
          const uniqueSkills = Array.from(new Set(data.skills.map((skill) => safeStr(skill)).filter(Boolean))).sort();
          result["Skills"] = uniqueSkills.join(", ");
        }

        if (Array.isArray(data.certifications)) {
          result["Certifications"] = data.certifications.map((item) => safeStr(item)).filter(Boolean).join("\n");
        }

        if (Array.isArray(data.awards)) {
          result["Awards"] = data.awards.map((item) => safeStr(item)).filter(Boolean).join("\n");
        }

        if (Array.isArray(data.achievements)) {
          result["Achievements"] = data.achievements.map((item) => safeStr(item)).filter(Boolean).join("\n");
        }

        const latestExperience = getLatestExperience(data.experience);
        if (latestExperience) {
          const endDate = latestExperience.endDate;
          result["Latest Experience Company Name"] = safeStr(latestExperience.companyName);
          result["Latest Experience Job Title"] = safeStr(latestExperience.jobTitle);
          result["Latest Experience Start Date"] = safeStr(latestExperience.startDate);
          result["Latest Experience End Date"] = safeStr(endDate);
          result["Currently Working? (Yes/No)"] = isPresentStr(endDate) ? "Yes" : "No";
        }
      }

      if (context.analysisType === "All Data") {
        result["Years of IT Experience"] = safeStr(data.yearsITExperience);
        result["Years of Non-IT Experience"] = safeStr(data.yearsNonITExperience);

        const education = typeof data.education === "object" && data.education !== null ? (data.education as Record<string, Record<string, unknown>>) : {};
        result["Highest Education Institute Name"] = getHighestEducationInstitute(education);

        const masters = education.masters_doctorate ?? {};
        result["Masters/Doctorate Course Name"] = safeStr(masters.courseName);
        result["Masters/Doctorate Branch"] = safeStr(masters.branch);
        result["Masters/Doctorate College Name"] = safeStr(masters.collegeName);
        result["Masters/Doctorate Year of Completion"] = safeStr(masters.completionYear);
        result["Masters/Doctorate Percentage"] = safeStr(masters.percentage);

        const bachelors = education.bachelors ?? {};
        result["Bachelors Course Name"] = safeStr(bachelors.courseName);
        result["Bachelors Branch"] = safeStr(bachelors.branch);
        result["Bachelors College Name"] = safeStr(bachelors.collegeName);
        result["Bachelors Year of Completion"] = safeStr(bachelors.completionYear);
        result["Bachelors Percentage"] = safeStr(bachelors.percentage);

        const diploma = education.diploma ?? {};
        result["Diploma Course Name"] = safeStr(diploma.courseName);
        result["Diploma Branch"] = safeStr(diploma.branch);
        result["Diploma College Name"] = safeStr(diploma.collegeName);
        result["Diploma Year of Completion"] = safeStr(diploma.completionYear);
        result["Diploma Percentage"] = safeStr(diploma.percentage);

        const intermediate = education.intermediate_puc_12th ?? {};
        result["Intermediate / PUC / 12th Board"] = safeStr(intermediate.board);
        result["Intermediate / PUC / 12th Stream/Branch"] = safeStr(intermediate.stream);
        result["Intermediate / PUC / 12th School/College Name"] = safeStr(intermediate.school_college_name);
        result["Intermediate / PUC / 12th Year of Completion"] = safeStr(intermediate.completionYear);
        result["Intermediate / PUC / 12th Percentage"] = safeStr(intermediate.percentage);

        const ssc = education.ssc_10th ?? {};
        result["SSC / 10th Board"] = safeStr(ssc.board);
        result["SSC / 10th School Name"] = safeStr(ssc.schoolName);
        result["SSC / 10th Year of Completion"] = safeStr(ssc.completionYear);
        result["SSC / 10th Percentage"] = safeStr(ssc.percentage);
      }
    }
  } catch (error) {
    const displayError = `Error: ${error instanceof Error ? error.message : String(error)}`;
    if (context.analysisType === "Internal Projects Matching") {
      result["Total Projects Count"] = displayError;
    } else {
      result["Full Name"] = displayError;
    }

    if (error instanceof SyntaxError || safeStr(error instanceof Error ? error.message : error).includes("Could not parse AI JSON")) {
      console.debug("Malformed AI response", aiResponseForDebug);
    }
  }

  return result;
};
