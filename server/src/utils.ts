import dayjs from "dayjs";
import { stringify } from "csv-stringify/sync";
import { DETAILED_EDUCATION_COLUMNS, EXPERIENCE_COLUMNS, INTERNAL_MATCHING_COLUMNS, OTHER_COLUMNS, PRIORITY_ORDER, PROJECT_COLUMNS, SKILL_COLUMNS, SKILLS_TO_ASSESS } from "./constants.js";
import type { FilterQuery, RowResult } from "./types.js";

const CONTROL_CHARS_RE = /[\x00-\x1f\x7f-\x9f]/g;

export const safeStr = (value: unknown): string => {
  const asString = value === null || value === undefined ? "" : String(value);
  return asString.replace(CONTROL_CHARS_RE, "");
};

export const isPresentStr = (value: unknown): boolean => {
  const lower = safeStr(value).toLowerCase();
  return lower.includes("present") || lower.includes("current");
};

export const sanitizeJsonText = (text: string): string => {
  let sanitized = safeStr(text).trim();

  const firstBrace = sanitized.indexOf("{");
  const firstBracket = sanitized.indexOf("[");

  let startPos = -1;
  if (firstBrace === -1 && firstBracket === -1) {
    return '{"error": "No valid JSON object or array found in response"}';
  }

  if (firstBrace !== -1 && firstBracket !== -1) {
    startPos = Math.min(firstBrace, firstBracket);
  } else if (firstBrace !== -1) {
    startPos = firstBrace;
  } else {
    startPos = firstBracket;
  }

  const lastBrace = sanitized.lastIndexOf("}");
  const lastBracket = sanitized.lastIndexOf("]");
  const endPos = Math.max(lastBrace, lastBracket);

  if (endPos < startPos) {
    return '{"error": "No valid JSON structure found in response"}';
  }

  sanitized = sanitized.slice(startPos, endPos + 1);
  sanitized = sanitized.replace(/[\x00-\x1f\x7f-\x9f]/g, "");
  sanitized = sanitized.replace(/\\(?!["\\/bfnrtu])/g, "\\\\");

  if ((sanitized.match(/"/g) ?? []).length % 2 !== 0) {
    sanitized += '"';
  }

  const openBraces = (sanitized.match(/\{/g) ?? []).length;
  const closeBraces = (sanitized.match(/\}/g) ?? []).length;
  if (openBraces > closeBraces) {
    sanitized += "}".repeat(openBraces - closeBraces);
  }

  const openBrackets = (sanitized.match(/\[/g) ?? []).length;
  const closeBrackets = (sanitized.match(/\]/g) ?? []).length;
  if (openBrackets > closeBrackets) {
    sanitized += "]".repeat(openBrackets - closeBrackets);
  }

  return sanitized;
};

export const relaxedJsonLoads = (text: string): Record<string, unknown> => {
  const sanitized = sanitizeJsonText(text);
  return JSON.parse(sanitized) as Record<string, unknown>;
};

export const assignPriorityBand = (probability: unknown): string => {
  const numeric = Number(probability);
  if (Number.isNaN(numeric)) return "Not Shortlisted";
  if (numeric >= 90) return "P1";
  if (numeric >= 75) return "P2";
  if (numeric >= 60) return "P3";
  return "Not Shortlisted";
};

export const formatMobileNumber = (rawNumber: string): string => {
  const raw = safeStr(rawNumber);
  if (!raw.trim()) return "";

  const digitsOnly = raw.replace(/\D/g, "");
  if (digitsOnly.length === 12 && digitsOnly.startsWith("91")) {
    return digitsOnly.slice(2);
  }
  if (digitsOnly.length === 11 && digitsOnly.startsWith("0")) {
    return digitsOnly.slice(1);
  }
  if (digitsOnly.length === 10) {
    return digitsOnly;
  }
  return "";
};

export const extractGithubUsername = (url: string): string | null => {
  const match = /github\.com\/([^/]+)/i.exec(url);
  return match?.[1] ?? null;
};

export const sortLinks = (links: string[]): { linkedin: string; github: string; otherLinks: string[] } => {
  let linkedin = "";
  let github = "";
  const otherLinks: string[] = [];

  for (const linkRaw of links) {
    const link = safeStr(linkRaw);
    if (link.includes("linkedin.com/in/") && !linkedin) {
      linkedin = link;
    } else if (link.includes("github.com") && !github) {
      github = link;
    } else {
      otherLinks.push(link);
    }
  }

  if (github) {
    const username = extractGithubUsername(github);
    if (username) {
      github = `https://github.com/${username}`;
    }
  }

  return { linkedin, github, otherLinks };
};

export const getLatestExperience = (expList: unknown): Record<string, unknown> | null => {
  if (!Array.isArray(expList) || expList.length === 0) {
    return null;
  }

  for (const exp of expList) {
    if (!exp || typeof exp !== "object") continue;
    const endDate = safeStr((exp as Record<string, unknown>).endDate);
    if (isPresentStr(endDate)) {
      return exp as Record<string, unknown>;
    }
  }

  for (const exp of expList) {
    if (exp && typeof exp === "object") {
      return exp as Record<string, unknown>;
    }
  }

  return null;
};

export const getHighestEducationInstitute = (education: unknown): string => {
  if (!education || typeof education !== "object") {
    return "";
  }

  const edu = education as Record<string, Record<string, unknown>>;

  const picks: Array<[string, string]> = [
    ["masters_doctorate", "collegeName"],
    ["bachelors", "collegeName"],
    ["diploma", "collegeName"],
    ["intermediate_puc_12th", "school_college_name"],
    ["ssc_10th", "schoolName"],
  ];

  for (const [level, key] of picks) {
    const value = safeStr(edu[level]?.[key]);
    if (value) return value;
  }

  return "";
};

const hasRegexMatch = (pattern: RegExp, value: string): boolean => pattern.test(value);

export const calculateSkillProbabilities = (data: unknown): Record<string, number> => {
  const scores = Object.fromEntries(SKILL_COLUMNS.map((column) => [column, 0])) as Record<string, number>;
  if (!data || typeof data !== "object") {
    return scores;
  }

  const payload = data as Record<string, unknown>;

  const skillsList = Array.isArray(payload.skills) ? payload.skills.map((s) => safeStr(s).toLowerCase()) : [];
  const skillsText = skillsList.join(" ");

  const projects = payload.projects;
  const projectsText = Array.isArray(projects)
    ? projects.map((project) => safeStr(project).toLowerCase()).join(" ")
    : safeStr(projects).toLowerCase();

  const certs = payload.certifications;
  const certsText = Array.isArray(certs)
    ? certs.map((cert) => safeStr(cert).toLowerCase()).join(" ")
    : safeStr(certs).toLowerCase();

  let experienceText = "";
  if (Array.isArray(payload.experience)) {
    const chunks: string[] = [];
    for (const exp of payload.experience) {
      if (!exp || typeof exp !== "object") continue;
      const expObj = exp as Record<string, unknown>;
      const description = expObj.description;
      if (Array.isArray(description)) {
        chunks.push(description.map((d) => safeStr(d)).join(" "));
      } else {
        chunks.push(safeStr(description));
      }
      chunks.push(safeStr(expObj.jobTitle));
      chunks.push(safeStr(expObj.companyName));
    }
    experienceText = chunks.join(" ").toLowerCase();
  } else {
    experienceText = safeStr(payload.experience).toLowerCase();
  }

  const educationText = safeStr(payload.education).toLowerCase();
  const foundationalText = `${skillsText} ${projectsText} ${experienceText} ${educationText}`;

  for (const skill of SKILLS_TO_ASSESS) {
    let score = 0;
    const skillLower = skill.toLowerCase();
    const pattern =
      skill === ".Net"
        ? /(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])/i
        : new RegExp(`\\b${skillLower.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");

    if (hasRegexMatch(pattern, foundationalText)) score += 10;
    if (hasRegexMatch(pattern, experienceText)) score += 30;
    if (hasRegexMatch(pattern, projectsText)) score += 20;
    if (hasRegexMatch(pattern, certsText)) score += 20;

    scores[`${skill}_Probability`] = score;
  }

  return scores;
};

export const classifyAndFormatProjectsFromAi = (projectsRaw: unknown): Record<string, string> => {
  const internalTitles: string[] = [];
  const internalTechs: string[] = [];
  const externalTitles: string[] = [];
  const externalTechs: string[] = [];

  const projects = Array.isArray(projectsRaw) ? projectsRaw : [];

  const pickFirstString = (values: unknown[]): string => {
    for (const value of values) {
      const normalized = safeStr(value).trim();
      if (normalized) return normalized;
    }
    return "";
  };

  const normalizeTechStack = (projectObj: Record<string, unknown>): string => {
    const candidates = [
      projectObj.techStack,
      projectObj.techstack,
      projectObj.tech_stack,
      projectObj.technologies,
      projectObj.technology,
      projectObj.stack,
      projectObj.tools,
    ];

    for (const candidate of candidates) {
      if (Array.isArray(candidate)) {
        const joined = candidate
          .map((item) => safeStr(item).trim())
          .filter(Boolean)
          .join(", ");
        if (joined) return joined;
        continue;
      }

      const value = safeStr(candidate).trim();
      if (value) return value;
    }

    return "";
  };

  const normalizeClassification = (projectObj: Record<string, unknown>): "internal" | "external" => {
    const raw = pickFirstString([
      projectObj.classification,
      projectObj.projectClassification,
      projectObj.project_classification,
      projectObj.type,
      projectObj.projectType,
      projectObj.project_type,
      projectObj.source,
    ]).toLowerCase();

    return raw.includes("internal") ? "internal" : "external";
  };

  for (const project of projects) {
    if (typeof project === "string") {
      const title = safeStr(project).trim();
      if (title) {
        externalTitles.push(title);
      }
      continue;
    }

    if (!project || typeof project !== "object") continue;
    const projectObj = project as Record<string, unknown>;

    const title = pickFirstString([
      projectObj.title,
      projectObj.projectTitle,
      projectObj.project_title,
      projectObj.name,
      projectObj.projectName,
      projectObj.project_name,
    ]);
    if (!title) continue;

    const techStack = normalizeTechStack(projectObj);
    const classification = normalizeClassification(projectObj);

    if (classification === "internal") {
      internalTitles.push(title);
      if (techStack) {
        internalTechs.push(techStack);
      }
    } else {
      externalTitles.push(title);
      if (techStack) {
        externalTechs.push(techStack);
      }
    }
  }

  return {
    "Internal Project Title": internalTitles.join("\n"),
    "Internal Projects Techstacks": internalTechs.join("\n"),
    "External Project Title": externalTitles.join("\n"),
    "External Projects Techstacks": externalTechs.join("\n"),
  };
};

const asNumber = (value: unknown): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const asInt = (value: unknown): number => Math.trunc(asNumber(value));

const containsSearchTerm = (row: RowResult, term: string): boolean => {
  const searchableColumns = ["Skills", "Overall Remarks", "Internal Project Title", "External Project Title"];
  const combined = searchableColumns.map((column) => safeStr(row[column])).join(" ").toLowerCase();
  return combined.includes(term.toLowerCase());
};

export const applyFilters = (rows: RowResult[], filters: FilterQuery): RowResult[] => {
  return rows.filter((row) => {
    if (filters.priorityBands && filters.priorityBands.length > 0) {
      const band = safeStr(row["Priority Band"]);
      if (!filters.priorityBands.includes(band)) {
        return false;
      }
    }

    if (filters.searchTerm && !containsSearchTerm(row, filters.searchTerm)) {
      return false;
    }

    const numericRanges: Array<[string, number | undefined, number | undefined]> = [
      ["Overall Probability", filters.overallMin, filters.overallMax],
      ["Skills Probability", filters.skillsMin, filters.skillsMax],
      ["Experience Probability", filters.expMin, filters.expMax],
      ["Projects Probability", filters.projectsMin, filters.projectsMax],
      ["Other Probability", filters.otherMin, filters.otherMax],
    ];

    for (const [column, min, max] of numericRanges) {
      if (min === undefined && max === undefined) continue;
      if (!(column in row)) continue;
      const numeric = asNumber(row[column]);
      if (min !== undefined && numeric < min) return false;
      if (max !== undefined && numeric > max) return false;
    }

    if (filters.onlyInternal && asInt(row["Internal Projects Count"]) <= 0) {
      return false;
    }

    if (filters.onlyExternal && asInt(row["External Projects Count"]) <= 0) {
      return false;
    }

    if (filters.minTotalProjects !== undefined && asInt(row["Total Projects Count"]) < filters.minTotalProjects) {
      return false;
    }

    if (filters.minInternalProjects !== undefined && asInt(row["Internal Projects Count"]) < filters.minInternalProjects) {
      return false;
    }

    if (filters.minExternalProjects !== undefined && asInt(row["External Projects Count"]) < filters.minExternalProjects) {
      return false;
    }

    return true;
  });
};

const toNumberColumns = (rows: RowResult[], columns: string[]): RowResult[] => {
  return rows.map((row) => {
    const next = { ...row };
    for (const column of columns) {
      if (column in next) {
        next[column] = asNumber(next[column]);
      }
    }
    return next;
  });
};

const toIntColumns = (rows: RowResult[], columns: string[]): RowResult[] => {
  return rows.map((row) => {
    const next = { ...row };
    for (const column of columns) {
      if (column in next) {
        next[column] = asInt(next[column]);
      }
    }
    return next;
  });
};

const priorityIndex = (value: string): number => {
  const idx = PRIORITY_ORDER.indexOf(value as (typeof PRIORITY_ORDER)[number]);
  return idx === -1 ? PRIORITY_ORDER.length : idx;
};

export const orderAndSelectColumns = (
  rows: RowResult[],
  mode: string,
  shortlistingMode: string,
): { rows: RowResult[]; columns: string[]; fileName: string } => {
  const timestamp = dayjs().format("YYYYMMDD_HHmmss");

  if (mode === "shortlisting") {
    const numericRows = toNumberColumns(rows, [
      "Overall Probability",
      "Projects Probability",
      "Skills Probability",
      "Experience Probability",
      "Other Probability",
    ]);
    const withCounts = toIntColumns(numericRows, [
      "Total Projects Count",
      "Internal Projects Count",
      "External Projects Count",
    ]);

    const baseDisplayCols = [
      "User ID",
      "Resume Link",
      "Overall Probability",
      "Overall Remarks",
      "Projects Probability",
      "Projects Remarks",
      "Skills Probability",
      "Skills Remarks",
      "Experience Probability",
      "Experience Remarks",
      "Other Probability",
      "Other Remarks",
      "Total Projects Count",
      "Internal Projects Count",
      "External Projects Count",
      "Internal Project Title",
      "Internal Projects Techstacks",
      "External Project Title",
      "External Projects Techstacks",
    ];

    let ordered: RowResult[];
    let columns = [...baseDisplayCols];

    if (shortlistingMode === "Priority Wise (P1 / P2 / P3 Bands)") {
      columns = [...baseDisplayCols];
      columns.splice(3, 0, "Priority Band");
      columns.push("Company Name", "Analysis Datetime");

      ordered = [...withCounts].sort((a, b) => {
        const bandCompare = priorityIndex(safeStr(a["Priority Band"])) - priorityIndex(safeStr(b["Priority Band"]));
        if (bandCompare !== 0) return bandCompare;
        return asNumber(b["Overall Probability"]) - asNumber(a["Overall Probability"]);
      });
    } else {
      columns.push("Company Name", "Analysis Datetime");
      ordered = [...withCounts].sort((a, b) => asNumber(b["Overall Probability"]) - asNumber(a["Overall Probability"]));
    }

    const projected = ordered.map((row) =>
      Object.fromEntries(columns.map((column) => [column, row[column] ?? ""])) as RowResult,
    );

    return {
      rows: projected,
      columns,
      fileName: `resume_shortlist_${shortlistingMode.replaceAll(" ", "_").toLowerCase()}_${timestamp}.csv`,
    };
  }

  if (mode === "Internal Projects Matching") {
    const withCounts = toIntColumns(rows, [
      "Total Projects Count",
      "Internal Projects Count",
      "External Projects Count",
    ]);

    const columns = [...INTERNAL_MATCHING_COLUMNS, "Company Name", "Analysis Datetime"];
    const projected = withCounts.map((row) =>
      Object.fromEntries(columns.map((column) => [column, row[column] ?? ""])) as RowResult,
    );

    return {
      rows: projected,
      columns,
      fileName: `resume_analysis_${mode.replaceAll(" ", "_").toLowerCase()}_${timestamp}.csv`,
    };
  }

  const withSkillNumbers = toIntColumns(rows, [...SKILL_COLUMNS]);

  const finalColumns = [
    "User ID",
    "Resume Link",
    ...[
      "Full Name",
      "Mobile Number",
      "Email ID",
      "LinkedIn Link",
      "GitHub Link",
      "GitHub Repo Count",
      "Other Links",
      "City",
      "State",
      "Years of IT Experience",
      "Years of Non-IT Experience",
      "Highest Education Institute Name",
      "Skills",
    ],
    ...SKILL_COLUMNS,
    ...DETAILED_EDUCATION_COLUMNS,
    ...PROJECT_COLUMNS,
    ...EXPERIENCE_COLUMNS,
    ...OTHER_COLUMNS,
    "Company Name",
    "Analysis Datetime",
  ];

  const projected = withSkillNumbers.map((row) =>
    Object.fromEntries(finalColumns.map((column) => [column, row[column] ?? ""])) as RowResult,
  );

  return {
    rows: projected,
    columns: finalColumns,
    fileName: `resume_analysis_${mode.replaceAll(" ", "_").toLowerCase()}_${timestamp}.csv`,
  };
};

export const toCsvBuffer = (rows: RowResult[], columns: string[]): Buffer => {
  const records = rows.map((row) => columns.map((column) => row[column] ?? ""));
  const csv = stringify([columns, ...records]);
  return Buffer.from(csv, "utf8");
};

export const nowTimestamp = (): string => dayjs().format("YYYY-MM-DD HH:mm:ss");
