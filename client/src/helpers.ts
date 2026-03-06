import type { FilterState, ShortlistingMode } from "./types";

export type TableRow = Record<string, string | number | boolean | null>;

const asNumber = (value: unknown): number => {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
};

const asInt = (value: unknown): number => Math.trunc(asNumber(value));

export const priorityOrder = ["P1", "P2", "P3", "Not Shortlisted"];

const getPriorityIndex = (value: string): number => {
  const index = priorityOrder.indexOf(value);
  return index === -1 ? priorityOrder.length : index;
};

export const sortAndOrderRows = (rows: TableRow[], mode: string, shortlistingMode: ShortlistingMode): { rows: TableRow[]; columns: string[] } => {
  if (mode === "shortlisting") {
    if (shortlistingMode === "Sectionwise") {
      const columns = [
        "User ID",
        "Resume Link",
        "Skills",
        "Skills Pro",
        "Projects",
        "Projects Pro",
        "Experience",
        "Experience Pro",
        "Certifications",
        "Certification Pro",
        "Education",
        "Education Pro",
        "Summary or Overview",
        "Summary Pro",
        "Company Name",
        "Analysis Datetime",
      ];

      const ordered = [...rows].sort((a, b) => asNumber(b["Overall Probability"]) - asNumber(a["Overall Probability"]));
      return { rows: ordered, columns };
    }

    const baseColumns = [
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

    let columns = [...baseColumns];
    let ordered = [...rows];

    if (shortlistingMode === "Priority Wise (P1 / P2 / P3 Bands)") {
      columns.splice(3, 0, "Priority Band");
      ordered.sort((a, b) => {
        const byBand = getPriorityIndex(String(a["Priority Band"] ?? "")) - getPriorityIndex(String(b["Priority Band"] ?? ""));
        if (byBand !== 0) return byBand;
        return asNumber(b["Overall Probability"]) - asNumber(a["Overall Probability"]);
      });
    } else {
      ordered.sort((a, b) => asNumber(b["Overall Probability"]) - asNumber(a["Overall Probability"]));
    }

    columns = [...columns, "Company Name", "Analysis Datetime"].filter((column, index, arr) => arr.indexOf(column) === index);

    return {
      rows: ordered,
      columns,
    };
  }

  if (mode === "Internal Projects Matching") {
    const columns = [
      "User ID",
      "Resume Link",
      "Total Projects Count",
      "Internal Projects Count",
      "External Projects Count",
      "Internal Project Titles",
      "Internal Project Techstacks",
      "External Project Titles",
      "External Project Techstacks",
      "Company Name",
      "Analysis Datetime",
    ];

    return { rows: [...rows], columns };
  }

  const columns = [
    "User ID",
    "Resume Link",
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
    "JavaScript_Probability",
    "Python_Probability",
    "Node_Probability",
    "React_Probability",
    "Java_Probability",
    "Springboot_Probability",
    "DSA_Probability",
    "AI_Probability",
    "ML_Probability",
    "PHP_Probability",
    ".Net_Probability",
    "Testing_Probability",
    "AWS_Probability",
    "Django_Probability",
    "PowerBI_Probability",
    "Tableau_Probability",
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
    "Total Projects Count",
    "Internal Projects Count",
    "External Projects Count",
    "Internal Project Title",
    "Internal Projects Techstacks",
    "External Project Title",
    "External Projects Techstacks",
    "Latest Experience Company Name",
    "Latest Experience Job Title",
    "Latest Experience Start Date",
    "Latest Experience End Date",
    "Currently Working? (Yes/No)",
    "Certifications",
    "Awards",
    "Achievements",
    "Company Name",
    "Analysis Datetime",
  ];

  return { rows: [...rows], columns };
};

export const defaultFilters: FilterState = {
  priorityBands: [],
  searchTerm: "",
  overallRange: [0, 100],
  skillsRange: [0, 100],
  experienceRange: [0, 100],
  projectsRange: [0, 100],
  otherRange: [0, 100],
  onlyInternal: false,
  onlyExternal: false,
  minTotalProjects: 0,
  minInternalProjects: 0,
  minExternalProjects: 0,
};

export const applyFilters = (rows: TableRow[], filters: FilterState): TableRow[] => {
  return rows.filter((row) => {
    if (filters.priorityBands.length > 0) {
      const band = String(row["Priority Band"] ?? "");
      if (!filters.priorityBands.includes(band)) return false;
    }

    if (filters.searchTerm) {
      const text = ["Skills", "Overall Remarks", "Internal Project Title", "External Project Title"]
        .map((key) => String(row[key] ?? ""))
        .join(" ")
        .toLowerCase();
      if (!text.includes(filters.searchTerm.toLowerCase())) return false;
    }

    const rangeChecks: Array<[string, [number, number]]> = [
      ["Overall Probability", filters.overallRange],
      ["Skills Probability", filters.skillsRange],
      ["Experience Probability", filters.experienceRange],
      ["Projects Probability", filters.projectsRange],
      ["Other Probability", filters.otherRange],
    ];

    for (const [column, [min, max]] of rangeChecks) {
      if (!(column in row)) continue;
      const value = asNumber(row[column]);
      if (value < min || value > max) return false;
    }

    if (filters.onlyInternal && asInt(row["Internal Projects Count"]) <= 0) {
      return false;
    }
    if (filters.onlyExternal && asInt(row["External Projects Count"]) <= 0) {
      return false;
    }

    if (asInt(row["Total Projects Count"]) < filters.minTotalProjects) {
      return false;
    }
    if (asInt(row["Internal Projects Count"]) < filters.minInternalProjects) {
      return false;
    }
    if (asInt(row["External Projects Count"]) < filters.minExternalProjects) {
      return false;
    }

    return true;
  });
};

export const toQueryParamsFromFilters = (filters: FilterState): Record<string, string | number | boolean | undefined> => ({
  priorityBands: filters.priorityBands.length ? filters.priorityBands.join(",") : undefined,
  searchTerm: filters.searchTerm || undefined,
  overallMin: filters.overallRange[0],
  overallMax: filters.overallRange[1],
  skillsMin: filters.skillsRange[0],
  skillsMax: filters.skillsRange[1],
  expMin: filters.experienceRange[0],
  expMax: filters.experienceRange[1],
  projectsMin: filters.projectsRange[0],
  projectsMax: filters.projectsRange[1],
  otherMin: filters.otherRange[0],
  otherMax: filters.otherRange[1],
  onlyInternal: filters.onlyInternal || undefined,
  onlyExternal: filters.onlyExternal || undefined,
  minTotalProjects: filters.minTotalProjects || undefined,
  minInternalProjects: filters.minInternalProjects || undefined,
  minExternalProjects: filters.minExternalProjects || undefined,
});
