import axios from "axios";
import { analyzeTextWithProvider } from "./ai.js";
import { createTempFilePath, cleanupTempFile, downloadAndIdentifyFile, extractTextAndUrlsFromPdf, extractTextFromImage } from "./extraction.js";
import { SKILL_COLUMNS } from "./constants.js";
import { calculateSkillProbabilities, classifyAndFormatProjectsFromAi, extractGithubUsername, formatMobileNumber, getHighestEducationInstitute, getLatestExperience, isPresentStr, relaxedJsonLoads, safeStr, sortLinks } from "./utils.js";
import type { AnalysisType, Provider, ResumeInputRow, RowResult } from "./types.js";

export interface WorkerContext {
  provider: Provider;
  apiKey: string;
  enableOcr: boolean;
  companyName: string;
  userRequirements: string;
  analysisType: AnalysisType;
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

const getProjectInstructionBlock = (internalProjectsString: string, analysisType?: AnalysisType): string => {
  if (internalProjectsString && analysisType !== "Personal Details") {
    return `
"projects": Analyze the resume for projects. For each project, extract its title and techStack. CRITICALLY, you must add a "classification" field. Classify a project as "Internal" if its title, description or context matches any project from the OFFICIAL INTERNAL PROJECTS LIST provided below. Otherwise, classify it as "External". Be flexible in your matching (e.g., 'Jobby App' in the list should match 'Jobby-app' or 'Jobby Application' in a resume).
OFFICIAL INTERNAL PROJECTS LIST: ${internalProjectsString}
Example Project Entry: { "title": "Jobby App", "techStack": ["React", "JS"], "classification": "Internal" }
`;
  }

  return `
"projects": [ { "title": "string", "techStack": ["list of tech keywords"], "classification": "External" } ]
`;
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
      const { text, urls } = await extractTextAndUrlsFromPdf(tempFilePath, enableOcr);
      return { text, clickableLinks: urls };
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
  const result = defaultShortlistResult(row, context.companyName);

  try {
    const { text: resumeText } = await loadResumeText(row["Resume link"], context.enableOcr);

    if (!resumeText.trim()) {
      throw new Error("Could not extract any text from the file.");
    }

    const textLower = resumeText.toLowerCase();
    const reqsLower = context.userRequirements.toLowerCase();
    let systemWarning = "";

    if (/\bjava\b/.test(reqsLower) && !/\bjava\b/.test(textLower)) {
      systemWarning +=
        "\n\n[SYSTEM WARNING]: The user explicitly requires 'Java' (the backend language). I have scanned the text and 'Java' appears to be MISSING as a standalone word. The text might contain 'JavaScript', but THAT IS NOT JAVA. Treat 'Java' as MISSING.";
    }

    const projectInstructionBlock = getProjectInstructionBlock(context.internalProjectsString);

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

---
**Required Criteria:**
${context.userRequirements}
---
**Resume Text:**
${resumeText}
---
`;

    const aiResponse = await analyzeTextWithProvider(prompt, context.provider, context.apiKey);
    const data = relaxedJsonLoads(aiResponse);

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

    const classified = classifyAndFormatProjectsFromAi(data.projects ?? []);
    Object.assign(result, classified);

    const internalTitles = safeStr(classified["Internal Project Title"]);
    const externalTitles = safeStr(classified["External Project Title"]);
    const internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
    const externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;

    result["Internal Projects Count"] = internalCount;
    result["External Projects Count"] = externalCount;
    result["Total Projects Count"] = internalCount + externalCount;
  } catch (error) {
    result["Overall Remarks"] = `Error: ${error instanceof Error ? error.message : String(error)}`;
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

  try {
    const { text: resumeText, clickableLinks } = await loadResumeText(row["Resume link"], context.enableOcr);

    if (!resumeText.trim()) {
      throw new Error("Could not extract any text from the file.");
    }

    const projectInstructionBlock = getProjectInstructionBlock(context.internalProjectsString, context.analysisType);

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
      prompt = `
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

**CRITICAL NOTE ON PROJECTS:**
Refer to the "projects" instruction above. You MUST use the OFFICIAL INTERNAL PROJECTS LIST to classify projects correctly.

Resume Text:
---
${resumeText}
---
`;
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

**CRITICAL NOTE ON PROJECTS:**
Refer to the "projects" instruction above. You MUST use the OFFICIAL INTERNAL PROJECTS LIST to classify projects correctly.

Resume Text:
---
${resumeText}
---
`;
    }

    const aiResponse = await analyzeTextWithProvider(prompt, context.provider, context.apiKey);
    aiResponseForDebug = aiResponse;
    const data = relaxedJsonLoads(aiResponse);

    if (typeof data !== "object" || Array.isArray(data)) {
      throw new Error(`AI returned non-dict data. Type: ${typeof data}`);
    }
    if (data.error) {
      throw new Error(safeStr(data.error));
    }

    const classifiedProjects = classifyAndFormatProjectsFromAi(data.projects ?? []);

    if (context.analysisType === "Internal Projects Matching") {
      const internalTitles = safeStr(classifiedProjects["Internal Project Title"]);
      const externalTitles = safeStr(classifiedProjects["External Project Title"]);
      const internalCount = internalTitles ? internalTitles.split(/\r?\n/).filter(Boolean).length : 0;
      const externalCount = externalTitles ? externalTitles.split(/\r?\n/).filter(Boolean).length : 0;

      result["Total Projects Count"] = internalCount + externalCount;
      result["Internal Projects Count"] = internalCount;
      result["External Projects Count"] = externalCount;
      result["Internal Project Titles"] = internalTitles;
      result["Internal Project Techstacks"] = safeStr(classifiedProjects["Internal Projects Techstacks"]);
      result["External Project Titles"] = externalTitles;
      result["External Project Techstacks"] = safeStr(classifiedProjects["External Projects Techstacks"]);
    } else {
      Object.assign(result, classifiedProjects);

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

    if (error instanceof SyntaxError) {
      console.debug("Malformed AI response", aiResponseForDebug);
    }
  }

  return result;
};
