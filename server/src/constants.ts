export const MISTRAL_MODEL = "mistral-medium-latest";
export const MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions";

export const OPENAI_MODEL = "gpt-4o-mini";
export const OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions";

export const GSHEET_NAME = "AI Resume Analysis Results";

export const SHORTLISTING_MODES = [
  "Probability Wise (Default)",
  "Priority Wise (P1 / P2 / P3 Bands)",
] as const;

export const ANALYSIS_TYPES = [
  "All Data",
  "Personal Details",
  "Skills & Projects",
  "Internal Projects Matching",
] as const;

export const SKILLS_TO_ASSESS = [
  "JavaScript",
  "Python",
  "Node",
  "React",
  "Java",
  "Springboot",
  "DSA",
  "AI",
  "ML",
  "PHP",
  ".Net",
  "Testing",
  "AWS",
  "Django",
  "PowerBI",
  "Tableau",
] as const;

export const SKILL_COLUMNS = SKILLS_TO_ASSESS.map((skill) => `${skill}_Probability`);

export const SHORTLIST_BASE_COLUMNS = [
  "User ID",
  "Resume Link",
  "Company Name",
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
  "Internal Project Title",
  "Internal Projects Techstacks",
  "External Project Title",
  "External Projects Techstacks",
  "Total Projects Count",
  "Internal Projects Count",
  "External Projects Count",
] as const;

export const DETAILED_EDUCATION_COLUMNS = [
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
] as const;

export const ALL_DATA_FINAL_BASE_COLUMNS = [
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
] as const;

export const PROJECT_COLUMNS = [
  "Total Projects Count",
  "Internal Projects Count",
  "External Projects Count",
  "Internal Project Title",
  "Internal Projects Techstacks",
  "External Project Title",
  "External Projects Techstacks",
] as const;

export const EXPERIENCE_COLUMNS = [
  "Latest Experience Company Name",
  "Latest Experience Job Title",
  "Latest Experience Start Date",
  "Latest Experience End Date",
  "Currently Working? (Yes/No)",
] as const;

export const OTHER_COLUMNS = ["Certifications", "Awards", "Achievements"] as const;

export const INTERNAL_MATCHING_COLUMNS = [
  "User ID",
  "Resume Link",
  "Total Projects Count",
  "Internal Projects Count",
  "External Projects Count",
  "Internal Project Titles",
  "Internal Project Techstacks",
  "External Project Titles",
  "External Project Techstacks",
] as const;

export const PRIORITY_ORDER = ["P1", "P2", "P3", "Not Shortlisted"] as const;
