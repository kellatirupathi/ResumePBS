const fs = require("node:fs");
const path = require("node:path");

const projectRoot = path.resolve(__dirname, "..");
const sourceEnvPath = path.join(projectRoot, ".env");
const sourceExamplePath = path.join(projectRoot, ".env.example");
const outputPath = path.join(__dirname, ".env.runtime");

const copySource = fs.existsSync(sourceEnvPath) ? sourceEnvPath : sourceExamplePath;

if (!fs.existsSync(copySource)) {
  throw new Error("Neither .env nor .env.example exists. Cannot prepare desktop runtime env.");
}

fs.copyFileSync(copySource, outputPath);
console.log(`Prepared desktop env bundle from ${path.basename(copySource)} -> desktop/.env.runtime`);

