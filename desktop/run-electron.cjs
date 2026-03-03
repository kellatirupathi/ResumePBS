const { spawn } = require("node:child_process");
const path = require("node:path");

delete process.env.ELECTRON_RUN_AS_NODE;

const electronBinary = require("electron");
const projectRoot = path.resolve(__dirname, "..");

const child = spawn(electronBinary, ["."], {
  cwd: projectRoot,
  stdio: "inherit",
  env: process.env,
});

child.on("error", (error) => {
  console.error("Failed to start Electron:", error);
  process.exit(1);
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});

