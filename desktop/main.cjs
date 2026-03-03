const { app, BrowserWindow, dialog, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { pathToFileURL } = require("node:url");

const DEFAULT_PORT = 4010;
const SERVER_READY_TIMEOUT_MS = 60_000;
const SERVER_RETRY_DELAY_MS = 500;

let serverPort = DEFAULT_PORT;
let mainWindow = null;
let serverBootPromise = null;
let hasCheckedForUpdates = false;

app.setName("AI Resume Analyzer");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const showErrorDialog = (title, details) => {
  dialog.showErrorBox(title, details);
};

process.on("uncaughtException", (error) => {
  console.error("Uncaught exception:", error);
  showErrorDialog("Desktop Runtime Error", error instanceof Error ? error.message : String(error));
});

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled promise rejection:", reason);
});

const readPortFromEnvFile = (envFilePath) => {
  if (!envFilePath || !fs.existsSync(envFilePath)) return null;

  try {
    const raw = fs.readFileSync(envFilePath, "utf8");
    const match = raw.match(/^\s*PORT\s*=\s*"?(\d+)"?\s*$/m);
    if (!match) return null;

    const parsed = Number(match[1]);
    if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 65535) return null;
    return parsed;
  } catch {
    return null;
  }
};

const ensureUserEnvFile = () => {
  const userDataEnvPath = path.join(app.getPath("userData"), ".env");
  if (fs.existsSync(userDataEnvPath)) {
    return userDataEnvPath;
  }

  const templateCandidates = [
    path.join(app.getAppPath(), ".env"),
    path.join(process.resourcesPath, ".env"),
    path.join(app.getAppPath(), ".env.example"),
    path.join(process.resourcesPath, ".env.example"),
  ];

  const templatePath = templateCandidates.find((candidate) => fs.existsSync(candidate));
  if (templatePath) {
    fs.copyFileSync(templatePath, userDataEnvPath);
    return userDataEnvPath;
  }

  const fallbackContent = [
    `PORT=${DEFAULT_PORT}`,
    "OPENAI_API_KEY=",
    "MISTRAL_API_KEY=",
    "GOOGLE_SERVICE_ACCOUNT_JSON=",
    "GOOGLE_SERVICE_ACCOUNT_PATH=",
    `VITE_API_BASE_URL=http://localhost:${DEFAULT_PORT}`,
    "",
  ].join("\n");

  fs.writeFileSync(userDataEnvPath, fallbackContent, "utf8");
  return userDataEnvPath;
};

const configureRuntimeEnvironment = () => {
  const envFilePath = ensureUserEnvFile();
  process.env.APP_ENV_PATH = envFilePath;
  serverPort = readPortFromEnvFile(envFilePath) ?? DEFAULT_PORT;
};

const pingServerHealth = () =>
  new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: serverPort,
        path: "/api/health",
        method: "GET",
        timeout: 2_000,
      },
      (response) => {
        response.resume();
        if ((response.statusCode ?? 500) >= 200 && (response.statusCode ?? 500) < 300) {
          resolve();
          return;
        }
        reject(new Error(`Health check returned status ${response.statusCode ?? "unknown"}`));
      },
    );

    req.on("error", reject);
    req.on("timeout", () => req.destroy(new Error("Health check timed out")));
    req.end();
  });

const waitForServerReady = async () => {
  const deadline = Date.now() + SERVER_READY_TIMEOUT_MS;
  let lastError = null;

  while (Date.now() < deadline) {
    try {
      await pingServerHealth();
      return;
    } catch (error) {
      lastError = error;
      await sleep(SERVER_RETRY_DELAY_MS);
    }
  }

  throw new Error(
    `Backend server did not start on http://localhost:${serverPort} within ${SERVER_READY_TIMEOUT_MS / 1000}s.` +
      (lastError instanceof Error ? ` Last error: ${lastError.message}` : ""),
  );
};

const startInternalServer = async () => {
  if (serverBootPromise) {
    return serverBootPromise;
  }

  serverBootPromise = (async () => {
    configureRuntimeEnvironment();

    const serverEntryPath = path.join(app.getAppPath(), "server", "dist", "index.js");
    if (!fs.existsSync(serverEntryPath)) {
      throw new Error(
        `Missing backend build at ${serverEntryPath}. Run "npm run build --workspace server" before packaging.`,
      );
    }

    await import(pathToFileURL(serverEntryPath).href);
    await waitForServerReady();
  })();

  return serverBootPromise;
};

const createMainWindow = async () => {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 920,
    minWidth: 1200,
    minHeight: 720,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  mainWindow.setMenuBarVisibility(false);

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("did-fail-load", (_event, errorCode, errorDescription, validatedURL) => {
    showErrorDialog(
      "UI Load Failed",
      `Failed to load app UI.\nCode: ${errorCode}\nReason: ${errorDescription}\nURL: ${validatedURL}`,
    );
  });

  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    showErrorDialog("UI Process Crashed", `Reason: ${details.reason}`);
  });

  const devUrl = process.env.ELECTRON_START_URL;
  if (devUrl) {
    await mainWindow.loadURL(devUrl);
    return;
  }

  const clientEntryPath = path.join(app.getAppPath(), "client", "dist", "index.html");
  if (!fs.existsSync(clientEntryPath)) {
    throw new Error(
      `Missing frontend build at ${clientEntryPath}. Run "npm run build --workspace client" before packaging.`,
    );
  }

  await mainWindow.loadFile(clientEntryPath);
};

const setupAutoUpdates = () => {
  if (!app.isPackaged || hasCheckedForUpdates) return;
  hasCheckedForUpdates = true;

  try {
    const { autoUpdater } = require("electron-updater");
    autoUpdater.autoDownload = true;
    autoUpdater.autoInstallOnAppQuit = true;

    autoUpdater.on("error", (error) => {
      console.error("Auto-update error:", error instanceof Error ? error.message : String(error));
    });

    autoUpdater.on("update-downloaded", async () => {
      const result = await dialog.showMessageBox(mainWindow ?? undefined, {
        type: "info",
        buttons: ["Restart now", "Later"],
        defaultId: 0,
        cancelId: 1,
        title: "Update Ready",
        message: "A new update has been downloaded.",
        detail: "Restart the app now to apply the update.",
      });

      if (result.response === 0) {
        setImmediate(() => autoUpdater.quitAndInstall(false, true));
      }
    });

    void autoUpdater.checkForUpdatesAndNotify();
  } catch (error) {
    console.error("Auto-update setup failed:", error);
  }
};

app.whenReady().then(async () => {
  try {
    await startInternalServer();
    await createMainWindow();
    setupAutoUpdates();
  } catch (error) {
    showErrorDialog(
      "Desktop Startup Failed",
      error instanceof Error ? error.message : String(error),
    );
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    void createMainWindow().catch((error) => {
      showErrorDialog("Window Creation Failed", error instanceof Error ? error.message : String(error));
    });
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
