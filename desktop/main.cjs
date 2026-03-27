const { app, BrowserWindow, dialog, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");
const { pathToFileURL } = require("node:url");
const dotenv = require("dotenv");

const DEFAULT_PORT = 4010;
const SERVER_READY_TIMEOUT_MS = 60_000;
const SERVER_RETRY_DELAY_MS = 500;
const AUTO_UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const UPDATE_STATUS_CHECK_VISIBILITY_DELAY_MS = 1200;
const UPDATE_STATUS_AUTO_CLOSE_MS = 1800;
const UPDATE_COMMAND_POLL_INTERVAL_MS = 1500;

let serverPort = DEFAULT_PORT;
let mainWindow = null;
let serverBootPromise = null;
let hasCheckedForUpdates = false;
let autoUpdaterInstance = null;
let autoUpdateInterval = null;
let isUpdateCheckInFlight = false;
let isUpdateDownloadInProgress = false;
let hasPendingDownloadedUpdate = false;
let updateStatusWindow = null;
let updateStatusWindowPromise = null;
let updateStatusPayload = null;
let pendingUpdateStatusShowTimer = null;
let pendingUpdateStatusCloseTimer = null;
let hasDownloadHandlerAttached = false;
let desktopUpdateStatusPath = "";
let desktopUpdateCommandPath = "";
let updateCommandInterval = null;

app.setName("AI Resume Analyzer");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const showErrorDialog = (title, details) => {
  dialog.showErrorBox(title, details);
};

const MANAGED_ENV_KEYS = [
  "PORT",
  "VITE_API_BASE_URL",
  "OPENAI_API_KEY",
  "MISTRAL_API_KEY",
  "GOOGLE_SERVICE_ACCOUNT_JSON",
  "GOOGLE_SERVICE_ACCOUNT_PATH",
  ...Array.from({ length: 12 }, (_, idx) => `OPENAI_API_KEY_${idx + 1}`),
  ...Array.from({ length: 12 }, (_, idx) => `MISTRAL_API_KEY_${idx + 1}`),
];

const readEnvEntries = (envFilePath) => {
  if (!envFilePath || !fs.existsSync(envFilePath)) {
    return {};
  }

  try {
    return dotenv.parse(fs.readFileSync(envFilePath, "utf8"));
  } catch {
    return {};
  }
};

const getBundledEnvPath = () => {
  const templateCandidates = [
    path.join(app.getAppPath(), ".env"),
    path.join(process.resourcesPath, ".env"),
    path.join(app.getAppPath(), ".env.example"),
    path.join(process.resourcesPath, ".env.example"),
  ];

  return templateCandidates.find((candidate) => fs.existsSync(candidate)) || "";
};

const shouldRefreshUserEnvFromTemplate = (userEnvPath, templatePath) => {
  if (!userEnvPath || !templatePath || !fs.existsSync(userEnvPath) || !fs.existsSync(templatePath)) {
    return false;
  }

  const userEntries = readEnvEntries(userEnvPath);
  const templateEntries = readEnvEntries(templatePath);

  return MANAGED_ENV_KEYS.some((key) => {
    const templateValue = String(templateEntries[key] ?? "").trim();
    const userValue = String(userEntries[key] ?? "").trim();
    return Boolean(templateValue) && !userValue;
  });
};

const writeDesktopUpdateStatus = (status) => {
  if (!desktopUpdateStatusPath) return;

  const payload = {
    isDesktopApp: true,
    currentVersion: app.getVersion(),
    checkedAt: new Date().toISOString(),
    ...status,
  };

  fs.writeFileSync(desktopUpdateStatusPath, JSON.stringify(payload, null, 2), "utf8");
};

const consumeDesktopUpdateCommand = async () => {
  if (!desktopUpdateCommandPath || !fs.existsSync(desktopUpdateCommandPath)) return;

  let command = null;
  try {
    command = JSON.parse(fs.readFileSync(desktopUpdateCommandPath, "utf8"));
  } catch {
    command = null;
  } finally {
    try {
      fs.unlinkSync(desktopUpdateCommandPath);
    } catch {}
  }

  if (!command || typeof command.action !== "string") return;

  if (command.action === "installNow" && autoUpdaterInstance && hasPendingDownloadedUpdate) {
    writeDesktopUpdateStatus({
      phase: "installing",
      availableVersion: command.availableVersion || app.getVersion(),
      message: "Restarting to apply the downloaded update.",
    });
    setImmediate(() => autoUpdaterInstance.quitAndInstall(false, true));
    return;
  }

  if (command.action === "checkNow" && autoUpdaterInstance && !isUpdateCheckInFlight && !isUpdateDownloadInProgress) {
    isUpdateCheckInFlight = true;
    try {
      await autoUpdaterInstance.checkForUpdates();
    } catch (error) {
      isUpdateCheckInFlight = false;
      console.error("Manual update check failed:", error instanceof Error ? error.message : String(error));
    }
  }
};

const scheduleDesktopUpdateCommandPolling = () => {
  if (updateCommandInterval || !desktopUpdateCommandPath) return;

  updateCommandInterval = setInterval(() => {
    void consumeDesktopUpdateCommand().catch((error) => {
      console.error("Failed to process desktop update command:", error instanceof Error ? error.message : String(error));
    });
  }, UPDATE_COMMAND_POLL_INTERVAL_MS);

  updateCommandInterval.unref?.();
};

const clearPendingUpdateStatusShowTimer = () => {
  if (pendingUpdateStatusShowTimer) {
    clearTimeout(pendingUpdateStatusShowTimer);
    pendingUpdateStatusShowTimer = null;
  }
};

const clearPendingUpdateStatusCloseTimer = () => {
  if (pendingUpdateStatusCloseTimer) {
    clearTimeout(pendingUpdateStatusCloseTimer);
    pendingUpdateStatusCloseTimer = null;
  }
};

const setMainWindowProgress = (value) => {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  mainWindow.setProgressBar(value);
};

const formatBytes = (bytes) => {
  const numeric = Number(bytes);
  if (!Number.isFinite(numeric) || numeric <= 0) return "0 B";

  const units = ["B", "KB", "MB", "GB"];
  let value = numeric;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  const decimals = unitIndex === 0 ? 0 : value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(decimals)} ${units[unitIndex]}`;
};

const resolveUniqueDownloadPath = (fileName) => {
  const downloadsDir = app.getPath("downloads");
  const parsed = path.parse(fileName || "results.csv");
  const extension = parsed.ext || ".csv";
  const baseName = parsed.name || "results";

  let candidatePath = path.join(downloadsDir, `${baseName}${extension}`);
  let suffix = 1;

  while (fs.existsSync(candidatePath)) {
    candidatePath = path.join(downloadsDir, `${baseName} (${suffix})${extension}`);
    suffix += 1;
  }

  return candidatePath;
};

const attachAutoDownloadHandler = (windowInstance) => {
  if (!windowInstance || hasDownloadHandlerAttached) return;

  windowInstance.webContents.session.on("will-download", (_event, item) => {
    try {
      const savePath = resolveUniqueDownloadPath(item.getFilename());
      item.setSavePath(savePath);
    } catch (error) {
      console.error("Failed to prepare download path:", error instanceof Error ? error.message : String(error));
    }
  });

  hasDownloadHandlerAttached = true;
};

const createUpdateStatusHtml = () => `
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Updater</title>
    <style>
      :root {
        color-scheme: light;
        font-family: "Segoe UI", Arial, sans-serif;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background: linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
        color: #11203a;
      }

      .shell {
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
        box-sizing: border-box;
      }

      .card {
        width: 100%;
        max-width: 420px;
        background: #ffffff;
        border: 1px solid #d9e4ef;
        border-radius: 18px;
        box-shadow: 0 20px 45px rgba(17, 32, 58, 0.12);
        padding: 24px 24px 20px;
        box-sizing: border-box;
      }

      .header {
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .indicator {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: #e9f2ff;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 auto;
      }

      .spinner {
        width: 22px;
        height: 22px;
        border: 3px solid #c8d8ee;
        border-top-color: #0b7a75;
        border-radius: 50%;
        animation: spin 1s linear infinite;
      }

      .ready-mark {
        display: none;
        font-size: 20px;
        font-weight: 700;
        color: #0b7a75;
        line-height: 1;
      }

      .title {
        margin: 0;
        font-size: 22px;
        font-weight: 700;
      }

      .message {
        margin: 6px 0 0;
        font-size: 14px;
        color: #41506b;
        line-height: 1.45;
      }

      .detail {
        margin: 16px 0 0;
        font-size: 13px;
        color: #5d6b84;
        line-height: 1.45;
        min-height: 18px;
      }

      .progress-shell {
        margin-top: 18px;
      }

      .progress-meta {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: baseline;
        font-size: 13px;
        color: #41506b;
        margin-bottom: 8px;
      }

      .progress-bar {
        height: 10px;
        border-radius: 999px;
        overflow: hidden;
        background: #e6edf5;
      }

      .progress-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #0b7a75 0%, #1f9f8e 100%);
        transition: width 0.18s ease;
      }

      .hint {
        margin-top: 14px;
        font-size: 12px;
        color: #74839b;
      }

      .hidden {
        display: none;
      }

      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <div class="header">
          <div class="indicator">
            <div id="spinner" class="spinner"></div>
            <div id="readyMark" class="ready-mark">OK</div>
          </div>
          <div>
            <h1 id="title" class="title">Checking for updates</h1>
            <p id="message" class="message">Looking for a newer desktop version.</p>
          </div>
        </div>

        <p id="detail" class="detail"></p>

        <div id="progressShell" class="progress-shell hidden">
          <div class="progress-meta">
            <span id="progressLabel">Downloading update...</span>
            <strong id="progressPercent">0%</strong>
          </div>
          <div class="progress-bar">
            <div id="progressFill" class="progress-fill"></div>
          </div>
        </div>

        <div id="hint" class="hint">You can continue using the app while the update is prepared.</div>
      </div>
    </div>

    <script>
      window.renderUpdateStatus = function renderUpdateStatus(payload) {
        const titleEl = document.getElementById("title");
        const messageEl = document.getElementById("message");
        const detailEl = document.getElementById("detail");
        const progressShellEl = document.getElementById("progressShell");
        const progressLabelEl = document.getElementById("progressLabel");
        const progressPercentEl = document.getElementById("progressPercent");
        const progressFillEl = document.getElementById("progressFill");
        const hintEl = document.getElementById("hint");
        const spinnerEl = document.getElementById("spinner");
        const readyMarkEl = document.getElementById("readyMark");

        const normalized = payload || {};
        titleEl.textContent = normalized.title || "Updater";
        messageEl.textContent = normalized.message || "";
        detailEl.textContent = normalized.detail || "";
        hintEl.textContent = normalized.hint || "";

        const hasProgress = typeof normalized.progressPercent === "number";
        progressShellEl.classList.toggle("hidden", !hasProgress);
        progressLabelEl.textContent = normalized.progressLabel || "Downloading update...";
        progressPercentEl.textContent = hasProgress ? normalized.progressPercent.toFixed(0) + "%" : "";
        progressFillEl.style.width = hasProgress ? Math.max(0, Math.min(100, normalized.progressPercent)) + "%" : "0%";

        const isReady = Boolean(normalized.isReady);
        spinnerEl.style.display = isReady ? "none" : "block";
        readyMarkEl.style.display = isReady ? "block" : "none";
      };
    </script>
  </body>
</html>
`;

const getSerializedPayloadScript = (payload) => {
  const serialized = JSON.stringify(payload).replace(/</g, "\\u003c").replace(/>/g, "\\u003e").replace(/&/g, "\\u0026");
  return `window.renderUpdateStatus(${serialized});`;
};

const renderUpdateStatusWindow = (payload) => {
  if (!updateStatusWindow || updateStatusWindow.isDestroyed()) return;

  const run = () => updateStatusWindow.webContents.executeJavaScript(getSerializedPayloadScript(payload)).catch(() => {});
  if (updateStatusWindow.webContents.isLoading()) {
    updateStatusWindow.webContents.once("did-finish-load", run);
    return;
  }

  run();
};

const ensureUpdateStatusWindow = async () => {
  if (updateStatusWindow && !updateStatusWindow.isDestroyed()) {
    return updateStatusWindow;
  }

  if (updateStatusWindowPromise) {
    return updateStatusWindowPromise;
  }

  updateStatusWindowPromise = (async () => {
    const parentWindow = mainWindow && !mainWindow.isDestroyed() ? mainWindow : undefined;

    updateStatusWindow = new BrowserWindow({
      width: 460,
      height: 250,
      resizable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      show: false,
      skipTaskbar: true,
      autoHideMenuBar: true,
      parent: parentWindow,
      backgroundColor: "#eef5fb",
      webPreferences: {
        contextIsolation: true,
        sandbox: true,
      },
    });

    updateStatusWindow.on("closed", () => {
      updateStatusWindow = null;
    });

    await updateStatusWindow.loadURL(`data:text/html;charset=UTF-8,${encodeURIComponent(createUpdateStatusHtml())}`);
    return updateStatusWindow;
  })().finally(() => {
    updateStatusWindowPromise = null;
  });

  return updateStatusWindowPromise;
};

const showUpdateStatus = async (payload, options = {}) => {
  updateStatusPayload = payload;
  clearPendingUpdateStatusCloseTimer();

  const windowInstance = await ensureUpdateStatusWindow();
  if (!windowInstance || windowInstance.isDestroyed()) return;

  renderUpdateStatusWindow(payload);

  if (!windowInstance.isVisible()) {
    windowInstance.show();
  }

  if (typeof options.autoCloseMs === "number" && options.autoCloseMs > 0) {
    pendingUpdateStatusCloseTimer = setTimeout(() => {
      if (updateStatusWindow && !updateStatusWindow.isDestroyed()) {
        updateStatusWindow.close();
      }
      updateStatusPayload = null;
      setMainWindowProgress(-1);
      pendingUpdateStatusCloseTimer = null;
    }, options.autoCloseMs);
  }
};

const closeUpdateStatus = () => {
  clearPendingUpdateStatusShowTimer();
  clearPendingUpdateStatusCloseTimer();
  updateStatusPayload = null;
  setMainWindowProgress(-1);

  if (updateStatusWindow && !updateStatusWindow.isDestroyed()) {
    updateStatusWindow.close();
  }
};

const scheduleCheckingForUpdatesStatus = () => {
  clearPendingUpdateStatusShowTimer();
  pendingUpdateStatusShowTimer = setTimeout(() => {
    pendingUpdateStatusShowTimer = null;
    setMainWindowProgress(2);
    void showUpdateStatus({
      title: "Checking for updates",
      message: "Looking for a newer desktop version.",
      detail: "If an update is available, the app will download it here.",
      hint: "You can continue using the app while we check.",
    });
  }, UPDATE_STATUS_CHECK_VISIBILITY_DELAY_MS);
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
  const templatePath = getBundledEnvPath();

  if (fs.existsSync(userDataEnvPath)) {
    if (templatePath && shouldRefreshUserEnvFromTemplate(userDataEnvPath, templatePath)) {
      fs.copyFileSync(templatePath, userDataEnvPath);
    }
    return userDataEnvPath;
  }

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
  desktopUpdateStatusPath = path.join(app.getPath("userData"), "desktop-update-status.json");
  desktopUpdateCommandPath = path.join(app.getPath("userData"), "desktop-update-command.json");
  process.env.DESKTOP_UPDATE_STATUS_PATH = desktopUpdateStatusPath;
  process.env.DESKTOP_UPDATE_COMMAND_PATH = desktopUpdateCommandPath;
  process.env.DESKTOP_APP_VERSION = app.getVersion();
  writeDesktopUpdateStatus({ phase: "idle", message: "" });
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
  attachAutoDownloadHandler(mainWindow);

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

const scheduleAutoUpdateChecks = () => {
  if (autoUpdateInterval) {
    clearInterval(autoUpdateInterval);
  }

  autoUpdateInterval = setInterval(() => {
    if (!autoUpdaterInstance) return;
    if (isUpdateCheckInFlight || isUpdateDownloadInProgress || hasPendingDownloadedUpdate) return;

    isUpdateCheckInFlight = true;
    void autoUpdaterInstance.checkForUpdates().catch((error) => {
      isUpdateCheckInFlight = false;
      console.error("Auto-update check failed:", error instanceof Error ? error.message : String(error));
    });
  }, AUTO_UPDATE_CHECK_INTERVAL_MS);

  autoUpdateInterval.unref?.();
};

const setupAutoUpdates = () => {
  if (!app.isPackaged || hasCheckedForUpdates) return;
  hasCheckedForUpdates = true;

  try {
    const { autoUpdater } = require("electron-updater");
    autoUpdaterInstance = autoUpdater;
    autoUpdater.autoDownload = true;
    autoUpdater.autoInstallOnAppQuit = true;
    autoUpdater.fullChangelog = true;
    scheduleDesktopUpdateCommandPolling();

    autoUpdater.on("checking-for-update", () => {
      isUpdateCheckInFlight = true;
      writeDesktopUpdateStatus({
        phase: "checking",
        message: "Checking for updates.",
      });
    });

    autoUpdater.on("update-available", (info) => {
      isUpdateCheckInFlight = false;
      isUpdateDownloadInProgress = true;
      writeDesktopUpdateStatus({
        phase: "available",
        availableVersion: info?.version ?? "",
        progressPercent: 0,
        message: `Version ${info?.version ?? "latest"} is downloading in the background.`,
      });
      console.info(`Auto-update available: ${info?.version ?? "unknown version"}`);
    });

    autoUpdater.on("update-not-available", () => {
      isUpdateCheckInFlight = false;
      isUpdateDownloadInProgress = false;
      hasPendingDownloadedUpdate = false;
      writeDesktopUpdateStatus({
        phase: "idle",
        message: "",
      });
    });

    autoUpdater.on("download-progress", (progress) => {
      isUpdateDownloadInProgress = true;
      const percent = Math.max(0, Math.min(100, Number(progress?.percent ?? 0)));
      writeDesktopUpdateStatus({
        phase: "downloading",
        availableVersion: progress?.version ?? "",
        message: `${formatBytes(progress?.transferred)} of ${formatBytes(progress?.total)} downloaded at ${formatBytes(progress?.bytesPerSecond)}/s`,
        progressPercent: percent,
      });
    });

    autoUpdater.on("error", (error) => {
      isUpdateCheckInFlight = false;
      isUpdateDownloadInProgress = false;
      writeDesktopUpdateStatus({
        phase: "error",
        message: error instanceof Error ? error.message : String(error),
      });
      console.error("Auto-update error:", error instanceof Error ? error.message : String(error));
    });

    autoUpdater.on("update-downloaded", (info) => {
      isUpdateCheckInFlight = false;
      isUpdateDownloadInProgress = false;
      hasPendingDownloadedUpdate = true;
      writeDesktopUpdateStatus({
        phase: "downloaded",
        availableVersion: info?.version ?? "",
        progressPercent: 100,
        message: `Version ${info?.version ?? "latest"} is ready to install.`,
      });
    });

    scheduleAutoUpdateChecks();

    isUpdateCheckInFlight = true;
    void autoUpdater.checkForUpdates().catch((error) => {
      isUpdateCheckInFlight = false;
      writeDesktopUpdateStatus({
        phase: "error",
        message: error instanceof Error ? error.message : String(error),
      });
      console.error("Initial auto-update check failed:", error instanceof Error ? error.message : String(error));
    });
  } catch (error) {
    writeDesktopUpdateStatus({
      phase: "error",
      message: error instanceof Error ? error.message : String(error),
    });
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

app.on("before-quit", () => {
  if (autoUpdateInterval) {
    clearInterval(autoUpdateInterval);
    autoUpdateInterval = null;
  }
  if (updateCommandInterval) {
    clearInterval(updateCommandInterval);
    updateCommandInterval = null;
  }
  clearPendingUpdateStatusShowTimer();
  clearPendingUpdateStatusCloseTimer();
});
