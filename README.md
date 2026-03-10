# AI Resume Analyzer (React + Node Migration)

This repository contains a strict-parity migration of the original Python + Streamlit resume analyzer to:

- Frontend: React + TypeScript + MUI
- Backend: Node.js + TypeScript + Express

The application keeps the same flows, fields, core logic, output columns, filtering, download behavior, provider routing, concurrency controls, and Google Sheets save behavior.

## Project Structure

- `client/`: React UI
- `server/`: Express API and processing pipeline
- `server/resources/INTERNAL_PROJECT_LIST.txt`: Internal project list used by AI classification

## Environment Variables

Copy `.env.example` to `.env` and fill values:

- `OPENAI_API_KEY`
- `MISTRAL_API_KEY` or `MISTRAL_API_KEY_1` .. `MISTRAL_API_KEY_12`
- `GOOGLE_SERVICE_ACCOUNT_JSON` or `GOOGLE_SERVICE_ACCOUNT_PATH`
- `VITE_API_BASE_URL` (defaults to `http://localhost:4010`)

## Install

```bash
npm install
```

## Run (Client + Server)

```bash
npm run dev
```

- Client: `http://localhost:5173`
- Server: `http://localhost:4010`

## Build

```bash
npm run build
```

## Desktop App (.exe)

Build a Windows installer locally:

```bash
npm run desktop:dist
```

Installer output:

- `release/*.exe`

Run desktop app locally (using built client + server dist):

```bash
npm run desktop:dev
```

Notes:

- Desktop app starts the same backend pipeline internally.
- On first run, app creates `%APPDATA%/AI Resume Analyzer/.env` if missing.
- That file is auto-copied from the bundled desktop env included at build time (`.env` if available, otherwise `.env.example`).
- End users do not need to create a separate `.env` manually when keys are bundled in the installer.
- Auto-update is enabled in packaged desktop builds and checks GitHub Releases on startup and periodically in the background.

## GitHub .exe Link (for sharing)

This repo includes GitHub Actions workflow:

- `.github/workflows/windows-desktop.yml`

How to publish downloadable `.exe` links:

1. Push code to GitHub.
2. Create and push a version tag (example `v1.0.0`).
3. Workflow runs `npm run desktop:publish` on Windows for tag builds.
4. `electron-builder` publishes the installer, `latest.yml`, and `*.blockmap` to the GitHub Release for that tag.
5. Installed desktop users receive the update automatically when the new version is available.
6. Share the GitHub Release asset link with others.

Auto-update requirement:

- Always publish a higher version tag (`v1.0.2` -> `v1.0.3`), never reuse same version/tag.

To bundle API keys in GitHub-built installers, configure repository secrets used by:

- `OPENAI_API_KEY`
- `MISTRAL_API_KEY` or `MISTRAL_API_KEY_1`..`MISTRAL_API_KEY_12`
- `GOOGLE_SERVICE_ACCOUNT_JSON` (optional)

## API Endpoints

- `GET /api/config`
- `POST /api/jobs`
- `GET /api/jobs/:jobId`
- `GET /api/jobs/:jobId/results`
- `GET /api/jobs/:jobId/download`

## Notes

- Google Sheets writing is best-effort; network/auth failures do not stop CSV availability.
- OCR fallback for scanned PDFs/images is best-effort and depends on local OCR/rendering support.
- Mistral key rotation is index-based round-robin across active keys (`i % numKeys`).
