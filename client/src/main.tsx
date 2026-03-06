import React from "react";
import ReactDOM from "react-dom/client";
import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import App from "./App";
import BigQueryPage from "./BigQueryPage";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#0f766e",
    },
    secondary: {
      main: "#334155",
    },
    background: {
      default: "#eef2f7",
      paper: "#ffffff",
    },
  },
  shape: {
    borderRadius: 10,
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {window.location.pathname.replace(/\/+$/, "").toLowerCase() === "/bigquery" ? <BigQueryPage /> : <App />}
    </ThemeProvider>
  </React.StrictMode>,
);
