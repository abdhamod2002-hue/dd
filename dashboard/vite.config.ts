import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        // Docker Compose overrides this via VITE_PROXY_TARGET=http://backend:8000
        // so the containerized Vite dev server can reach the backend service.
        // Host-mode dev keeps the default localhost:8000.
        target: process.env.VITE_PROXY_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
