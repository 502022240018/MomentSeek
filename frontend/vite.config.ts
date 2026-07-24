import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    allowedHosts: true,
    proxy: { "/api": { target: apiProxyTarget, changeOrigin: true } },
  },
});
