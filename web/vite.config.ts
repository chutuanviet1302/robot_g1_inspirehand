import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is written into the Python package, so `homehand serve` needs no Node.js.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../homehand/web_static", emptyOutDir: true, chunkSizeWarningLimit: 800 },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
