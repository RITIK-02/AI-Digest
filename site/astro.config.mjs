import { defineConfig } from "astro/config";

// Everything is precomputed — static output, no adapter, no SSR.
export default defineConfig({
  output: "static",
});
