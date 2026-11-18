import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  /* onnxruntime-node is a native addon — keep it external to the server bundle */
  serverExternalPackages: ["onnxruntime-node"],
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
