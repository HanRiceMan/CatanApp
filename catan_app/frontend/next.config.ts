import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 同じリポジトリにある test の依存関係と混ざらないようにする。
  turbopack: { root: process.cwd() },
  // 開発サーバーのJavaScriptを、このPCのLANアドレスから読み込めるようにする。
  allowedDevOrigins: ["localhost", "127.0.0.1", "192.168.0.128", "100.100.37.54"],
  // スマホは3002番だけに接続し、Python APIへはNext.jsがPC内部で中継する。
  async rewrites() {
    return [{
      source: "/python-api/:path*",
      destination: "http://127.0.0.1:8003/:path*",
    }];
  },
};

export default nextConfig;
