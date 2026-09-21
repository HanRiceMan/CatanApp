import type { Metadata } from "next";
import "./globals.css";
import "./layout-overrides.css";

export const metadata: Metadata = {
  title: "CATAN | ローカルテーブル",
  description: "4人でひとつの島を囲む、カタン学習プロジェクト。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ja"><body>{children}</body></html>;
}
