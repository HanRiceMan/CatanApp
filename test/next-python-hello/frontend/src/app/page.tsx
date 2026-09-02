"use client";

import { FormEvent, useState } from "react";

type HelloResponse = {
  message: string;
  received_name: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const [name, setName] = useState("カタン学習者");
  const [result, setResult] = useState<HelloResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function callPythonApi(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/hello?name=${encodeURIComponent(name || "ゲスト")}`,
      );

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: HelloResponse = await response.json();
      setResult(data);
    } catch {
      setError(
        "Python API に接続できません。backend 側で uvicorn が起動しているか確認してください。",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main>
      <p className="eyebrow">Next.js → FastAPI → JSON</p>
      <h1>Python API 接続サンプル</h1>
      <p className="description">
        入力した名前を Next.js から Python に送り、Python が作ったメッセージを表示します。
      </p>

      <form onSubmit={callPythonApi}>
        <label htmlFor="name">名前</label>
        <input
          id="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="例：太郎"
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? "通信中…" : "Python API を呼ぶ"}
        </button>
      </form>

      {result && (
        <section className="result" aria-live="polite">
          <h2>Python からの返答</h2>
          <p>{result.message}</p>
          <code>{JSON.stringify(result, null, 2)}</code>
        </section>
      )}

      {error && <p className="error">{error}</p>}
    </main>
  );
}
