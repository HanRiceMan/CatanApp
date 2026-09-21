"use client";

import { useState, type SubmitEvent } from "react";

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
  const [isLampOn, setIsLampOn] = useState(false);
  const [isLamp2On, setIsLamp2On] = useState(false);
  const [loadingLamp, setLoadingLamp] = useState<1 | 2 | null>(null);
  const [lampError, setLampError] = useState("");

  async function checkLamp(lampId: 1 | 2) {
    setLoadingLamp(lampId);
    setLampError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/lamp`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: { is_on: boolean } = await response.json();
      // 同じAPIを使い、押したボタンに対応するランプだけ更新する。
      if (lampId === 1) {
        setIsLampOn(data.is_on);
      } else {
        setIsLamp2On(data.is_on);
      }
    } catch {
      setLampError(`ランプ${lampId}の状態を取得できません。Python API が起動しているか確認してください。`);
    } finally {
      setLoadingLamp(null);
    }
  }

  async function callPythonApi(event: SubmitEvent<HTMLFormElement>) {
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

      <section className="lamp-panel" aria-label="ランプの点灯チェック">
        <button type="button" onClick={() => checkLamp(1)} disabled={loadingLamp !== null}>
          {loadingLamp === 1 ? "確認中…" : "ランプ1を確認"}
        </button>
        <p role="status">
          <span
            className={`lamp ${isLampOn ? "lamp-on" : "lamp-off"}`}
            aria-hidden="true"
          />
          ランプ1：{isLampOn ? "点灯しています" : "消灯しています"}
        </p>
        <button type="button" onClick={() => checkLamp(2)} disabled={loadingLamp !== null}>
          {loadingLamp === 2 ? "確認中…" : "ランプ2を確認"}
        </button>
        <p role="status">
          <span
            className={`lamp ${isLamp2On ? "lamp-on" : "lamp-off"}`}
            aria-hidden="true"
          />
          ランプ2：{isLamp2On ? "点灯しています" : "消灯しています"}
        </p>
        {lampError && <p className="error" role="alert">{lampError}</p>}
      </section>

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
