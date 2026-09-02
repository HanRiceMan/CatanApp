# Next.js が Python API を呼ぶ最小サンプル

親フォルダの [README](../README.md) の手順でセットアップ・起動してください。

## 役割

- `frontend/`: ブラウザで動く Next.js。ボタン操作と表示を担当する。
- `backend/`: Python / FastAPI。データやゲームルールの処理を担当する。

`frontend/src/app/page.tsx` の `fetch` が API 呼び出しの中心です。
`backend/app/main.py` の `@app.get("/api/hello")` が、呼び出される Python の関数です。
