# Next.js + Python 接続サンプル

このフォルダは、本番の `catan_app` とは別に、技術を小さく試す場所です。
`next-python-hello` は、Next.js の画面が FastAPI の API を呼び、Python が返した JSON を表示する最小サンプルです。

## 完成時の通信

```text
ブラウザ
  ↓ http://localhost:3000
Next.js (frontend)
  ↓ fetch("http://localhost:8000/api/hello?name=...")
FastAPI (backend)
  ↓ JSON を返す
Next.js が結果を画面に表示
```

フロントエンドとバックエンドは別々のプログラムです。開発中はターミナルを 2 つ開き、それぞれ起動します。

## 事前準備（Windows）

- Node.js 20.9 以上をインストールする（npm も一緒に入ります）
- Python 3.11 以上をインストールする。インストール時は「Add Python to PATH」を有効にする

インストールできたら、PowerShell を開き直し、次で確認します。

```powershell
node --version
npm --version
py --version
```

## 初回セットアップ

### 1. Python 側

```powershell
cd C:\Users\okome\Documents\CATAN\test\next-python-hello\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

PowerShell の実行ポリシーで有効化できない場合だけ、同じターミナルで次を一度実行してから、もう一度有効化してください。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Next.js 側

別の PowerShell を開き、次を実行します。

```powershell
cd C:\Users\okome\Documents\CATAN\test\next-python-hello\frontend
npm install
```

## 起動

### ターミナル A: Python API

仮想環境を有効化した状態で実行します。

```powershell
cd C:\Users\okome\Documents\CATAN\test\next-python-hello\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

`http://localhost:8000/docs` を開くと、FastAPI が自動生成した API 仕様と試用画面を見られます。

### ターミナル B: Next.js

```powershell
cd C:\Users\okome\Documents\CATAN\test\next-python-hello\frontend
npm run dev
```

`http://localhost:3000` を開き、名前を入力して「Python API を呼ぶ」を押してください。

## 試してみる変更

1. `backend/app/main.py` の `message` を変更する
2. API を再起動せず、ブラウザでボタンを押す
3. `frontend/src/app/page.tsx` の見出しやボタン名を変更する
4. ブラウザが自動更新されることを確認する

次は、`/api/roll-dice` のような API を追加し、Python 側でサイコロを振った結果を画面に表示する練習がおすすめです。
