# Tailscaleでスマホから遊ぶ手順

## 目的と公開範囲

この手順は、ゲームPCと招待したスマホだけをTailscaleの閉域ネットワークで接続する。
一般公開はしない。

```text
参加スマホ ── Tailscale（暗号化）── ゲームPC
                                      ├─ Tailscale Serve :80（HTTP）
                                      ├─ Next.js        127.0.0.1:3002
                                      └─ FastAPI        127.0.0.1:8003
```

- `tailscale serve` はTailscale内だけでサイトを中継する。
- `tailscale funnel` はインターネット全体へ公開する機能なので、**使わない**。
- Next.jsとFastAPIはPCのループバックアドレスにだけ待機するため、通常の自宅LANから3002番・8003番へは接続できない。
- ゲームPCがスリープ・再起動・終了すると対局も停止する。対局データは現時点ではPythonのメモリにだけ保存される。

## 初回：ゲームPCをTailscaleへ参加させる

1. PCへ [Tailscale for Windows](https://tailscale.com/download/windows) をインストールする。
2. タスクトレイのTailscaleアイコンからログインし、自分のTailscaleネットワークを作る。
3. Tailscale管理画面のDNS設定で **MagicDNS** を有効にする。
4. PCの名前は `catan-table` のような一般的な名前にする。

このアプリでは、証明書透明性の公開台帳に名前を残さないため、HTTPS Certificatesは有効にしない。
スマホのブラウザにはHTTP接続の警告が表示されることがあるが、ゲーム通信そのものはTailscaleの暗号化された閉域ネットワークを通る。

## 参加者を限定する

参加者自身のスマホにTailscaleをインストールし、各自のアカウントでログインしてもらう。
ゲームPCの管理画面から、**ゲームPC一台だけ**を参加者のアカウントへ共有する。

- 参加者を自分のtailnetへ「招待」しない。
- 自宅のほかのPC、NAS、プリンターは共有しない。
- 遊び終わった参加者は、Tailscale管理画面で共有を取り消す。

PC一台の共有は、共有先のユーザーにそのPCだけを見せる。共有先を変更しても、あなたのtailnetのユーザー数は増えない。

## 対局を開始する

PowerShellを3つ開く。以後、ゲームPCの3つのプロセスを起動したままにする。

### ターミナル A：FastAPI

```powershell
cd C:\Users\okome\Documents\CATAN\catan_app\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8003
```

### ターミナル B：Next.js

```powershell
cd C:\Users\okome\Documents\CATAN\catan_app\frontend
npm.cmd run build
npm.cmd run start:tailscale
```

`start:tailscale` はNext.jsを `127.0.0.1:3002` だけで待機させる。先に `build` が必要である。

### ターミナル C：Tailscale Serve（閉域HTTP中継）

```powershell
tailscale serve --bg --http=80 http://127.0.0.1:3002
tailscale serve status
```

`status` の出力にある `http://<PC名>.<tailnet名>.ts.net` がゲームのURLである。
HTTPS証明書は発行しないため、この名前が証明書の公開台帳へ記録されることはない。
`--bg` を付けているため、Serveはバックグラウンドで動作する。

このURLはTailscaleでゲームPCを共有された端末からだけ開ける。参加者にはTailscaleを接続したうえで、このURLをブラウザで開いてもらう。

## 終了と共有取り消し

ゲームを終えるときは、Next.jsとFastAPIのターミナルで `Ctrl + C` を押す。
Tailscale Serveの中継も止める場合は、次を実行する。

```powershell
tailscale serve --http=80 off
```

このPCでほかのTailscale Serve設定を使っていない場合は、次でも全Serve設定を消せる。

```powershell
tailscale serve reset
```

参加者のアクセスは、Tailscale管理画面のゲームPCの共有設定から個別に取り消す。

## 接続できない場合

1. PCとスマホの両方でTailscaleが「接続済み」か確認する。
2. `tailscale serve status` にHTTPのURLと `127.0.0.1:3002` への転送が表示されるか確認する。
3. PCで `http://localhost:3002` が開けるか確認する。
4. ターミナルA・Bが終了していないか確認する。
5. Tailscale管理画面で、対象スマホのアカウントにゲームPCを共有済みか確認する。

Tailscaleが動作していれば通常はルーターのポート開放やWindowsファイアウォールへの手動追加は不要である。
