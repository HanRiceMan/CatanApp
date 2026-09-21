from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class HelloResponse(BaseModel):
    """フロントエンドへ返す JSON の形を定義する。"""

    message: str
    received_name: str


app = FastAPI(title="Next.js + Python Sample API")

# 開発中は Next.js (http://localhost:3000) からの呼び出しを許可する。
# 別ポートは別の「オリジン」扱いになるため、この設定がないとブラウザが通信を止める。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/hello", response_model=HelloResponse)
def hello(name: str = Query(default="ゲスト", min_length=1)) -> HelloResponse:
    """クエリ文字列の name を受け取り、JSON であいさつを返す。"""

    return HelloResponse(
        message=f"こんにちは、{name}さん。Python (FastAPI) から返答しました。",
        received_name=name,
    )


@app.get("/api/lamp")
def get_lamp() -> dict[str, bool]:
    """画面のランプの点灯状態を返す。"""
    # False に変更すると、次の点灯チェックで消灯する。
    return {"is_on": True}
