# カタン・ゲームエンジン設計案（UIを除く）

## 1. この設計の目的

この文書は、Next.js の画面や FastAPI の HTTP / WebSocket API から独立した、カタンのゲームルール本体（ドメイン層）の設計案である。

目標は次のとおり。

- 同じゲームを人間・AIプレイヤーのどちらでも進行できる。
- 4台の端末が接続しても、ゲーム状態はPythonサーバーの一つだけを正とする。
- 対局の各行動を後からSQLite / PostgreSQLへ記録・分析できる。
- 盤面の見た目の座標と、ルール判定用の座標を混ぜない。
- 道や交差点を重複なく表現し、合法手を判定しやすくする。

ゲームルールは `backend/app/domain/` に置く。API、データベース、AI、画面はドメイン層の外側に置く。

```text
Next.js / AI / FastAPI API
           ↓ Action
      domain（ルール本体）
           ↓ GameState / GameEvent
   DB保存・WebSocket配信・分析
```

## 2. 最も重要な考え方：盤面を「グラフ」として扱う

カタンの盤面には3種類の場所がある。

| 実物 | プログラム上の名前 | 置かれるもの |
| --- | --- | --- |
| 六角形の土地 | `HexCoord` / `Tile` | 資源タイル、数字、盗賊 |
| 交差点 | `VertexCoord` / `Vertex` | 開拓地、都市 |
| 交差点どうしを結ぶ線 | `EdgeKey` / `Edge` | 道 |

つまり、交差点を**頂点（vertex）**、道を置く線を**辺（edge）**とするグラフである。

```text
Vertex ── Edge ── Vertex
  ▲                    ▲
開拓地・都市          開拓地・都市
          ↑
       道を置く
```

この表現なら、次を単純に判定できる。

- その交差点に建設物があるか
- その道がすでに誰かに使われているか
- 自分の道と接続しているか
- 相手の開拓地・都市で自分の道が分断されていないか
- 開拓地どうしが隣接していないか（距離ルール）

標準の陸地19枚の盤面は、交差点が54個、道を置ける辺が72本になる。この数は盤面生成のテストにも使う。

## 3. 座標の設計

### 3.1 六角形タイル：軸座標 `HexCoord`

六角形には、正方形の `(x, y)` ではなく、六角形専用の**軸座標（axial coordinate）**を使う。

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class HexCoord:
    """六角形タイルの論理座標。画面上のピクセル座標ではない。"""

    q: int
    r: int
```

標準盤面の19タイルは、中心を `(0, 0)` とした半径2の六角形として生成する。

```python
def standard_hex_coords() -> tuple[HexCoord, ...]:
    radius = 2
    return tuple(
        HexCoord(q, r)
        for q in range(-radius, radius + 1)
        for r in range(-radius, radius + 1)
        if max(abs(q), abs(r), abs(-q - r)) <= radius
    )
```

この結果は `3 - 4 - 5 - 4 - 3` の19タイルになる。隣接タイルは、次の6方向を足して求められる。

```python
HEX_DIRECTIONS: tuple[HexCoord, ...] = (
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
)


def add_hex(a: HexCoord, b: HexCoord) -> HexCoord:
    return HexCoord(a.q + b.q, a.r + b.r)
```

### 3.2 交差点：整数の `VertexCoord`

交差点を「タイル番号と角番号」のように表すと、隣り合うタイルが同じ交差点を別々に持ってしまう。そのため、**盤面全体で共有される一意の交差点座標**を使う。

本設計では、平らな上辺を持つ六角形（flat-top hex）を仮定し、整数だけで交差点を表す。

```python
@dataclass(frozen=True, slots=True, order=True)
class VertexCoord:
    """交差点の論理座標。画面のpx座標ではない。"""

    x: int
    y: int
```

`HexCoord(q, r)` の六角形中心を、以下の整数座標に対応させる。

```text
中心 x = 3 * q
中心 y = q + 2 * r
```

その周りの6交差点のオフセットを、時計回りに次のように固定する。

```python
CORNER_OFFSETS: tuple[tuple[int, int], ...] = (
    (2, 0),    # 0: 右
    (1, 1),    # 1: 右下
    (-1, 1),   # 2: 左下
    (-2, 0),   # 3: 左
    (-1, -1),  # 4: 左上
    (1, -1),   # 5: 右上
)


def corners_of(hex_coord: HexCoord) -> tuple[VertexCoord, ...]:
    center_x = 3 * hex_coord.q
    center_y = hex_coord.q + 2 * hex_coord.r
    return tuple(
        VertexCoord(center_x + dx, center_y + dy)
        for dx, dy in CORNER_OFFSETS
    )
```

この方式では、隣接する六角形が共有する角は**必ず同じ `VertexCoord`** になる。したがって辞書のキーとして安全に使える。

```text
タイルAの角 ─┐
             ├─ VertexCoord(1, 1)  ← 同じ交差点は一つだけ
タイルBの角 ─┘
```

ルール判定で小数・三角関数・ピクセル座標を一切使わないことが重要である。画面での描画位置への変換は、Next.js側にだけ実装する。

### 3.3 道：両端の交差点から作る `EdgeKey`

道を置く場所は、二つの交差点を結んだ辺である。辺は向きを持たないので、端点を必ず並べ替えて一意のキーにする。

```python
from typing import TypeAlias

EdgeKey: TypeAlias = tuple[VertexCoord, VertexCoord]


def edge_key(a: VertexCoord, b: VertexCoord) -> EdgeKey:
    """A→B と B→A を同じ道として扱う。"""

    return (a, b) if a < b else (b, a)


def edges_of(hex_coord: HexCoord) -> tuple[EdgeKey, ...]:
    corners = corners_of(hex_coord)
    return tuple(
        edge_key(corners[index], corners[(index + 1) % 6])
        for index in range(6)
    )
```

二つのタイルが同じ境界を共有していても、両方から生成される `EdgeKey` は同一になる。`set` や `dict` に入れれば、重複は自動的に消える。

## 4. 盤面を「固定の形」と「対局中の状態」に分ける

盤面の形は対局中に変化しない。一方、盗賊、建設物、道の所有者は変化する。これらを同じクラスに入れない。

### 4.1 固定情報：`BoardTopology`

`BoardTopology` は一度作ったら変更しない。標準盤面以外（拡張、ランダム配置）にも対応しやすい。

```python
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vertex:
    coord: VertexCoord
    adjacent_hexes: frozenset[HexCoord]  # 1〜3枚の土地
    incident_edges: frozenset[EdgeKey]  # 接続する道の候補


@dataclass(frozen=True, slots=True)
class Edge:
    key: EdgeKey
    adjacent_hexes: frozenset[HexCoord]  # 海岸は1枚、内側は2枚


@dataclass(frozen=True, slots=True)
class BoardTopology:
    hexes: frozenset[HexCoord]
    vertices: Mapping[VertexCoord, Vertex]
    edges: Mapping[EdgeKey, Edge]
    tile_corners: Mapping[HexCoord, tuple[VertexCoord, ...]]
    tile_edges: Mapping[HexCoord, tuple[EdgeKey, ...]]
```

盤面生成時には、全タイルから `corners_of()` と `edges_of()` を取り出し、次を構築する。

1. `vertices[vertex_coord]` に接するタイルと辺を追加する。
2. `edges[edge_key]` に接するタイルを追加する。
3. `tile_corners` と `tile_edges` に、各タイルの周囲情報を順番どおり保存する。

`Vertex.adjacent_hexes` を持たせることで、「開拓地が出目8で受け取る資源」をすぐ計算できる。

### 4.2 変化する情報：`BoardState`

対局中に変わる盤面情報は、固定トポロジーと別にする。

```python
from enum import StrEnum


class BuildingKind(StrEnum):
    SETTLEMENT = "settlement"
    CITY = "city"


@dataclass(frozen=True, slots=True)
class Building:
    owner_id: int
    kind: BuildingKind


@dataclass(slots=True)
class BoardState:
    robber_hex: HexCoord
    buildings: dict[VertexCoord, Building]
    roads: dict[EdgeKey, int]  # EdgeKey → 所有プレイヤーID
```

例：`roads[edge_key(VertexCoord(1, 1), VertexCoord(2, 0))] = 3` は、プレイヤー3がその道を所有する、という意味である。

`buildings` と `roads` にないキーは、まだ誰も建設していない場所である。空のマスを全件持つ必要はない。

## 5. タイル、資源、プレイヤーのデータ

### 5.1 タイルの内容

土地の種類・数字チップは、ゲーム開始時に決める。盗賊の位置は `BoardState` に置く。

```python
class ResourceType(StrEnum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


@dataclass(frozen=True, slots=True)
class Tile:
    resource: ResourceType | None  # 砂漠なら None
    number_token: int | None       # 砂漠なら None


@dataclass(frozen=True, slots=True)
class BoardSetup:
    """対局開始時に固定される土地・数字の割り当て。"""

    tiles: Mapping[HexCoord, Tile]
```

### 5.2 プレイヤー

資源カードはプレイヤーが持つ。勝利点は必要に応じて都度計算できるが、UI向けにキャッシュする場合でも、建設物との整合性を保つ。

```python
from collections import Counter

PlayerId: TypeAlias = int


@dataclass(slots=True)
class PlayerState:
    id: PlayerId
    name: str
    is_ai: bool
    resources: Counter[ResourceType]
    development_cards: list["DevelopmentCard"]
    played_knights: int = 0
    longest_road_length: int = 0
```

建設物と道の所有情報は `BoardState` を唯一の正とする。`PlayerState` に「自分の道一覧」を重複して保存しない。必要なら `board.roads` を検索して求める。

## 6. 対局全体：`GameState`

```python
class GamePhase(StrEnum):
    SETUP_SETTLEMENT = "setup_settlement"
    SETUP_ROAD = "setup_road"
    ROLL_DICE = "roll_dice"
    ACTION = "action"
    DISCARD = "discard"
    MOVE_ROBBER = "move_robber"
    FINISHED = "finished"


@dataclass(slots=True)
class TurnState:
    player_id: PlayerId
    phase: GamePhase
    turn_number: int
    dice_roll: int | None = None


@dataclass(slots=True)
class GameState:
    game_id: str
    version: int
    topology: BoardTopology
    setup: BoardSetup
    board: BoardState
    players: dict[PlayerId, PlayerState]
    turn: TurnState
    bank: Counter[ResourceType]
    winner_id: PlayerId | None = None
```

`version` は複数端末向けに必須である。プレイヤーが操作を送る時、画面が知っている `version` も送る。サーバー側の `version` と異なれば、その操作を拒否して最新状態を取得させる。

```text
スマホA: version 18 の盤面で「道を置く」を送信
サーバー: 合法なら version 19 に更新し、全員へ配信
スマホB: 古い version 18 で操作を送信
サーバー: 拒否。最新 version 19 を取得させる
```

## 7. 操作はデータとして表現する

人間のクリック、AIの判断、APIリクエストを同じ形式にそろえる。

```python
@dataclass(frozen=True, slots=True)
class PlaceSettlement:
    vertex: VertexCoord


@dataclass(frozen=True, slots=True)
class PlaceRoad:
    edge: EdgeKey


@dataclass(frozen=True, slots=True)
class RollDice:
    pass


Action = PlaceSettlement | PlaceRoad | RollDice


@dataclass(frozen=True, slots=True)
class ActionRequest:
    game_id: str
    player_id: PlayerId
    expected_version: int
    action: Action
```

ゲームエンジンの入口は一つにする。

```python
def apply_action(state: GameState, request: ActionRequest) -> "ApplyResult":
    """手番・資源・接続・距離ルールを検証してから状態を更新する。"""
```

FastAPI は `ActionRequest` を受け取ってこの関数を呼ぶだけにする。AIも、合法手の候補から `ActionRequest` を作って同じ関数を呼ぶ。

## 8. 代表的なルール判定と参照する変数

| 判定 | 主に使う情報 |
| --- | --- |
| 交差点が空いているか | `board.buildings[vertex]` がない |
| 開拓地の距離ルール | `topology.vertices[vertex].incident_edges` の両端に建設物がない |
| 道が空いているか | `edge not in board.roads` |
| 道が自分のネットワークへ接続するか | `edge` の両端の `VertexCoord` と自分の道・建設物 |
| 相手の建設物による道の分断 | 端点の `board.buildings` の所有者 |
| 資源配布 | 各 `building` の `vertex.adjacent_hexes` と `setup.tiles` |
| 盗賊による資源停止 | `board.robber_hex` |

## 9. 推奨ディレクトリ構成

```text
catan_app/
└─ backend/
   ├─ app/
   │  ├─ domain/
   │  │  ├─ types.py            # HexCoord, VertexCoord, EdgeKey, PlayerId
   │  │  ├─ board_topology.py   # 19タイル、54交差点、72辺の生成
   │  │  ├─ board_state.py      # robber, buildings, roads
   │  │  ├─ models.py           # GameState, PlayerState, Tile
   │  │  ├─ actions.py          # PlaceRoad等のAction
   │  │  ├─ rules.py            # 合法手判定、apply_action
   │  │  ├─ scoring.py          # 勝利点、最長交易路
   │  │  └─ events.py           # ActionRecord、GameEvent
   │  ├─ ai/                    # AIはdomainのActionを返す
   │  ├─ api/                   # FastAPIのルーティング
   │  └─ db/                    # SQLAlchemy等の永続化
   └─ tests/
      └─ domain/
         ├─ test_board_topology.py
         └─ test_rules.py
```

## 10. 最初に書くテスト

UIより先に、以下のテストを作る。

```python
def test_standard_board_has_19_hexes_54_vertices_and_72_edges():
    topology = create_standard_topology()
    assert len(topology.hexes) == 19
    assert len(topology.vertices) == 54
    assert len(topology.edges) == 72


def test_shared_tile_corner_is_one_vertex():
    # 隣接タイルの共通角が、同一のVertexCoordとして扱われること
    ...


def test_cannot_build_settlements_on_adjacent_vertices():
    ...


def test_cannot_build_a_road_already_owned_by_another_player():
    ...
```

特に `19 / 54 / 72` のテストが通れば、六角形・交差点・道のトポロジー生成が大きく崩れていないことを最初に確認できる。

## 11. 実装の順序

1. `HexCoord`、`VertexCoord`、`EdgeKey` と `BoardTopology` だけを実装する。
2. 19 / 54 / 72 のテストを通す。
3. `BoardState.buildings` と `BoardState.roads` を追加する。
4. 初期配置の「開拓地を置く」「隣に置けない」「対応する道を置く」を実装する。
5. 通常ターン、資源、ダイス、盗賊、交易、発展カードの順に増やす。
6. ドメインのテストが揃ってから FastAPI・Next.js・AIを接続する。

この順なら、画面の見た目に左右されず、カタンのルールそのものを小さなテストで理解・実装できる。
