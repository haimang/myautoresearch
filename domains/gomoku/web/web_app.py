"""Browser-based Gomoku UI backed by the existing Python game engine."""

from __future__ import annotations

import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── path setup for decoupled project structure ──
_THIS_DIR = Path(__file__).resolve().parent          # domains/gomoku/web/
_DOMAIN_DIR = _THIS_DIR.parent                       # domains/gomoku/
_PROJECT_ROOT = _DOMAIN_DIR.parent.parent             # mag-gomoku/
if str(_DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_DOMAIN_DIR))
if str(_PROJECT_ROOT / "framework") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "framework"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from game import BLACK, WHITE, Board
from play_service import create_player, get_frontend_opponents


ROOT_DIR = _PROJECT_ROOT
WEB_DIR = _THIS_DIR


class CreateSessionRequest(BaseModel):
    opponentType: str = Field(..., pattern="^(nn|minimax)$")
    opponentId: str
    humanColor: str = Field(..., pattern="^(black|white)$")


class MoveRequest(BaseModel):
    row: int
    col: int


@dataclass
class GameSession:
    session_id: str
    opponent: dict
    human_color: int
    black_fn: Optional[callable]
    white_fn: Optional[callable]
    board: Board


app = FastAPI(title="MAG Gomoku Web")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

_sessions: dict[str, GameSession] = {}
_session_lock = threading.Lock()


def _player_name(player: int) -> str:
    return "black" if player == BLACK else "white"


def _winner_name(winner: int) -> Optional[str]:
    if winner == BLACK:
        return "black"
    if winner == WHITE:
        return "white"
    if winner == -1:
        return "draw"
    return None


def _serialize_session(session: GameSession) -> dict:
    board = session.board
    human_can_move = (
        board.winner == 0 and board.current_player == session.human_color
    )
    moves = [
        {
            "index": index + 1,
            "row": row,
            "col": col,
            "player": _player_name(player),
        }
        for index, (row, col, player) in enumerate(board.history)
    ]
    last_move = None
    if board.last_move and board.history:
        last_row, last_col = board.last_move
        last_move = {
            "row": last_row,
            "col": last_col,
            "player": _player_name(board.history[-1][2]),
        }

    return {
        "sessionId": session.session_id,
        "board": board.grid.astype(int).tolist(),
        "currentPlayer": _player_name(board.current_player),
        "humanColor": _player_name(session.human_color),
        "status": "finished" if board.winner != 0 else "ongoing",
        "winner": _winner_name(board.winner),
        "moveCount": board.move_count,
        "lastMove": last_move,
        "moves": moves,
        "opponent": session.opponent,
        "canHumanMove": human_can_move,
        "aiThinking": False,
    }


def _apply_ai_turns(session: GameSession) -> None:
    while session.board.winner == 0 and session.board.current_player != session.human_color:
        player_fn = session.black_fn if session.board.current_player == BLACK else session.white_fn
        if player_fn is None:
            return
        row, col = player_fn(session.board)
        if not session.board.place(row, col):
            raise RuntimeError(f"AI attempted illegal move at ({row}, {col})")


def _new_session(req: CreateSessionRequest) -> GameSession:
    opponent_fn, opponent_meta = create_player(
        req.opponentType,
        req.opponentId,
    )
    human_color = BLACK if req.humanColor == "black" else WHITE
    black_fn = None if human_color == BLACK else opponent_fn
    white_fn = None if human_color == WHITE else opponent_fn
    session = GameSession(
        session_id=str(uuid.uuid4()),
        opponent=opponent_meta,
        human_color=human_color,
        black_fn=black_fn,
        white_fn=white_fn,
        board=Board(),
    )
    _apply_ai_turns(session)
    return session


@app.get("/")
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found")
    return FileResponse(index_path)


@app.get("/api/opponents")
def list_opponents():
    return get_frontend_opponents()


@app.post("/api/session")
def create_session(req: CreateSessionRequest):
    try:
        session = _new_session(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _session_lock:
        _sessions[session.session_id] = session
    return _serialize_session(session)


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_session(session)


@app.post("/api/session/{session_id}/move")
def play_move(session_id: str, req: MoveRequest):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    board = session.board
    if board.winner != 0:
        raise HTTPException(status_code=400, detail="Game already finished")
    if board.current_player != session.human_color:
        raise HTTPException(status_code=400, detail="It is not the human player's turn")
    if not board.place(req.row, req.col):
        raise HTTPException(status_code=400, detail="Illegal move")
    try:
        _apply_ai_turns(session)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _serialize_session(session)


@app.post("/api/session/{session_id}/reset")
def reset_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    req = CreateSessionRequest(
        opponentType=session.opponent["type"],
        opponentId=session.opponent["id"],
        humanColor=_player_name(session.human_color),
    )
    try:
        new_session = _new_session(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    new_session.session_id = session_id
    with _session_lock:
        _sessions[session_id] = new_session
    return _serialize_session(new_session)


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    with _session_lock:
        existed = _sessions.pop(session_id, None)
    if not existed:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


def main() -> None:
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()