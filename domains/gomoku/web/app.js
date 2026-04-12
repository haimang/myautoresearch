const boardCanvas = document.getElementById("board-canvas");
const boardContext = boardCanvas.getContext("2d");

const modeSelect = document.getElementById("mode-select");
const humanControls = document.getElementById("human-controls");
const autoControls = document.getElementById("auto-controls");
const opponentSelect = document.getElementById("opponent-select");
const colorSelect = document.getElementById("color-select");
const mctsSelect = document.getElementById("mcts-select");
const blackSelect = document.getElementById("black-select");
const blackMctsSelect = document.getElementById("black-mcts");
const whiteSelect = document.getElementById("white-select");
const whiteMctsSelect = document.getElementById("white-mcts");
const startButton = document.getElementById("start-button");
const resetButton = document.getElementById("reset-button");
const stepButton = document.getElementById("step-button");
const autoplayButton = document.getElementById("autoplay-button");
const statusText = document.getElementById("status-text");
const matchTitle = document.getElementById("match-title");
const turnPill = document.getElementById("turn-pill");
const opponentName = document.getElementById("opponent-name");
const humanColor = document.getElementById("human-color");
const moveCount = document.getElementById("move-count");
const resultText = document.getElementById("result-text");
const moveList = document.getElementById("move-list");
const lastMoveText = document.getElementById("last-move-text");

const BOARD_SIZE = 15;
const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;

let opponents = [];
let currentSession = null;
let autoPlayTimer = null;

function setStatus(message) {
  statusText.textContent = message;
}

function formatOpponent(item) {
  if (item.type === "minimax") {
    return `${item.label} · built-in`;
  }
  const parts = [item.label];
  if (item.num_res_blocks && item.num_filters) {
    parts.push(`${item.num_res_blocks}x${item.num_filters}`);
  }
  return parts.join(" · ");
}

function populateSelect(selectEl, items) {
  selectEl.innerHTML = "";
  items.forEach((item) => {
    const option = document.createElement("option");
    option.value = `${item.type}:${item.id}`;
    option.textContent = formatOpponent(item);
    selectEl.appendChild(option);
  });
}

async function fetchOpponents() {
  const response = await fetch("/api/opponents");
  if (!response.ok) throw new Error("Failed to load opponents");
  opponents = await response.json();
  populateSelect(opponentSelect, opponents);
  populateSelect(blackSelect, opponents);
  populateSelect(whiteSelect, opponents);

  // Default selections
  const preferNN = opponents.find((i) => i.id === "S0" && i.type === "nn")
    || opponents.find((i) => i.type === "nn")
    || opponents[0];
  if (preferNN) {
    opponentSelect.value = `${preferNN.type}:${preferNN.id}`;
    blackSelect.value = `${preferNN.type}:${preferNN.id}`;
  }
  const minimax = opponents.find((i) => i.type === "minimax" && i.level === 1)
    || opponents.find((i) => i.type === "minimax");
  if (minimax) {
    whiteSelect.value = `${minimax.type}:${minimax.id}`;
  }
}

// ── Mode switching ──

modeSelect.addEventListener("change", () => {
  const isAuto = modeSelect.value === "auto";
  humanControls.style.display = isAuto ? "none" : "";
  autoControls.style.display = isAuto ? "" : "none";
  stepButton.style.display = "none";
  autoplayButton.style.display = "none";
  stopAutoPlay();
});

// ── Session creation ──

async function createSession() {
  stopAutoPlay();
  if (modeSelect.value === "auto") {
    return createAutoSession();
  }
  const [opponentType, opponentId] = opponentSelect.value.split(":");
  const mctsSims = parseInt(mctsSelect.value, 10);
  setStatus("Creating session...");
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ opponentType, opponentId, humanColor: colorSelect.value, mctsSims }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Failed to create session");
  currentSession = payload;
  stepButton.style.display = "none";
  autoplayButton.style.display = "none";
  renderSession();
  setStatus("Your turn.");
}

async function createAutoSession() {
  const [blackType, blackId] = blackSelect.value.split(":");
  const [whiteType, whiteId] = whiteSelect.value.split(":");
  const blackMcts = parseInt(blackMctsSelect.value, 10);
  const whiteMcts = parseInt(whiteMctsSelect.value, 10);
  setStatus("Creating AI vs AI session...");
  const response = await fetch("/api/session/auto", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blackType, blackId, blackMcts, whiteType, whiteId, whiteMcts }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Failed to create auto session");
  currentSession = payload;
  stepButton.style.display = "";
  autoplayButton.style.display = "";
  autoplayButton.textContent = "Auto Play ▶";
  renderSession();
  setStatus("AI vs AI ready. Click Step or Auto Play.");
}

// ── Auto-play controls ──

async function stepOnce() {
  if (!currentSession || !currentSession.isAuto) return;
  if (currentSession.status === "finished") {
    setStatus("Game finished.");
    stopAutoPlay();
    return;
  }
  setStatus("AI thinking...");
  const response = await fetch(`/api/session/${currentSession.sessionId}/step`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Step failed");
  currentSession = payload;
  renderSession();
  if (currentSession.status === "finished") {
    setStatus(`Game finished. Winner: ${currentSession.winner}`);
    stopAutoPlay();
  } else {
    setStatus(`Move ${currentSession.moveCount}. ${currentSession.currentPlayer}'s turn.`);
  }
}

function startAutoPlay() {
  if (autoPlayTimer) return;
  autoplayButton.textContent = "Pause ⏸";
  autoPlayTimer = setInterval(async () => {
    try {
      await stepOnce();
    } catch (e) {
      setStatus(e.message);
      stopAutoPlay();
    }
  }, 200);
}

function stopAutoPlay() {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer);
    autoPlayTimer = null;
  }
  autoplayButton.textContent = "Auto Play ▶";
}

function toggleAutoPlay() {
  if (autoPlayTimer) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
}

// ── Human move ──

async function resetSession() {
  stopAutoPlay();
  if (!currentSession) {
    await createSession();
    return;
  }
  setStatus("Resetting game...");
  const response = await fetch(`/api/session/${currentSession.sessionId}/reset`, { method: "POST" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Failed to reset");
  currentSession = payload;
  renderSession();
  setStatus("Board reset.");
  if (currentSession.isAuto) {
    autoplayButton.textContent = "Auto Play ▶";
  }
}

async function playMove(row, col) {
  if (!currentSession || !currentSession.canHumanMove) return;
  setStatus(`Playing at (${row + 1}, ${col + 1})...`);
  const response = await fetch(`/api/session/${currentSession.sessionId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Move rejected");
  currentSession = payload;
  renderSession();
  setStatus(currentSession.status === "finished" ? "Game finished." : "Your turn.");
}

// ── Board rendering ──

function drawBoardGrid(size, padding, cell) {
  boardContext.clearRect(0, 0, size, size);
  const wood = boardContext.createLinearGradient(0, 0, size, size);
  wood.addColorStop(0, "#d1a15d");
  wood.addColorStop(1, "#b57a3c");
  boardContext.fillStyle = wood;
  boardContext.fillRect(0, 0, size, size);
  boardContext.strokeStyle = "#4b3628";
  boardContext.lineWidth = 1.3;
  for (let i = 0; i < BOARD_SIZE; i++) {
    const pos = padding + i * cell;
    boardContext.beginPath(); boardContext.moveTo(padding, pos); boardContext.lineTo(size - padding, pos); boardContext.stroke();
    boardContext.beginPath(); boardContext.moveTo(pos, padding); boardContext.lineTo(pos, size - padding); boardContext.stroke();
  }
  const stars = [[3,3],[3,11],[7,7],[11,3],[11,11]];
  boardContext.fillStyle = "#4b3628";
  stars.forEach(([r,c]) => { const x=padding+c*cell, y=padding+r*cell; boardContext.beginPath(); boardContext.arc(x,y,4,0,Math.PI*2); boardContext.fill(); });
}

function drawStone(x, y, radius, color) {
  const g = boardContext.createRadialGradient(x-radius*0.35, y-radius*0.35, radius*0.2, x, y, radius);
  if (color === BLACK) { g.addColorStop(0,"#6a625c"); g.addColorStop(1,"#171411"); }
  else { g.addColorStop(0,"#ffffff"); g.addColorStop(1,"#dbd7d0"); }
  boardContext.fillStyle = g;
  boardContext.beginPath(); boardContext.arc(x,y,radius,0,Math.PI*2); boardContext.fill();
  boardContext.strokeStyle = color===BLACK?"#090807":"#a59d95"; boardContext.lineWidth=1; boardContext.stroke();
}

function renderBoard() {
  const size = boardCanvas.width;
  const padding = 48;
  const cell = (size - padding * 2) / (BOARD_SIZE - 1);
  drawBoardGrid(size, padding, cell);
  if (!currentSession) return;
  const board = currentSession.board;
  const radius = Math.max(10, cell * 0.42);
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== EMPTY) drawStone(padding+c*cell, padding+r*cell, radius, board[r][c]);
    }
  }
  if (currentSession.lastMove) {
    const x = padding + currentSession.lastMove.col * cell;
    const y = padding + currentSession.lastMove.row * cell;
    boardContext.fillStyle = "#c93f28";
    boardContext.beginPath(); boardContext.arc(x,y,5,0,Math.PI*2); boardContext.fill();
  }
}

function renderMoveList() {
  moveList.innerHTML = "";
  if (!currentSession) return;
  currentSession.moves.forEach((m) => {
    const li = document.createElement("li");
    li.textContent = `${m.index}. ${m.player} → (${m.row+1}, ${m.col+1})`;
    moveList.appendChild(li);
  });
}

function renderSession() {
  renderBoard();
  if (!currentSession) {
    matchTitle.textContent = "No Game";
    opponentName.textContent = "—"; humanColor.textContent = "—";
    moveCount.textContent = "0"; resultText.textContent = "—";
    turnPill.textContent = "Idle"; lastMoveText.textContent = "Last: —";
    moveList.innerHTML = ""; return;
  }
  if (currentSession.isAuto) {
    const b = currentSession.opponent.black?.label || "?";
    const w = currentSession.opponent.white?.label || "?";
    matchTitle.textContent = `${b} vs ${w}`;
    opponentName.textContent = `${b} / ${w}`;
    humanColor.textContent = "spectator";
  } else {
    matchTitle.textContent = `Human vs ${currentSession.opponent.label}`;
    opponentName.textContent = currentSession.opponent.label;
    humanColor.textContent = currentSession.humanColor;
  }
  moveCount.textContent = String(currentSession.moveCount);
  resultText.textContent = currentSession.winner || (currentSession.status === "ongoing" ? "ongoing" : "—");
  turnPill.textContent = currentSession.status === "finished"
    ? `Winner: ${currentSession.winner}` : `Turn: ${currentSession.currentPlayer}`;
  if (currentSession.lastMove) {
    const {row,col,player} = currentSession.lastMove;
    lastMoveText.textContent = `Last: ${player} (${row+1}, ${col+1})`;
  } else {
    lastMoveText.textContent = "Last: —";
  }
  renderMoveList();
}

function canvasPositionToMove(event) {
  const rect = boardCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * boardCanvas.width / rect.width;
  const y = (event.clientY - rect.top) * boardCanvas.height / rect.height;
  const padding = 48;
  const cell = (boardCanvas.width - padding * 2) / (BOARD_SIZE - 1);
  const col = Math.round((x - padding) / cell);
  const row = Math.round((y - padding) / cell);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return null;
  return { row, col };
}

// ── Event listeners ──

boardCanvas.addEventListener("click", async (e) => {
  try { const m = canvasPositionToMove(e); if (m) await playMove(m.row, m.col); }
  catch (err) { setStatus(err.message); }
});
startButton.addEventListener("click", async () => { try { await createSession(); } catch(e) { setStatus(e.message); } });
resetButton.addEventListener("click", async () => { try { await resetSession(); } catch(e) { setStatus(e.message); } });
stepButton.addEventListener("click", async () => { try { await stepOnce(); } catch(e) { setStatus(e.message); } });
autoplayButton.addEventListener("click", () => toggleAutoPlay());
window.addEventListener("resize", renderBoard);

async function bootstrap() {
  try { await fetchOpponents(); await createSession(); }
  catch (e) { setStatus(e.message); renderSession(); }
}
renderSession();
bootstrap();
