const boardCanvas = document.getElementById("board-canvas");
const boardContext = boardCanvas.getContext("2d");

const opponentSelect = document.getElementById("opponent-select");
const colorSelect = document.getElementById("color-select");
const startButton = document.getElementById("start-button");
const resetButton = document.getElementById("reset-button");
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

async function fetchOpponents() {
  const response = await fetch("/api/opponents");
  if (!response.ok) {
    throw new Error("Failed to load opponents");
  }
  opponents = await response.json();
  opponentSelect.innerHTML = "";
  opponents.forEach((item) => {
    const option = document.createElement("option");
    option.value = `${item.type}:${item.id}`;
    option.textContent = formatOpponent(item);
    opponentSelect.appendChild(option);
  });
  // Prefer registered NN opponents (S0, S1, ...), then L4, then first available
  const preferred = opponents.find((item) => item.id === "S0" && item.type === "nn")
    || opponents.find((item) => item.type === "nn")
    || opponents[0];
  if (preferred) {
    opponentSelect.value = `${preferred.type}:${preferred.id}`;
  }
}

async function createSession() {
  const [opponentType, opponentId] = opponentSelect.value.split(":");
  const mctsSelect = document.getElementById("mcts-select");
  const mctsSims = parseInt(mctsSelect ? mctsSelect.value : "0", 10);
  setStatus("Creating session...");
  const response = await fetch("/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      opponentType,
      opponentId,
      humanColor: colorSelect.value,
      mctsSims,
    }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to create session");
  }
  currentSession = payload;
  renderSession();
  setStatus("Session ready.");
}

async function resetSession() {
  if (!currentSession) {
    await createSession();
    return;
  }
  setStatus("Resetting game...");
  const response = await fetch(`/api/session/${currentSession.sessionId}/reset`, {
    method: "POST",
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Failed to reset session");
  }
  currentSession = payload;
  renderSession();
  setStatus("Board reset.");
}

async function playMove(row, col) {
  if (!currentSession || !currentSession.canHumanMove) {
    return;
  }
  setStatus(`Playing at (${row + 1}, ${col + 1})...`);
  const response = await fetch(`/api/session/${currentSession.sessionId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "Move rejected");
  }
  currentSession = payload;
  renderSession();
  setStatus(currentSession.status === "finished" ? "Game finished." : "Your turn.");
}

function drawBoardGrid(size, padding, cell) {
  boardContext.clearRect(0, 0, size, size);

  const wood = boardContext.createLinearGradient(0, 0, size, size);
  wood.addColorStop(0, "#d1a15d");
  wood.addColorStop(1, "#b57a3c");
  boardContext.fillStyle = wood;
  boardContext.fillRect(0, 0, size, size);

  boardContext.strokeStyle = "#4b3628";
  boardContext.lineWidth = 1.3;
  for (let index = 0; index < BOARD_SIZE; index += 1) {
    const pos = padding + index * cell;
    boardContext.beginPath();
    boardContext.moveTo(padding, pos);
    boardContext.lineTo(size - padding, pos);
    boardContext.stroke();
    boardContext.beginPath();
    boardContext.moveTo(pos, padding);
    boardContext.lineTo(pos, size - padding);
    boardContext.stroke();
  }

  const stars = [
    [3, 3], [3, 11], [7, 7], [11, 3], [11, 11],
  ];
  boardContext.fillStyle = "#4b3628";
  stars.forEach(([row, col]) => {
    const x = padding + col * cell;
    const y = padding + row * cell;
    boardContext.beginPath();
    boardContext.arc(x, y, 4, 0, Math.PI * 2);
    boardContext.fill();
  });
}

function drawStone(x, y, radius, color) {
  const gradient = boardContext.createRadialGradient(
    x - radius * 0.35,
    y - radius * 0.35,
    radius * 0.2,
    x,
    y,
    radius
  );
  if (color === BLACK) {
    gradient.addColorStop(0, "#6a625c");
    gradient.addColorStop(1, "#171411");
  } else {
    gradient.addColorStop(0, "#ffffff");
    gradient.addColorStop(1, "#dbd7d0");
  }
  boardContext.fillStyle = gradient;
  boardContext.beginPath();
  boardContext.arc(x, y, radius, 0, Math.PI * 2);
  boardContext.fill();
  boardContext.strokeStyle = color === BLACK ? "#090807" : "#a59d95";
  boardContext.lineWidth = 1;
  boardContext.stroke();
}

function renderBoard() {
  const size = boardCanvas.width;
  const padding = 48;
  const cell = (size - padding * 2) / (BOARD_SIZE - 1);
  drawBoardGrid(size, padding, cell);

  if (!currentSession) {
    return;
  }

  const board = currentSession.board;
  const radius = Math.max(10, cell * 0.42);
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const value = board[row][col];
      if (value === EMPTY) {
        continue;
      }
      const x = padding + col * cell;
      const y = padding + row * cell;
      drawStone(x, y, radius, value);
    }
  }

  if (currentSession.lastMove) {
    const x = padding + currentSession.lastMove.col * cell;
    const y = padding + currentSession.lastMove.row * cell;
    boardContext.fillStyle = "#c93f28";
    boardContext.beginPath();
    boardContext.arc(x, y, 5, 0, Math.PI * 2);
    boardContext.fill();
  }
}

function renderMoveList() {
  moveList.innerHTML = "";
  if (!currentSession) {
    return;
  }
  currentSession.moves.forEach((move) => {
    const item = document.createElement("li");
    item.textContent = `${move.index}. ${move.player} → (${move.row + 1}, ${move.col + 1})`;
    moveList.appendChild(item);
  });
}

function renderSession() {
  renderBoard();
  if (!currentSession) {
    matchTitle.textContent = "No Game";
    opponentName.textContent = "—";
    humanColor.textContent = "—";
    moveCount.textContent = "0";
    resultText.textContent = "—";
    turnPill.textContent = "Idle";
    lastMoveText.textContent = "Last: —";
    moveList.innerHTML = "";
    return;
  }

  matchTitle.textContent = `Human vs ${currentSession.opponent.label}`;
  opponentName.textContent = currentSession.opponent.label;
  humanColor.textContent = currentSession.humanColor;
  moveCount.textContent = String(currentSession.moveCount);
  resultText.textContent = currentSession.winner || (currentSession.status === "ongoing" ? "ongoing" : "—");
  turnPill.textContent = currentSession.status === "finished"
    ? `Winner: ${currentSession.winner}`
    : `Turn: ${currentSession.currentPlayer}`;

  if (currentSession.lastMove) {
    const { row, col, player } = currentSession.lastMove;
    lastMoveText.textContent = `Last: ${player} (${row + 1}, ${col + 1})`;
  } else {
    lastMoveText.textContent = "Last: —";
  }

  renderMoveList();
}

function canvasPositionToMove(event) {
  const rect = boardCanvas.getBoundingClientRect();
  const scaleX = boardCanvas.width / rect.width;
  const scaleY = boardCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  const padding = 48;
  const cell = (boardCanvas.width - padding * 2) / (BOARD_SIZE - 1);
  const col = Math.round((x - padding) / cell);
  const row = Math.round((y - padding) / cell);
  if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) {
    return null;
  }
  return { row, col };
}

boardCanvas.addEventListener("click", async (event) => {
  try {
    const move = canvasPositionToMove(event);
    if (!move) {
      return;
    }
    await playMove(move.row, move.col);
  } catch (error) {
    setStatus(error.message);
  }
});

startButton.addEventListener("click", async () => {
  try {
    await createSession();
  } catch (error) {
    setStatus(error.message);
  }
});

resetButton.addEventListener("click", async () => {
  try {
    await resetSession();
  } catch (error) {
    setStatus(error.message);
  }
});

window.addEventListener("resize", renderBoard);

async function bootstrap() {
  try {
    await fetchOpponents();
    await createSession();
  } catch (error) {
    setStatus(error.message);
    renderSession();
  }
}

renderSession();
bootstrap();