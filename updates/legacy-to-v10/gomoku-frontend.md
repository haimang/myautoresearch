# Update Plan — Browser Gomoku Frontend

> 2026-04-12 | 浏览器人机对弈界面规划

---

## 1. 目标

为项目增加一个浏览器中的五子棋对弈界面，使用户可以：

1. 选择任意可用对手进行人机对弈
2. 在开局前切换先后手
3. 在浏览器中完成整局对弈，不依赖 pygame 窗口
4. 查看当前局面、落子顺序、胜负结果、当前轮到谁
5. 支持重新开始、切换对手、切换执黑执白

扩展目标：

1. 支持注册的 NN 对手（L0/L1/L2/L3/L4 和未来新对手）
2. 支持 minimax 对手
3. 后续可扩展为 AI vs AI 观战、局后复盘、候选点热力图

---

## 2. 当前基础能力

现有代码已经具备实现浏览器版所需的核心后端能力：

1. [src/game.py](src/game.py) 已有纯 Python 棋盘逻辑 `Board`，可直接复用，不需要重写规则引擎。
2. [src/play.py](src/play.py) 已有 NN 模型加载、对手走子、minimax 对手接入、先后手切换逻辑。
3. [src/tracker.py](src/tracker.py) 已有注册对手列表与 SQLite 索引能力，可用于前端下拉选择 opponent。

缺的不是“AI 对弈能力”，而是“浏览器交互层”。

---

## 3. 结论

可以做，而且推荐做。

推荐路线不是把现有 pygame 界面硬搬进浏览器，而是：

1. 保留现有 Python 棋盘与 AI 逻辑作为后端事实来源
2. 新增一个轻量 Web 服务暴露 HTTP API
3. 新增一个前端页面负责渲染棋盘、处理点击、展示对局状态

这是当前仓库最稳妥、最少重复实现、后续最容易扩展的方案。

---

## 4. 方案选型

### 4.1 推荐方案

后端：FastAPI

前端：原生 HTML + CSS + JavaScript

理由：

1. 当前项目是 Python 为中心，没有前端构建体系。
2. 需求是单页应用级别，不需要一开始引入 React/Vite 这类额外复杂度。
3. FastAPI 非常适合暴露对局状态、合法落子、AI 回合、opponent 列表等 JSON API。
4. 原生前端足以做出完整棋盘、控制面板、状态栏、落子列表。

### 4.2 暂不推荐方案

1. 继续扩展 pygame：无法在浏览器里使用，不满足目标。
2. 直接把 AI 跑到前端：MLX 模型只能在 Python/本机侧跑，不适合浏览器端推理。
3. 一开始就上 React/Vite：可以做，但对当前仓库来说会引入额外工程复杂度。

---

## 5. 产品范围

### 5.1 首版必须支持

1. 打开浏览器页面看到完整 15x15 五子棋盘。
2. 选择对手：
   - 注册的 NN opponent
   - minimax L0-L3
3. 选择执黑或执白。
4. 点击棋盘落子。
5. 玩家落子后，AI 自动应手。
6. 页面显示：
   - 当前对手
   - 当前轮到谁
   - 执子颜色
   - 对局结果
   - 步数
   - 最近一步
7. 支持“新开一局”。
8. 支持“重新匹配对手/颜色后再开局”。

### 5.2 第二阶段建议支持

1. AI 思考中状态
2. 非法点击提示
3. Move list 落子列表
4. 棋盘 hover 高亮
5. 悔棋（仅限人机本地对局）
6. AI vs AI 观战模式
7. 局后导出 GameRecord

### 5.3 第三阶段可选增强

1. 显示 policy top-k 候选点
2. 显示 value 估值
3. 回放任意历史 recorded game
4. 在页面中直接切换 checkpoint / opponent alias
5. 多局统计面板

---

## 6. 技术架构

### 6.1 总体结构

```text
Browser UI
  -> HTTP / JSON
FastAPI server
  -> Game session manager
  -> Existing Board logic
  -> Existing NN/minimax player adapters
SQLite tracker.db
  -> opponent list / checkpoint metadata
```

### 6.2 会话模型

浏览器端不直接保存最终真相状态，后端维护每一局 session：

1. 一个 session 对应一盘棋
2. session 保存：
   - board
   - black/white player type
   - opponent alias / level / checkpoint
   - human color
   - game status
   - move history
3. 浏览器只是渲染后端返回的状态

这样可以避免前后端规则不一致。

---

## 7. 代码组织建议

推荐新增以下文件：

```text
src/
  web_app.py               # FastAPI 应用入口
  web_api.py               # 路由定义（也可合并到 web_app.py）
  play_service.py          # 抽离 play.py 中的对手加载和会话逻辑
  schemas.py               # Pydantic 请求/响应模型

web/
  index.html               # 页面骨架
  app.js                   # 棋盘交互、API 调用、状态刷新
  styles.css               # 界面样式
```

### 7.1 重构原则

不要把业务逻辑复制一份到前端项目里。应优先抽离复用：

1. 从 [src/play.py](src/play.py) 抽出：
   - checkpoint 解析
   - opponent 列表获取
   - NN player 加载
   - minimax player 适配
2. 把它们变成可被 CLI 和 Web 共用的 service 层。
3. `play.py` 继续保留 pygame 模式，不破坏现有入口。

---

## 8. API 设计

### 8.1 `GET /api/opponents`

返回所有可选对手。

返回示例：

```json
[
  {
    "id": "L4",
    "label": "L4",
    "type": "nn",
    "description": "registered NN opponent",
    "num_res_blocks": 10,
    "num_filters": 64
  },
  {
    "id": "minimax-2",
    "label": "Minimax L2",
    "type": "minimax",
    "level": 2
  }
]
```

### 8.2 `POST /api/session`

创建新对局。

请求示例：

```json
{
  "opponentType": "nn",
  "opponentId": "L4",
  "humanColor": "black"
}
```

响应示例：

```json
{
  "sessionId": "uuid",
  "state": { ... }
}
```

### 8.3 `GET /api/session/{sessionId}`

获取当前对局状态。

状态字段建议包括：

```json
{
  "board": [[0,0,1,...], ...],
  "currentPlayer": "white",
  "humanColor": "black",
  "status": "ongoing",
  "winner": null,
  "moveCount": 17,
  "lastMove": {"row": 7, "col": 8, "player": "black"},
  "moves": [...],
  "opponent": {"type": "nn", "id": "L4"},
  "canHumanMove": false,
  "aiThinking": false
}
```

### 8.4 `POST /api/session/{sessionId}/move`

玩家落子。

请求：

```json
{
  "row": 7,
  "col": 7
}
```

服务端行为：

1. 校验当前是否轮到人类
2. 校验落子是否合法
3. 执行人类落子
4. 如果棋局未结束，则立即执行 AI 落子
5. 返回更新后的完整状态

### 8.5 `POST /api/session/{sessionId}/reset`

按当前配置重开一局。

### 8.6 `DELETE /api/session/{sessionId}`

释放 session，避免内存中累积旧对局。

---

## 9. 前端界面设计

### 9.1 页面布局

推荐单页三栏布局：

1. 左侧：控制面板
   - 对手选择
   - 执黑/执白切换
   - 新开一局
   - 重置
2. 中间：棋盘区
   - 15x15 格
   - 坐标标识
   - 最近一步高亮
   - 胜利连线高亮（第二阶段）
3. 右侧：对局信息
   - 当前回合
   - 对局结果
   - move list
   - 对手信息

### 9.2 样式方向

视觉上应保留棋类质感，而不是普通表格：

1. 木纹/暖色背景
2. 棋盘网格与星位
3. 黑白棋子立体感
4. 最近一步有明确标记
5. 移动端至少保证可读，不要求首版完整移动端交互优化

### 9.3 交互细节

1. AI 回合时禁用棋盘点击
2. 非法位置点击不给提交或给轻提示
3. 棋局结束后显示结果 Banner
4. 选择白棋开局时，创建 session 后由后端先走 AI 第一步

---

## 10. 与现有代码的衔接

### 10.1 复用现有棋盘逻辑

直接复用 [src/game.py](src/game.py) 中：

1. `Board`
2. 胜负判断
3. 历史记录结构

### 10.2 复用现有对手逻辑

直接复用或抽离 [src/play.py](src/play.py) 中：

1. `resolve_checkpoint()`
2. `load_nn_player()`
3. minimax 对手接入逻辑
4. `--swap` 的黑白切换语义

### 10.3 复用现有对手注册数据

通过 [src/tracker.py](src/tracker.py) 的 SQLite 能力读取 opponents 表，生成前端选择列表。

---

## 11. 风险与约束

### 11.1 模型推理延迟

NN opponent 每步都要跑一次模型推理。风险：

1. 如果以后接入 MCTS，浏览器响应会明显变慢
2. 如果多个浏览器会话同时存在，推理资源会叠加

首版建议：

1. 先只支持纯 policy 推理
2. 明确单机本地使用场景
3. 不做多用户并发保证

### 11.2 状态一致性

如果前端自己判断合法落子，容易和后端脱节。解决办法：

1. 合法性以后端为准
2. 前端只做基础拦截，最终以服务端返回状态覆盖

### 11.3 代码重复风险

如果把 `play.py` 里的逻辑复制到 web 入口，后续会分叉。解决办法：

1. 优先抽 service 层
2. CLI 与 Web 共用同一套 player factory

---

## 12. 实施里程碑

### Milestone 1: 后端最小闭环

目标：浏览器通过 API 能完整下一盘棋。

任务：

1. 增加 FastAPI 依赖
2. 创建 `web_app.py`
3. 实现 session manager
4. 实现 `GET /api/opponents`
5. 实现 `POST /api/session`
6. 实现 `POST /api/session/{id}/move`

验收：

1. 使用 curl 或浏览器 devtools 可完整驱动一盘人机对局

### Milestone 2: 浏览器棋盘 UI

目标：能在页面上和 AI 下完整一盘。

任务：

1. 绘制棋盘
2. 渲染棋子
3. 实现点击落子
4. 对接 session 创建和 move API
5. 展示当前状态与结果

验收：

1. 浏览器打开页面后，可选择对手并完成一局对弈

### Milestone 3: 完整对弈体验

目标：达到你现在描述的“完整的五子棋界面”。

任务：

1. 先后手切换
2. move list
3. 最近一步高亮
4. 重开一局
5. 更好的错误提示
6. AI thinking 状态

验收：

1. 页面可以稳定作为本地人机对弈入口使用

### Milestone 4: 扩展能力

可选：

1. AI vs AI
2. 回放 recorded games
3. 候选点和价值显示
4. checkpoint 直选

---

## 13. 依赖变更建议

建议新增：

```toml
fastapi
uvicorn
jinja2
```

说明：

1. `fastapi` 用于 API
2. `uvicorn` 用于本地启动服务
3. `jinja2` 可选，用于服务端返回 HTML；若直接提供静态文件，也可以不加

首版完全可以不引入 npm 工具链。

---

## 14. 启动方式建议

目标命令：

```bash
uv run python src/web_app.py
```

或者：

```bash
uv run uvicorn web_app:app --app-dir src --reload
```

打开浏览器：

```text
http://127.0.0.1:8000
```

---

## 15. 最终交付定义

完成后，项目应新增一个浏览器入口，使用户可以：

1. 打开网页
2. 选择 opponent
3. 选择自己执黑或执白
4. 与 L4 或任意已注册 opponent 直接对局
5. 全程在网页中完成整局操作

这会把项目从“训练系统 + pygame 演示器”推进到“可交互的本地 AI 棋类应用”。

---

## 16. 建议执行顺序

建议按照下面顺序落地：

1. 抽离 [src/play.py](src/play.py) 中可复用的 player/service 逻辑
2. 做 FastAPI session API
3. 做最小棋盘页面
4. 接 opponent 选择和先后手切换
5. 再补 move list、重开、状态提示

不要一开始就上复杂前端框架，也不要先做 AI vs AI。先把“人类 vs 任意 opponent”闭环做通。

---

## 17. 本次实施回填

### 17.1 已完成工作

本次已经按最小可用版本直接落地，完成内容如下：

1. 新增共享对弈服务 [src/play_service.py](src/play_service.py)
2. 新增浏览器后端入口 [src/web_app.py](src/web_app.py)
3. 新增浏览器前端页面 [web/index.html](web/index.html)
4. 新增前端交互逻辑 [web/app.js](web/app.js)
5. 新增前端样式 [web/styles.css](web/styles.css)
6. 修改 [src/play.py](src/play.py)，改为复用共享的对弈服务
7. 修改 [pyproject.toml](pyproject.toml)，加入 `fastapi` 与 `uvicorn` 依赖

### 17.2 实际实现的能力

当前浏览器版本已经支持：

1. 打开本地网页进入五子棋界面
2. 选择 minimax 对手
3. 选择已注册的 NN opponent，例如 L1-L4
4. 选择自己执黑或执白
5. 点击棋盘落子
6. 玩家落子后由后端驱动 AI 自动应手
7. 显示当前对手、当前回合、步数、结果、最近一步
8. 显示 move list
9. 重开当前配置的一局新棋

另外，当前浏览器入口对 NN opponent 增加了一个轻量战术护栏：

1. 若 AI 当前有一步必胜，则直接下该步
2. 若对手下一手存在立即取胜点，则优先堵住

这层护栏只用于人机对弈入口，不影响训练和 benchmark。

如果选择白棋，后端会在建局时先让 AI 落第一手，这部分已经实装。

### 17.3 技术实现说明

最终采用的是“Python 后端为真相源，浏览器负责交互”的结构：

1. [src/game.py](src/game.py) 中的 `Board` 继续作为唯一规则引擎
2. [src/play_service.py](src/play_service.py) 负责统一管理：
  - checkpoint / opponent 解析
  - registered opponent 加载
  - minimax / NN player 创建
3. [src/web_app.py](src/web_app.py) 提供：
  - `GET /api/opponents`
  - `POST /api/session`
  - `GET /api/session/{id}`
  - `POST /api/session/{id}/move`
  - `POST /api/session/{id}/reset`
  - `DELETE /api/session/{id}`
4. [web/app.js](web/app.js) 负责棋盘绘制与 API 调用

这样做的好处是：CLI 对弈和浏览器对弈共用一套 AI 走子逻辑，不会分叉成两套规则。

### 17.4 本地测试结果

本次已完成以下实际测试：

1. Python 侧静态检查
  - [src/play.py](src/play.py)：无错误
  - [src/play_service.py](src/play_service.py)：无错误
  - [src/web_app.py](src/web_app.py)：无错误

2. Web 服务启动测试
  - 使用命令：`uv run python src/web_app.py`
  - 结果：服务成功启动在 `http://127.0.0.1:8000`

3. API 端到端测试
  - `GET /api/opponents` 成功返回 9 个可选 opponent
  - `POST /api/session` 成功创建新会话
  - `POST /api/session/{id}/move` 成功处理玩家落子，并触发 AI 自动应手

4. 白棋开局测试
  - 创建 `humanColor = white` 的 session 后
  - 后端已自动让黑方 AI 先走一步
  - 返回结果中 `moveCount = 1`，且 `canHumanMove = true`

5. NN opponent 测试
  - 成功创建 `opponentType = nn`、`opponentId = L4` 的 session
  - 玩家落子后，L4 已成功返回应手
  - 在“一步必须堵住”的局面中，浏览器入口中的 L4 已能落到唯一堵点

6. CLI 兼容性测试
  - `uv run python src/play.py --list-opponents` 正常工作
  - 说明 pygame 入口与 Web 入口可以共存

7. 浏览器打开测试
  - 页面已成功通过本地浏览器打开

### 17.5 当前已知边界

当前版本是最小可用版，以下内容暂未实现：

1. AI thinking 动画
2. 悔棋
3. AI vs AI 浏览器观战
4. 候选点热力图 / value 显示
5. checkpoint 直选

这些不影响你直接与 L4 或其他 opponent 在浏览器中对局。

### 17.6 如何使用前端

首次使用建议先同步依赖：

```bash
uv sync
```

然后启动浏览器前端：

```bash
uv run python src/web_app.py
```

或使用 uvicorn：

```bash
uv run uvicorn web_app:app --app-dir src --reload
```

启动后打开浏览器：

```text
http://127.0.0.1:8000
```

使用步骤：

1. 在左侧 `Opponent` 下拉框中选择对手
2. 在 `Your Color` 中选择黑棋或白棋
3. 点击 `Start New Game`
4. 在棋盘上点击合法位置落子
5. AI 会自动应手
6. 右侧面板可查看步数、结果、最近一步与完整 move list
7. 若想同配置重开，点击 `Reset Current Game`

### 17.7 后续建议

下一步最值得补的是两项：

1. 浏览器版 AI vs AI 观战
2. 对局结束后保存 GameRecord，便于后续复盘与导出