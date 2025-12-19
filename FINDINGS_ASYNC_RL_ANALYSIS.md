# Анализ GELab-Zero: Находки для изучения Асинхронного RL с Верификаторами Наград

## Введение

GELab-Zero — production-grade система для GUI-агентов на мобильных устройствах. Хотя это не классический RL с обучением политики, здесь реализованы ключевые паттерны **асинхронного инференса**, **верификации состояний** и **рубрик для принятия решений**, которые напрямую применимы к асинхронному RL.

---

## 🟢 БАЗОВЫЙ УРОВЕНЬ

### Находка 1: Архитектура Agent Loop с отделением инференса от исполнения

**Файл:** `copilot_agent_client/mcp_agent_loop.py:267-382`

```python
for step_idx in range(max_steps):
    # 1. Capture observation
    image_path = capture_screenshot(device_id, "tmp_screenshot")
    image_b64_url = make_b64_url(image_path)

    # 2. Inference
    server_return = agent_server.automate_step(payload)
    action, global_step_idx = server_return['action'], server_return['current_step']

    # 3. Execute action
    act_on_device(action, device_id, device_wm_size)

    # 4. Check terminal conditions
    if action['action_type'].upper() in ['COMPLETE', "ABORT"]:
        break
```

**Почему важно для асинхронного RL:**
- Классическая декомпозиция RL: `Observation → Policy → Action → Environment`
- Инференс модели полностью отделён от исполнения действий
- Позволяет масштабировать: один сервер политики обслуживает N workers

---

### Находка 2: Session-based State Management

**Файл:** `copilot_agent_server/local_server.py:35-69`

```python
def get_session(self, payload: dict) -> str:
    session_id = str(uuid.uuid4())

    logger = LocalServerLogger({
        "log_dir": self.server_config["log_dir"],
        "session_id": session_id
    })

    message_to_log = {
        "log_type": "session_start",
        "task": payload["task"],
        "model_config": payload["model_config"],
    }
    logger.log_str(message_to_log)
    return session_id
```

**Почему важно:**
- Каждый rollout изолирован через UUID-сессию
- История хранится персистентно в JSONL
- Позволяет продолжать прерванные траектории (критично для async RL)
- Базовый паттерн для Experience Replay Buffer

---

### Находка 3: Structured Action Space Definition

**Файл:** `copilot_tools/parser_0920_summary.py:23-52`

```python
task_define_prompt = """你是一个手机 GUI-Agent 操作专家...

# Action Space:
1. CLICK：点击手机屏幕坐标，需包含点击的坐标位置 point。
例如：action:CLICK\tpoint:x,y
2. TYPE：在手机输入框中输入文字...
3. COMPLETE：任务完成后向用户报告结果...
4. WAIT：等待指定时长...
5. AWAKE：唤醒指定应用...
6. INFO：询问用户问题或详细信息...  # ← Критически важный action!
7. ABORT：终止当前任务...
8. SLIDE：在手机屏幕上滑动...
9. LONGPRESS：长按手机屏幕坐标...
"""
```

**Почему важно:**
- **Дискретное пространство действий** с чёткими параметрами
- Действие `INFO` — механизм запроса обратной связи от верификатора
- Действия `COMPLETE/ABORT` — терминальные сигналы (reward signals)
- В RL: определение action space критично для сходимости обучения

---

### Находка 4: Deterministic Action Parsing с Error Recovery

**Файл:** `copilot_tools/parser_0920_summary.py:255-313`

```python
def str2action(self, command_str):
    # Normalize THINK tags: fix typos, case, and spacing
    command_str = (
        command_str
        .replace("<TINK>", "<THINK>").replace("</TINK>", "</THINK>")
        .replace("<think>", "<THINK>").replace("</think>", "</THINK>")
    )
    command_str = re.sub(r"<\s*/?THINK\s*>", lambda m: ..., command_str)

    try:
        cot_part = command_str.split("<THINK>")[1].split("</THINK>")[0].strip()
        kv_part = command_str.split("</THINK>")[1].strip()
    except IndexError:
        print(f"[Parser Warning] Missing <THINK> tags, treating entire response as kv")
        kv_part = command_str
        cot_part = ""
```

**Почему важно:**
- **Robustness к ошибкам генерации** — модели иногда выдают `<TINK>` вместо `<THINK>`
- Graceful degradation при отсутствии ожидаемого формата
- В RL: политика должна уметь обрабатывать невалидные выходы

---

## 🟡 СРЕДНИЙ УРОВЕНЬ

### Находка 5: Concurrent Verification Thread (Асинхронный Верификатор)

**Файл:** `copilot_agent_client/mcp_agent_loop.py:285-296`

```python
if enable_intermediate_image_caption:
    # Запускаем верификацию параллельно с основным инференсом
    caption_result_container = {}
    caption_thread = threading.Thread(
        target=lambda: caption_current_screenshot(
            current_task=task,
            current_image_url=image_b64_url,
            model_config=agent_loop_config['caption_config'],
            result_container=caption_result_container
        )
    )
    caption_thread.start()

# ... основной инференс агента происходит здесь ...

if enable_intermediate_image_caption:
    caption_thread.join()  # Синхронизация
    caption_text = caption_result_container.get('caption', '')
```

**Почему это КЛЮЧЕВАЯ находка для async RL:**

1. **Асинхронная верификация**: Пока агент думает, отдельный процесс анализирует текущее состояние
2. **Non-blocking design**: Верификатор не блокирует основной цикл
3. **Паттерн Actor-Critic**: Caption model — это "critic", оценивающий состояние
4. **Применимость**: В async RL reward verifier может работать параллельно с policy network

---

### Находка 6: Markov State Abstraction через Summary

**Файл:** `copilot_tools/parser_0920_summary.py:315-324`

```python
def env2messages4ask(self, task, environments, actions, ...) -> list:
    # Use the summary of the LAST action as the historical summary
    summary_history = ""
    if len(actions) > 0:
        last_action = self.action2action(actions[-1])
        summary_history = last_action.get('summary', '')  # ← Markov abstraction!

    current_env = environments[-1]
```

**Архитектурное решение:**
```
Полная история: [s0, a0, s1, a1, s2, a2, ...] — O(n) памяти
Markov summary: summary(a_t-1) + s_t — O(1) памяти
```

**Почему важно для асинхронного RL:**
- **Сжатие состояния**: Вместо передачи всей истории, используется summary предыдущего действия
- **Markov assumption**: Текущее решение зависит только от summary + текущего observation
- Критично для масштабирования: асинхронные workers не хранят полную историю
- Аналог: **frame stacking** в Atari или **state summarization** в трансформерах

---

### Находка 7: Multi-Device Parallel Rollouts

**Файл:** `copilot_agent_client/local_server_based_runner.py:15-45, 210-234`

```python
class CopilotClientRolloutRunner:
    def __init__(self, device_task_map: dict, server, rollout_config: dict, ...):
        # Очередь задач для каждого устройства
        self.task_queue = {}
        for device_id in device_task_map:
            self.task_queue[device_id] = Queue()

        self.done_queue = Queue()  # Результаты от всех workers
        self.log_queue = Queue()   # Централизованный логгер

    def run(self):
        workers = []
        # Запуск логгера в отдельном процессе
        logger = Process(target=self.logger_runner)
        logger.start()

        # Worker для каждого устройства
        for device_id in self.device_task_map:
            worker = Process(target=self.work_runner, args=(device_id,))
            workers.append(worker)
            worker.start()

        # Writer собирает результаты
        writer = Process(target=self.writer_runner)
        writer.start()
```

**Архитектура:**
```
                    ┌─────────────┐
                    │  LogQueue   │
                    └──────▲──────┘
                           │
    ┌──────────┬───────────┼───────────┬──────────┐
    │          │           │           │          │
┌───▼───┐  ┌───▼───┐  ┌────▼────┐  ┌───▼───┐  ┌───▼───┐
│Worker1│  │Worker2│  │  Logger │  │Worker3│  │Writer │
│Device1│  │Device2│  │ Process │  │Device3│  │Process│
└───┬───┘  └───┬───┘  └─────────┘  └───┬───┘  └───▲───┘
    │          │                       │          │
    └──────────┴───────────────────────┴──────────┘
                           │
                    ┌──────▼──────┐
                    │  DoneQueue  │
                    └─────────────┘
```

**Почему важно:**
- **Параллельные rollouts** — основа асинхронного RL (A3C, IMPALA, Apex)
- Каждый worker независимо собирает траектории
- Централизованный writer — аналог Replay Buffer
- Deduplication: `key_func(task, model_name)` предотвращает повторную работу

---

### Находка 8: INFO Action как механизм Human-in-the-Loop Verification

**Файл:** `copilot_agent_client/mcp_agent_loop.py:342-368`

```python
if action['action_type'].upper() == "INFO":
    if reply_mode == "auto_reply":
        # Автоматический верификатор (LLM)
        reply_info = auto_reply(image_b64_url, task, action, ...)

    elif reply_mode == "no_reply":
        # Игнорирование (agent may get stuck)
        reply_info = "Please follow the task and continue."

    elif reply_mode == "manual_reply":
        # Human-in-the-loop
        reply_info = input("Your reply:")

    elif reply_mode == "pass_to_client":
        # Делегирование внешнему клиенту
        stop_reason = "INFO_ACTION_NEEDS_REPLY"
        break
```

**Почему это рубрика (rubric):**
- Агент **явно запрашивает верификацию** когда неуверен
- 4 режима обработки — гибкая настройка уровня автономности
- `auto_reply` — **LLM-based reward verifier** оценивает состояние и отвечает
- `pass_to_client` — асинхронное прерывание для внешней верификации

---

### Находка 9: Stop Reason Classification (Reward Signals)

**Файл:** `copilot_agent_client/mcp_agent_loop.py:426-436`

```python
if stop_reason in ['MANUAL_STOP_SCREEN_OFF', 'INFO_ACTION_NEEDS_REPLY', "NOT_STARTED"]:
    pass  # Neutral — не успех и не провал
elif action['action_type'].upper() == 'COMPLETE':
    stop_reason = "TASK_COMPLETED_SUCCESSFULLY"  # Positive reward
elif action['action_type'].upper() == 'ABORT':
    stop_reason = "TASK_ABORTED_BY_AGENT"        # Negative reward
elif step_idx == max_steps - 1:
    stop_reason = "MAX_STEPS_REACHED"            # Timeout penalty
```

**Mapping к RL rewards:**
| Stop Reason | RL Interpretation | Reward Signal |
|-------------|-------------------|---------------|
| TASK_COMPLETED_SUCCESSFULLY | Goal reached | +1.0 |
| TASK_ABORTED_BY_AGENT | Agent determined impossible | -0.5 |
| MAX_STEPS_REACHED | Timeout | -0.2 |
| INFO_ACTION_NEEDS_REPLY | Needs external verification | 0 (pending) |
| MANUAL_STOP | External interrupt | 0 (cancelled) |

---

## 🔴 ПРОДВИНУТЫЙ УРОВЕНЬ

### Находка 10: Chain-of-Thought как Interpretable Policy

**Файл:** `copilot_tools/parser_0920_summary.py:86-97`

```python
status_conversation = [
    {
        "type": "text",
        "text": f'''
在执行操作之前，请务必回顾你的历史操作记录和限定的动作空间，先进行思考和解释然后输出动作空间和对应的参数：
1. 思考（THINK）：在 <THINK> 和 </THINK> 标签之间。
2. 解释（explain）：在动作格式中，使用 explain: 开头，简要说明当前动作的目的和执行方式。
在执行完操作后，请输出执行完当前步骤后的新历史总结。
输出格式示例：
<THINK> 思考的内容 </THINK>
explain:解释的内容\taction:动作空间和对应的参数\tsummary:执行完当前步骤后的新历史总结
'''
    }
]
```

**Структура выхода политики:**
```
<THINK>                           ← Chain-of-Thought (интерпретируемость)
用户要打开微信，当前在主屏幕，
需要找到微信图标并点击
</THINK>
explain:点击微信图标打开应用      ← Explanation (для верификации)
action:CLICK                      ← Discrete action
point:234,567                     ← Action parameters
summary:已打开微信主界面          ← Next state prediction (для Markov)
```

**Почему это продвинутый паттерн:**
1. **CoT улучшает reasoning** — модель "думает вслух"
2. **Explain как auxiliary task** — дополнительный supervision signal
3. **Summary как world model** — предсказание следующего состояния
4. Применимость в RL: Process Reward Models (PRM) могут верифицировать каждый шаг CoT

---

### Находка 11: Auto-Reply Verifier как Learned Reward Model

**Файл:** `copilot_agent_client/mcp_agent_loop.py:30-88`

```python
def auto_reply(current_image_url, task, info_action, model_provider, model_name):
    messages_to_ask = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""# 角色
你将扮演一个正在使用GUI Agent完成任务的用户。

# 任务
阅读下方提供的所有背景信息，针对[Agent的澄清问题]，生成一个提供关键信息的、简短直接的回答。

# 背景信息
- **任务目标:** {task}
- **agent 问的问题:** {json.dumps(info_action, ensure_ascii=False)}

# 输出要求
- 你的回答必须极其简短和明确。
- 你的回答应直接命中问题的核心，解决Agent的疑惑。
"""
                },
                {'type': "image_url", 'image_url': {'url': current_image_url}},
            ]
        }
    ]

    response = ask_llm_anything(model_provider, model_name, messages_to_ask, ...)
    return response
```

**Архитектура верификатора:**
```
┌─────────────────────────────────────────────────────────┐
│                   Auto-Reply Verifier                    │
├─────────────────────────────────────────────────────────┤
│  Input:                                                  │
│    • current_image_url — текущий скриншот               │
│    • task — исходная задача                             │
│    • info_action — вопрос агента                        │
│                                                          │
│  Processing:                                             │
│    • Multimodal LLM анализирует контекст                │
│    • Генерирует ответ с позиции "идеального пользователя"│
│                                                          │
│  Output:                                                 │
│    • Краткий, информативный ответ                       │
└─────────────────────────────────────────────────────────┘
```

**Применимость к RL с верификаторами:**
- Это **Outcome Reward Model (ORM)** — оценивает, что нужно делать дальше
- Можно заменить на обученный reward model вместо LLM
- Паттерн: агент не знает → спрашивает → верификатор отвечает → агент продолжает

---

### Находка 12: Caption Model как State Verifier

**Файл:** `copilot_agent_client/mcp_agent_loop.py:90-132`

```python
def caption_current_screenshot(current_task, current_image_url, model_config, result_container=None):
    messages_to_ask = [
        {
            "role": "user",
            "content": [
                {'type': "image_url", 'image_url': {'url': current_image_url}},
                {
                    "type": "text",
                    "text": f"当前的任务是：{current_task}。\n"
                            f"请根据任务需求，详细描述出当前截图和任务相关的部分。"
                            f"如果有列表，请列出所有选项。"
                },
            ]
        }
    ]

    response = ask_llm_anything(model_provider, model_name, messages_to_ask, args={
        "max_tokens": 256,
        "temperature": 0.5,
    })

    if result_container is not None:
        result_container['caption'] = response
    return response
```

**Это Process Reward Model (PRM):**
- Анализирует **промежуточное состояние**, а не только финальный результат
- Генерирует **текстовое описание** для интерпретируемости
- Может использоваться для:
  - Раннего обнаружения ошибок
  - Подсчёта промежуточных наград
  - Credit assignment в длинных траекториях

---

### Находка 13: Q&A History Tracking для Multi-Turn Verification

**Файл:** `copilot_tools/parser_0920_summary.py:332-351`

```python
def env2messages4ask(self, task, environments, actions, ...):
    historica_qa = []

    for idx in range(1, len(environments)):
        prev_act = actions[idx - 1]
        current_env = environments[idx]

        if prev_act['action'] == "INFO":
            # Сохраняем пару: вопрос агента + ответ верификатора
            q = prev_act['value']
            a = current_env['user_comment'].strip()
            historica_qa.append((q, a))

    if len(historica_qa) > 0:
        qa_prompt = "这是你和用户的对话历史：\n" + \
                    "\n".join([f"你曾经提出的问题：{qa[0]}\n用户对你的指示：{qa[1]}"
                              for qa in historica_qa]) + \
                    "\n\n你需要更加注意用户最后的指示。"
```

**Паттерн Iterative Refinement:**
```
Turn 1: Agent → INFO("Какой товар добавить?") → Verifier → "Первый в списке"
Turn 2: Agent → CLICK(товар) → ...
Turn 3: Agent → INFO("Какой размер?") → Verifier → "XL"
...
```

**Применимость:**
- **Constitutional AI** стиль: агент учится через диалог с верификатором
- История Q&A — это **preference data** для RLHF
- "更加注意用户最后的指示" — инструкция следовать последней коррекции

---

### Находка 14: Asynchronous Session Continuation

**Файл:** `mcp_server/detailed_gelab_mcp_server.py:66-89`

```python
@mcp.tool
def ask_agent(
    device_id: str,
    task: str | None,  # Новая задача ИЛИ None для продолжения
    session_id: str | None = None,  # ID для продолжения сессии
    reply_from_client: str | None = None,  # Ответ на INFO
) -> dict:

    if task is not None:
        assert session_id is None  # Новая задача
        reset_environment = True
    else:
        assert session_id is not None  # Продолжение
        reset_environment = False
```

**Асинхронный паттерн:**
```
[Time T0] Client → ask_agent(task="Купить товар") → session_123
[Time T1] Agent → INFO("Какой размер?") → stop_reason=INFO_ACTION_NEEDS_REPLY
[Time T2] Client получает результат, показывает пользователю
[Time T3] User отвечает "XL"
[Time T4] Client → ask_agent(session_id="session_123", reply="XL") → продолжение
```

**Почему это критично для production async RL:**
- **Non-blocking verification**: Внешний верификатор может думать сколько угодно
- **State persistence**: Траектория не теряется между вызовами
- **Composability**: Можно строить сложные multi-agent системы

---

## ✨ ИНТЕРЕСНЫЕ НАХОДКИ

### Находка 15: Coordinate Normalization (0-1000 space)

**Файл:** `copilot_front_end/pu_frontend_executor.py:182-186`

```python
def _convert_point_to_realworld_point(point, wm_size):
    x, y = point
    real_x = (float(x) / 1000) * wm_size[0]
    real_y = (float(y) / 1000) * wm_size[1]
    return (real_x, real_y)
```

**Инженерное решение:**
- Координаты в диапазоне [0, 1000] × [0, 1000]
- Независимость от разрешения экрана
- Для RL: нормализованное action space улучшает transfer learning между устройствами

---

### Находка 16: Thinking Tag Typo Recovery

**Файл:** `copilot_tools/parser_0920_summary.py:259-264`

```python
command_str = (
    command_str
    .replace("<TINK>", "<THINK>")   # Частая опечатка
    .replace("</TINK>", "</THINK>")
    .replace("<think>", "<THINK>")  # Case normalization
    .replace("</think>", "</THINK>")
)
```

**Практический инсайт:**
- LLM часто делают предсказуемые ошибки (`TINK` vs `THINK`)
- Quantized модели (Q4/Q8) чаще ошибаются в spelling
- Robust parsing критичен для production

---

### Находка 17: Image Compression для эффективного логирования

**Файл:** `copilot_agent_server/local_server_logger.py:77-94`

```python
def save_image(self, image: Image.Image, image_name: str) -> str:
    buffered = BytesIO()
    image = image.convert('RGB')  # Убираем alpha channel
    image.save(buffered, format="JPEG", quality=85)  # Lossy compression
    image_data = buffered.getvalue()
    image_path = f"{self.image_dir}/{self.session_id}_{image_name}.jpeg"
```

**Оптимизация для масштаба:**
- PNG → JPEG экономит ~70% места
- quality=85 — баланс размер/качество
- Для RL: тысячи траекторий × сотни шагов = терабайты без сжатия

---

### Находка 18: MCP (Model Context Protocol) интеграция

**Файл:** `mcp_server/detailed_gelab_mcp_server.py:23-26`

```python
from fastmcp import FastMCP

mcp = FastMCP(name="Gelab-MCP-Server", instructions="""
This MCP server provides tools to interact with connected mobile devices using a GUI agent.
""")
```

**Интеграционный паттерн:**
- MCP — стандарт от Anthropic для tool use
- Позволяет подключать GUI Agent к Claude/другим LLM
- Для multi-agent RL: стандартизированный интерфейс между агентами

---

### Находка 19: Reasoning Content Extraction

**Файл:** `tools/ask_llm_v2.py:109-111`

```python
reasoning = completion.choices[0].message.get("reasoning_content", "")
if reasoning is not None and len(reasoning) > 0:
    result = "<think>" + reasoning + "</think>" + "\n" + result
```

**Поддержка reasoning models:**
- Некоторые модели возвращают отдельно `reasoning_content`
- Код объединяет reasoning + response в единый формат
- Совместимость с o1-style models

---

### Находка 20: App Package Mapping

**Файл:** `copilot_front_end/package_map.py` (упоминается в executor)

```python
from copilot_front_end.package_map import find_package_name

# В act_on_device:
app_name = frontend_action["value"]  # "微信"
package_name = find_package_name(app_name)  # "com.tencent.mm"
```

**Практический mapping:**
- 200+ приложений: название → package name
- Позволяет естественные команды: "открой WeChat" вместо "открой com.tencent.mm"
- Для RL: абстракция повышает sample efficiency

---

## Рекомендации для дальнейшего изучения

### Архитектурные паттерны для Async RL:

1. **A3C/IMPALA стиль**: Изучи `local_server_based_runner.py` — там реализован multi-worker rollout
2. **Experience Replay**: `LocalServerLogger` — персистентное хранение траекторий
3. **Verifier Integration**: `caption_current_screenshot` + `auto_reply` — два типа верификаторов

### Код для глубокого изучения:

| Концепция | Файл | Строки |
|-----------|------|--------|
| Agent Loop | `mcp_agent_loop.py` | 267-382 |
| Async Verification | `mcp_agent_loop.py` | 285-296 |
| Multi-worker | `local_server_based_runner.py` | 210-234 |
| State Abstraction | `parser_0920_summary.py` | 315-324 |
| Reward Model | `mcp_agent_loop.py` | 90-132 |

### Эксперименты:

1. Замени `auto_reply` на обученную reward model
2. Добавь RLHF на основе `historica_qa`
3. Реализуй distributed training с `CopilotClientRolloutRunner`

---

*Документ создан для изучения паттернов асинхронного RL на примере production-системы GELab-Zero*
