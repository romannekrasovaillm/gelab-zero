# Детальный анализ GELab-Zero: Находки для Асинхронного RL

## Содержание

- [Введение](#введение)
- [Базовый уровень (1-4)](#-базовый-уровень)
- [Средний уровень (5-9)](#-средний-уровень)
- [Продвинутый уровень (10-14)](#-продвинутый-уровень)
- [Интересные находки (15-20)](#-интересные-находки)
- [Практические рекомендации](#практические-рекомендации)

---

## Введение

GELab-Zero — production-grade система для GUI-агентов на мобильных устройствах. Этот документ содержит **20 детальных технических находок**, извлечённых из кодовой базы для изучения паттернов асинхронного RL с верификаторами наград и рубриками.

**Ключевые концепции:**
- Асинхронный инференс с параллельными workers
- Верификация состояний через LLM (ORM и PRM)
- Рубрики принятия решений через INFO action
- Session-based управление траекториями

---

## 🟢 БАЗОВЫЙ УРОВЕНЬ

### Находка 1: Архитектура Agent Loop с отделением инференса от исполнения

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:267-382`

**Суть:** Цикл агента реализует классическую декомпозицию RL: `Observation → Policy → Action → Environment`. Инференс модели (Policy Server) полностью отделён от исполнения действий (Worker).

**Архитектура:**
```
┌──────────┐    ┌──────────────┐    ┌────────────────┐
│Environment│───▶│ Observation  │───▶│ Policy Server  │
│ (Phone)   │    │ (Screenshot) │    │ (LLM Inference)│
└─────▲────┘    └──────────────┘    └───────┬────────┘
      │                                      │
      └──────────────── Action ◀─────────────┘
```

**Значимость для async RL:**
- Один Policy Server обслуживает N workers
- Workers могут работать на разных устройствах
- Масштабирование через добавление workers, не затрагивая inference

---

### Находка 2: Session-based State Management

**Расположение:** `copilot_agent_server/local_server.py:35-69`

**Суть:** Каждая траектория (episode) изолирована через UUID-сессию. Данные сохраняются персистентно в JSONL, что позволяет:
- Восстанавливать прерванные траектории
- Продолжать сессии после INFO action
- Использовать логи для offline обучения

**Структура файлов:**
```
running_log/
├── server_log/
│   └── {session_id}.jsonl  ← Логи сессии
└── image_log/
    └── {session_id}_step_N.jpeg  ← Скриншоты
```

**Значимость:** Это основа для Experience Replay Buffer в RL.

---

### Находка 3: Structured Action Space Definition

**Расположение:** `copilot_tools/parser_0920_summary.py:23-52`

**Суть:** 9 дискретных действий с чёткими параметрами:

| Категория | Действия | Назначение |
|---|---|---|
| Interaction | CLICK, TYPE, SLIDE, LONGPRESS | UI взаимодействие |
| Navigation | AWAKE | Запуск приложений |
| Control | WAIT | Управление таймингом |
| Terminal | COMPLETE, ABORT | Завершение эпизода |
| **Verification** | **INFO** | **Запрос обратной связи** |

**Ключевое:** Действие INFO позволяет агенту явно запрашивать guidance от верификатора.

---

### Находка 4: Deterministic Action Parsing с Error Recovery

**Расположение:** `copilot_tools/parser_0920_summary.py:255-313`

**Суть:** Парсер преобразует свободный текст LLM в структурированное действие с robust обработкой ошибок:

- Исправление опечаток: `<TINK>` → `<THINK>`
- Нормализация регистра: `<think>` → `<THINK>`
- Fallback при отсутствии тегов

**Значимость:** Production robustness — система не падает из-за ошибок генерации LLM.

---

## 🟡 СРЕДНИЙ УРОВЕНЬ

### Находка 5: Concurrent Verification Thread (Асинхронный Верификатор)

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:285-326`

**Суть:** Пока агент думает (LLM inference), параллельно запускается верификатор состояния:

```python
# Запуск верификатора параллельно
caption_thread = threading.Thread(
    target=lambda: caption_current_screenshot(...)
)
caption_thread.start()

# Основной инференс агента
server_return = agent_server.automate_step(payload)

# Синхронизация
caption_thread.join()
caption_text = caption_result_container.get('caption', '')
```

**Архитектура:**
```
MAIN THREAD          CAPTION THREAD
     │                     │
     ├─ Start thread ─────▶│
     │                     │
     ├─ Agent inference    ├─ Caption inference
     │  (2 секунды)        │  (1 секунда)
     │                     │
     ├─ thread.join() ◀────┤
     │                     │
     ▼                     ▼
```

**Значимость:** Это паттерн **Actor-Critic** в реальном времени:
- Actor = Agent LLM (генерирует действие)
- Critic = Caption LLM (оценивает состояние)

---

### Находка 6: Markov State Abstraction через Summary

**Расположение:** `copilot_tools/parser_0920_summary.py:315-324`

**Суть:** Вместо передачи всей истории (O(n) памяти), используется только `summary` последнего действия (O(1)):

```python
# Markov assumption: достаточно summary + текущий скриншот
summary_history = ""
if len(actions) > 0:
    last_action = self.action2action(actions[-1])
    summary_history = last_action.get('summary', '')
```

**Преобразование:**
```
Полная история:           Markov abstraction:
[Screenshot_0, Action_0,  [Summary_N, Screenshot_N]
 Screenshot_1, Action_1,
 ...,
 Screenshot_N]
```

**Значимость:**
- Масштабируемость для async workers
- Аналог LSTM hidden state в recurrent RL
- Текстовый summary интерпретируем человеком

---

### Находка 7: Multi-Device Parallel Rollouts

**Расположение:** `copilot_agent_client/local_server_based_runner.py:15-234`

**Суть:** Оркестратор параллельных rollouts на множестве устройств:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Worker 1   │  │  Worker 2   │  │  Worker N   │
│  (Device 1) │  │  (Device 2) │  │  (Device N) │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                 ┌──────▼──────┐
                 │  DONE QUEUE │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   WRITER    │
                 │ (JSONL file)│
                 └─────────────┘
```

**Особенности:**
- Fault tolerance: при падении worker задача возвращается в очередь
- Deduplication: пропуск уже выполненных задач
- Checkpointing: инкрементальное сохранение результатов

---

### Находка 8: INFO Action как механизм Human-in-the-Loop Verification

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:342-368`

**Суть:** 4 режима обработки INFO action:

| Режим | Описание | Latency | Use Case |
|---|---|---|---|
| `auto_reply` | LLM как верификатор | ~2s | Автономные тесты |
| `no_reply` | Игнорирование | 0 | Стресс-тесты |
| `manual_reply` | Ввод человека | ∞ | Демо, отладка |
| `pass_to_client` | Асинхронная верификация | Variable | Production |

**Значимость:** Это **рубрика** — механизм для запроса и получения guidance от верификатора.

---

### Находка 9: Stop Reason Classification (Reward Signals)

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:426-438`

**Mapping к RL rewards:**

| Stop Reason | RL Interpretation | Reward |
|---|---|---|
| TASK_COMPLETED_SUCCESSFULLY | Goal reached | +1.0 |
| TASK_ABORTED_BY_AGENT | Self-recognized failure | -0.5 |
| MAX_STEPS_REACHED | Timeout | -0.2 |
| INFO_ACTION_NEEDS_REPLY | Pending verification | 0 |
| MANUAL_STOP | External cancellation | 0 |

---

## 🔴 ПРОДВИНУТЫЙ УРОВЕНЬ

### Находка 10: Chain-of-Thought как Interpretable Policy

**Расположение:** `copilot_tools/parser_0920_summary.py:71-97`

**Суть:** Каждое действие включает структурированный вывод:

```
<THINK>
1. Текущий экран показывает главную страницу WeChat
2. Вижу список контактов, нужно найти "Иван"
3. "Иван" находится по координатам (500, 400)
</THINK>
explain:Нажимаю на контакт Иван
action:CLICK
point:500,400
summary:Открыл чат с Иваном
```

**Компоненты:**
- `<THINK>`: Reasoning process (для Process Reward Model)
- `explain`: Human-readable justification
- `action + parameters`: Discrete action
- `summary`: Next state prediction (world model)

**Значимость:** Позволяет применять Process Reward Models для оценки каждого шага мышления.

---

### Находка 11: Auto-Reply Verifier как Learned Reward Model

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:30-88`

**Суть:** LLM играет роль "идеального пользователя", отвечая на вопросы агента:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"""
                # Роль: Пользователь GUI Agent
                # Задача: {task}
                # Вопрос агента: {info_action}
                # Требования: Краткий, информативный ответ
            """},
            {"type": "image_url", "image_url": {"url": current_image}},
        ]
    }
]
response = ask_llm_anything(messages)
```

**Значимость:** Это генеративный **Outcome Reward Model** — вместо скалярного reward возвращает текстовый guidance.

---

### Находка 12: Caption Model как State Verifier (Process Reward Model)

**Расположение:** `copilot_agent_client/mcp_agent_loop.py:90-132`

**Суть:** Caption model генерирует task-relevant описание текущего состояния:

```
Вход: Screenshot + Task
Выход: "Экран показывает список товаров в Taobao.
        Видны: Nike Air Max - 899₽, Puma RS-X - 599₽ (самый дешёвый)..."
```

**Значимость:**
- **Early Error Detection**: Можно обнаружить ошибку до конца эпизода
- **Intermediate Rewards**: Caption преобразуется в числовой reward
- **Semantic State Compression**: Image (6M params) → Text (~100 tokens)

---

### Находка 13: Q&A History Tracking для Multi-Turn Verification

**Расположение:** `copilot_tools/parser_0920_summary.py:332-365`

**Суть:** История вопросов и ответов сохраняется и передаётся агенту:

```
这是你和用户的对话历史：
你曾经提出的问题：Какой товар добавить?
用户对你的指示：Первый в списке

你曾经提出的问题：Какой размер?
用户对你的指示：XL

你需要更加注意用户最后的指示。  ← Следуй последнему ответу!
```

**Значимость:**
- **Constitutional AI**: История Q&A — это preference data для RLHF
- **Iterative Refinement**: Агент учится на коррекциях верификатора
- **Instruction Following**: Проверка соблюдения последней инструкции

---

### Находка 14: Asynchronous Session Continuation

**Расположение:** `mcp_server/detailed_gelab_mcp_server.py:62-162`

**Суть:** Траектория может быть прервана и продолжена асинхронно:

```
T0: ask_agent(task="Купить товар") → session_ABC
T1: Agent → INFO("Какой размер?") → stop_reason=INFO_NEEDS_REPLY
T2: [Пользователь думает...]
T3: ask_agent(session_id="ABC", reply="XL") → Продолжение!
```

**Значимость:**
- **Non-blocking**: Другие задачи выполняются пока ждём ответ
- **Fault Tolerance**: Логи на диске, восстановление после рестарта
- **Multi-Agent**: Один клиент управляет множеством сессий

---

## ✨ ИНТЕРЕСНЫЕ НАХОДКИ

### Находка 15: Coordinate Normalization (0-1000 Space)

**Расположение:** `copilot_front_end/pu_frontend_executor.py:48-56, 182-186`

**Суть:** Координаты нормализуются в диапазон [0-1000], независимо от разрешения:

```
Samsung (1440x3200): (720, 2800) px → (500, 875)
Pixel (1080x2400):   (540, 2100) px → (500, 875)
```

**Значимость:** Transfer learning между устройствами с разными разрешениями.

---

### Находка 16: Thinking Tag Typo Recovery

**Расположение:** `copilot_tools/parser_0920_summary.py:259-264`

**Суть:** Исправление типичных ошибок quantized моделей:
- `<TINK>` → `<THINK>` (пропущена H)
- `<think>` → `<THINK>` (регистр)
- `< THINK >` → `<THINK>` (пробелы)

**Значимость:** Позволяет использовать дешёвые Q4/Q8 модели в production.

---

### Находка 17: Image Compression для эффективного логирования

**Расположение:** `copilot_agent_server/local_server_logger.py:77-94`

**Суть:** PNG → JPEG (quality=85) даёт 93% экономию:
- PNG: ~4 MB
- JPEG: ~300 KB

**Значимость:**
- Replay Buffer: 10K × 4MB = 40GB (не влезает) → 10K × 300KB = 3GB (OK)
- Network Transfer: 8x быстрее

---

### Находка 18: MCP (Model Context Protocol) интеграция

**Расположение:** `mcp_server/detailed_gelab_mcp_server.py:1-27`

**Суть:** Стандартизированный протокол для подключения tools к LLM (Claude, GPT-4, etc.)

**Значимость:**
- Один сервер, любой MCP-совместимый клиент
- Multi-Agent orchestration через HTTP

---

### Находка 19: Reasoning Content Extraction

**Расположение:** `tools/ask_llm_v2.py:105-115`

**Суть:** Унификация форматов reasoning для разных моделей:

```python
# o1-style модель возвращает reasoning отдельно
reasoning = completion.choices[0].message.get("reasoning_content", "")
if reasoning:
    result = "<think>" + reasoning + "</think>\n" + result
```

**Значимость:** Model-agnostic training — можно использовать любые модели как teacher.

---

### Находка 20: App Package Mapping

**Расположение:** `copilot_front_end/package_map.py`

**Суть:** Словарь соответствия названий приложений и package names:
```python
"微信" → "com.tencent.mm"
"WeChat" → "com.tencent.mm"
```

**Значимость:**
- Natural Language Actions: модель генерирует "WeChat", не "com.tencent.mm"
- Sample Efficiency: проще обучить на человеческих названиях
- Expandability: добавление приложения без переобучения

---

## Практические рекомендации

### Для начинающих:
1. Изучи `mcp_agent_loop.py:267-382` — это сердце системы
2. Разберись с Session Management в `local_server.py`
3. Пойми Action Space в `parser_0920_summary.py`

### Для продвинутых:
1. Реализуй собственный reward model на базе `auto_reply()`
2. Добавь Process Reward Model, используя `caption_current_screenshot()`
3. Масштабируй через `CopilotClientRolloutRunner`

### Ключевые файлы для изучения:

| Концепция | Файл | Строки |
|---|---|---|
| Agent Loop | `mcp_agent_loop.py` | 267-382 |
| Async Verification | `mcp_agent_loop.py` | 285-326 |
| Multi-worker | `local_server_based_runner.py` | 210-234 |
| State Abstraction | `parser_0920_summary.py` | 315-324 |
| Reward Model | `mcp_agent_loop.py` | 30-132 |

---

*Документ создан для глубокого изучения паттернов асинхронного RL с верификаторами на примере production-системы GELab-Zero*
