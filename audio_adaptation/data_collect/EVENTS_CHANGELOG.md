# Events Changelog - Что добавлено

## ✅ Новые события для Offline RL

Расширен парсер событий в [universal_demo_parser.py](universal_demo_parser.py) для более точного вычисления наград.

---

## До (v1.0) - Базовые события

Только критические sparse rewards:
- ✅ `kill` - убийства
- ✅ `death` - смерти
- ✅ `damage_dealt` / `damage_taken` - урон
- ✅ `bomb_plant` / `bomb_defuse` - бомба
- ✅ `round_end` - конец раунда

**Проблема**: Недостаточно данных для dense rewards и tactical decisions.

---

## После (v2.0) - Полный набор событий

### 1. Weapon Events (Accuracy Tracking)

**Добавлено**:
- ✅ `weapon_fire` - каждый выстрел
  ```json
  {"type": "weapon_fire", "tick": 1234, "weapon": "ak-47"}
  ```

- ✅ `weapon_reload` - перезарядка
  ```json
  {"type": "weapon_reload", "tick": 1500, "weapon": "ak-47"}
  ```

**Использование**:
```python
# Вычислить accuracy
weapon_fires = len([e for e in events if e['type'] == 'weapon_fire'])
damage_dealt = sum(e['damage'] for e in events if e['type'] == 'damage_dealt')
accuracy = damage_dealt / (weapon_fires * 10)  # Примерный урон за выстрел

# Reward за высокую точность
if accuracy > 0.3:
    reward += 0.5
```

---

### 2. Flash Events (CRITICAL!)

**Добавлено**:
- ✅ `player_blind` - игрок ослеплен флешкой
  ```json
  {
    "type": "player_blind",
    "tick": 2000,
    "attacker_id": 456,
    "blind_duration": 2.5
  }
  ```

**Использование**:
```python
# ШТРАФ за ослепление
for event in events:
    if event['type'] == 'player_blind':
        blind_duration = event['blind_duration']
        reward -= 0.5 * blind_duration  # Критический штраф!
```

**Важность**: Модель научится **избегать флешек** или **отворачиваться**.

---

### 3. Grenade Events (Tactical Utility)

**Добавлено**:
- ✅ `grenade_thrown` - бросок гранаты
  ```json
  {"type": "grenade_thrown", "tick": 1800, "grenade_type": "flashbang"}
  ```

- ✅ `flashbang_detonate` - взрыв флешки
  ```json
  {"type": "flashbang_detonate", "tick": 1850}
  ```

- ✅ `hegrenade_detonate` - взрыв HE
  ```json
  {"type": "hegrenade_detonate", "tick": 1900}
  ```

- ✅ `smokegrenade_detonate` - дым
  ```json
  {"type": "smokegrenade_detonate", "tick": 2000}
  ```

- ✅ `molotov_detonate` - молотов
  ```json
  {"type": "molotov_detonate", "tick": 2100}
  ```

**Использование**:
```python
# Награда за эффективную флешку
flash_ticks = [e['tick'] for e in events if e['type'] == 'flashbang_detonate']
for event in events:
    if event['type'] == 'kill':
        # Если kill в течение 3 сек после флешки
        for flash_tick in flash_ticks:
            if 0 < event['tick'] - flash_tick < 3 * 64:  # 3 sec @ 64 tickrate
                reward += 0.3  # Бонус за успешную флешку
                break
```

---

### 4. Bomb Defuse Events

**Добавлено**:
- ✅ `bomb_begindefuse` - начало дефьюза
  ```json
  {"type": "bomb_begindefuse", "tick": 7000}
  ```

- ✅ `bomb_abortdefuse` - прерывание дефьюза
  ```json
  {"type": "bomb_abortdefuse", "tick": 7050}
  ```

**Использование**:
```python
# Штраф за прерванный дефьюз (плохое решение)
for event in events:
    if event['type'] == 'bomb_abortdefuse':
        reward -= 0.2  # Возможно плохой тайминг
```

---

### 5. Economic Events

**Добавлено**:
- ✅ `item_purchase` - покупки
  ```json
  {"type": "item_purchase", "tick": 500, "item": "ak-47"}
  ```

**Использование**:
```python
# Tracking экономических решений
purchases = [e for e in events if e['type'] == 'item_purchase']

# Можно добавить штраф за плохие покупки
# Например: AWP при 0-5 score
for purchase in purchases:
    if purchase['item'] == 'awp' and score[0] < score[1]:
        reward -= 0.3  # Плохая покупка
```

---

## Сравнение v1.0 vs v2.0

| Категория | v1.0 | v2.0 | Улучшение |
|-----------|------|------|-----------|
| Sparse events | 7 | 7 | - |
| Dense events | 2 | 4 | +2 (weapon_fire, player_blind) |
| Tactical events | 2 | 9 | +7 (grenades, defuse, purchases) |
| **TOTAL** | **9** | **20** | **+11 событий** |

---

## Ключевые улучшения

### 1. Accuracy Tracking ✅

**До**: Нет информации о выстрелах → нельзя вычислить точность.

**После**: `weapon_fire` события → можно вычислить `accuracy = damage / fires`.

**Reward**:
```python
if accuracy > 0.3:
    reward += 0.5  # Бонус за точность
```

---

### 2. Flash Penalty ✅ (КРИТИЧНО!)

**До**: Нет информации об ослеплении → модель не понимает что флешки плохо.

**После**: `player_blind` события → **штраф** за ослепление.

**Reward**:
```python
if event['type'] == 'player_blind':
    reward -= 0.5 * blind_duration  # Критический штраф!
```

**Эффект**: Модель научится **избегать флешек** или **отворачиваться**.

---

### 3. Tactical Utility Effectiveness ✅

**До**: Нет информации об эффективности гранат → модель не понимает когда использовать.

**После**: `flashbang_detonate`, `hegrenade_detonate` → можно вычислить эффективность.

**Reward**:
```python
# Если после флешки убийство
if kill_after_flash:
    reward += 0.3  # Успешная флешка!
```

**Эффект**: Модель научится **правильно использовать** гранаты.

---

### 4. Economic Awareness ✅

**До**: Нет информации о покупках.

**После**: `item_purchase` → можно трекать экономические решения.

**Reward**:
```python
# Штраф за AWP при плохом счёте
if purchase == 'awp' and losing:
    reward -= 0.3
```

---

## Проверка корректности

Используй [test_pipeline.py](test_pipeline.py) для проверки:

```bash
python test_pipeline.py --demo "C:/path/to/match.dem" --steam-id 76561198386265483
```

Должен вывести:
```
✓ Event types: kill, death, damage_dealt, damage_taken, weapon_fire, player_blind, grenade_thrown, flashbang_detonate, hegrenade_detonate, smokegrenade_detonate, molotov_detonate, bomb_begindefuse, bomb_plant, bomb_defuse, round_end
```

---

## Интеграция с RewardCalculator

См. [REWARD_EXAMPLES.md](REWARD_EXAMPLES.md) для полных примеров вычисления наград.

**Быстрый пример**:
```python
from REWARD_EXAMPLES import CompleteRewardCalculator

calc = CompleteRewardCalculator()

for round_data in demo['rounds']:
    rewards = calc.compute_rewards(
        events=round_data['events'],
        player_team=round_data['player_team'],
        states=round_data['states']
    )

    # rewards: массив наград per-tick
    # Использовать для вычисления returns и IQL training
```

---

## Следующие шаги

1. ✅ **Парсинг событий** - ГОТОВО (universal_demo_parser.py)
2. ✅ **Примеры rewards** - ГОТОВО (REWARD_EXAMPLES.md)
3. ⏳ **TODO**: Реализовать `RewardCalculator.py` (Фаза 2.1.1)
4. ⏳ **TODO**: Интегрировать в `DatasetIntent.py` (добавить rewards/returns)
5. ⏳ **TODO**: Добавить IQL loss в `Train.py` (Фаза 2.1.3)

---

## Критические события для RL

**Top 5 самых важных**:

1. 🔥 **player_blind** - ослепление (штраф!)
2. 🎯 **weapon_fire** - для accuracy tracking
3. 💣 **flashbang_detonate** - для tactical utility
4. 💀 **kill** / **death** - основные sparse rewards
5. 🩸 **damage_dealt** - для dense rewards

---

## FAQ

**Q: Зачем так много событий?**

A: Offline RL требует **плотных наград** для обучения. Sparse rewards (только kill/death) недостаточно для convergence.

**Q: Какие события самые важные?**

A: **player_blind** (критично!), **weapon_fire** (accuracy), **damage_dealt** (dense).

**Q: Можно ли использовать только sparse rewards?**

A: Да, но обучение будет **медленным**. Dense rewards (damage, accuracy, grenades) ускоряют convergence в 5-10x.

**Q: Что делать если событие не парсится?**

A: Проверь версию demoparser2:
```bash
pip install --upgrade demoparser2
```

Некоторые события доступны только в новых версиях.
