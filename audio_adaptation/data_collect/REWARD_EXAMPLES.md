# Reward Calculation Examples for Offline RL

Примеры вычисления наград из извлеченных событий для offline RL обучения.

---

## Извлекаемые события

### Базовые события (sparse rewards)

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `kill` | sparse | `tick`, `victim_id`, `weapon`, `headshot` | +1.0 reward, +0.5 за headshot |
| `death` | sparse | `tick`, `killer_id`, `weapon` | -1.0 reward |
| `bomb_plant` | sparse | `tick` | +0.5 reward (T side) |
| `bomb_defuse` | sparse | `tick` | +2.0 reward (CT side) |
| `round_end` | sparse | `tick`, `winner`, `reason` | +2.0 если win, -2.0 если loss |

---

### Damage события (dense rewards)

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `damage_dealt` | dense | `tick`, `victim_id`, `damage`, `hitgroup`, `weapon` | +0.01 per HP |
| `damage_taken` | dense | `tick`, `attacker_id`, `damage`, `hitgroup`, `weapon` | -0.01 per HP |

---

### Weapon события (для accuracy)

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `weapon_fire` | dense | `tick`, `weapon` | Accuracy = damage_dealt / weapon_fire |
| `weapon_reload` | tactical | `tick`, `weapon` | Штраф за reload в неподходящее время |

---

### Grenade события (utility usage)

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `grenade_thrown` | tactical | `tick`, `grenade_type` | Track utility usage |
| `flashbang_detonate` | tactical | `tick` | Награда если после этого damage_dealt |
| `hegrenade_detonate` | tactical | `tick` | Награда если после этого damage_dealt |
| `smokegrenade_detonate` | tactical | `tick` | Награда если после этого successful push |
| `molotov_detonate` | tactical | `tick` | Награда если блокирует проход |

---

### Flash события (CRITICAL)

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `player_blind` | penalty | `tick`, `attacker_id`, `blind_duration` | **ШТРАФ** за ослепление! |

**Важно**: Если игрок был ослеплен → штраф -0.5 (или -0.1 * blind_duration)

---

### Bomb defuse события

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `bomb_begindefuse` | tactical | `tick` | Начало дефьюза |
| `bomb_abortdefuse` | penalty | `tick` | Прерывание → возможно плохое решение |

---

### Economic события

| Событие | Тип | Поля | Использование |
|---------|-----|------|---------------|
| `item_purchase` | economic | `tick`, `item` | Track economic decisions |

---

## Примеры вычисления наград

### Пример 1: Базовый sparse reward calculator

```python
class BasicRewardCalculator:
    """Простой калькулятор наград (только sparse events)."""

    def __init__(self):
        self.w_kill = 1.0
        self.w_death = -1.0
        self.w_headshot_bonus = 0.5
        self.w_round_win = 2.0
        self.w_round_loss = -2.0
        self.w_bomb_plant = 0.5
        self.w_bomb_defuse = 2.0

    def compute_rewards(self, events: List[Dict], player_team: str) -> np.ndarray:
        """
        Args:
            events: Список событий из раунда
            player_team: 'T' или 'CT'

        Returns:
            rewards: Массив наград по тикам
        """
        # Создать массив наград (tick -> reward)
        tick_rewards = {}

        for event in events:
            tick = event['tick']
            reward = 0.0

            if event['type'] == 'kill':
                reward += self.w_kill
                if event.get('headshot', False):
                    reward += self.w_headshot_bonus

            elif event['type'] == 'death':
                reward += self.w_death

            elif event['type'] == 'bomb_plant':
                if player_team == 'T':
                    reward += self.w_bomb_plant

            elif event['type'] == 'bomb_defuse':
                if player_team == 'CT':
                    reward += self.w_bomb_defuse

            elif event['type'] == 'round_end':
                winner = event['winner']
                if winner == player_team:
                    reward += self.w_round_win
                else:
                    reward += self.w_round_loss

            # Добавить награду к тику
            if tick not in tick_rewards:
                tick_rewards[tick] = 0.0
            tick_rewards[tick] += reward

        return tick_rewards


# Использование:
calc = BasicRewardCalculator()
rewards = calc.compute_rewards(round_data['events'], player_team='T')

# Пример:
# tick 1500: kill (+1.0)
# tick 1500: headshot (+0.5) → total = 1.5
# tick 2000: death (-1.0)
# tick 8500: round_end, winner='CT' (-2.0)
```

---

### Пример 2: Dense reward calculator (damage + accuracy)

```python
class DenseRewardCalculator:
    """Калькулятор наград с учетом урона и точности."""

    def __init__(self):
        # Sparse rewards
        self.w_kill = 1.0
        self.w_death = -1.0

        # Dense rewards
        self.w_damage_dealt = 0.01  # per 1 HP
        self.w_damage_taken = -0.01  # per 1 HP

        # Accuracy bonus
        self.w_accuracy_bonus = 0.5  # за высокую точность

        # Flash penalty
        self.w_blind_penalty = -0.5

    def compute_rewards(self, events: List[Dict]) -> Dict[int, float]:
        tick_rewards = {}

        # Track weapon fire count
        weapon_fires = 0
        damage_dealt_total = 0

        for event in events:
            tick = event['tick']
            reward = 0.0

            if event['type'] == 'kill':
                reward += self.w_kill

            elif event['type'] == 'death':
                reward += self.w_death

            elif event['type'] == 'damage_dealt':
                damage = event['damage']
                reward += self.w_damage_dealt * damage
                damage_dealt_total += damage

            elif event['type'] == 'damage_taken':
                damage = event['damage']
                reward += self.w_damage_taken * damage

            elif event['type'] == 'weapon_fire':
                weapon_fires += 1

            elif event['type'] == 'player_blind':
                # ШТРАФ за ослепление!
                blind_duration = event.get('blind_duration', 1.0)
                reward += self.w_blind_penalty * blind_duration

            if tick not in tick_rewards:
                tick_rewards[tick] = 0.0
            tick_rewards[tick] += reward

        # Accuracy bonus (в конце раунда)
        if weapon_fires > 0:
            accuracy = damage_dealt_total / (weapon_fires * 10)  # Примерный урон за выстрел
            if accuracy > 0.3:  # Если точность > 30%
                # Добавить бонус к последнему тику
                last_tick = max(tick_rewards.keys()) if tick_rewards else 0
                tick_rewards[last_tick] += self.w_accuracy_bonus * accuracy

        return tick_rewards
```

---

### Пример 3: Tactical reward calculator (grenades + utility)

```python
class TacticalRewardCalculator:
    """Награды за тактическое использование гранат."""

    def __init__(self):
        self.w_successful_flash = 0.3  # Если после флешки убийство
        self.w_successful_he = 0.2     # Если HE нанесла урон
        self.w_smoke_usage = 0.1       # За использование дыма
        self.w_molotov_usage = 0.1     # За использование молотова

    def compute_rewards(self, events: List[Dict]) -> Dict[int, float]:
        tick_rewards = {}

        # Tracking grenade effectiveness
        flash_ticks = []
        he_ticks = []

        for event in events:
            tick = event['tick']
            reward = 0.0

            if event['type'] == 'flashbang_detonate':
                flash_ticks.append(tick)

            elif event['type'] == 'hegrenade_detonate':
                he_ticks.append(tick)

            elif event['type'] == 'smokegrenade_detonate':
                reward += self.w_smoke_usage

            elif event['type'] == 'molotov_detonate':
                reward += self.w_molotov_usage

            # Check if flash was effective (kill within 3 seconds)
            elif event['type'] == 'kill':
                for flash_tick in flash_ticks:
                    if 0 < tick - flash_tick < 3 * 64:  # 3 seconds @ 64 tickrate
                        reward += self.w_successful_flash
                        flash_ticks.remove(flash_tick)
                        break

            # Check if HE was effective (damage within 1 second)
            elif event['type'] == 'damage_dealt':
                weapon = event.get('weapon', '')
                if 'hegrenade' in weapon:
                    for he_tick in he_ticks:
                        if 0 < tick - he_tick < 64:  # 1 second
                            reward += self.w_successful_he
                            he_ticks.remove(he_tick)
                            break

            if tick not in tick_rewards:
                tick_rewards[tick] = 0.0
            tick_rewards[tick] += reward

        return tick_rewards
```

---

### Пример 4: Complete reward calculator (все вместе)

```python
class CompleteRewardCalculator:
    """
    Полный калькулятор наград для offline RL.

    Комбинирует:
    - Sparse rewards (kills, deaths, round outcomes)
    - Dense rewards (damage, accuracy)
    - Tactical rewards (grenades, utility)
    - Penalties (blind, bad reloads)
    """

    def __init__(self):
        # Sparse
        self.w_kill = 1.0
        self.w_death = -1.0
        self.w_headshot_bonus = 0.5
        self.w_round_win = 2.0
        self.w_bomb_plant = 0.5

        # Dense
        self.w_damage = 0.01
        self.w_blind_penalty = -0.5

        # Tactical
        self.w_successful_flash = 0.3
        self.w_accuracy_bonus = 0.5

    def compute_rewards(
        self,
        events: List[Dict],
        player_team: str,
        states: List[Dict]
    ) -> np.ndarray:
        """
        Args:
            events: События раунда
            player_team: Команда игрока
            states: Состояния (для контекстных наград)

        Returns:
            rewards: Массив наград per-tick
        """
        tick_rewards = {}
        flash_ticks = []
        weapon_fires = 0
        damage_dealt_total = 0

        for event in events:
            tick = event['tick']
            reward = 0.0

            # Sparse rewards
            if event['type'] == 'kill':
                reward += self.w_kill
                if event.get('headshot', False):
                    reward += self.w_headshot_bonus

            elif event['type'] == 'death':
                reward += self.w_death

            elif event['type'] == 'round_end':
                if event['winner'] == player_team:
                    reward += self.w_round_win
                else:
                    reward -= self.w_round_win

            elif event['type'] == 'bomb_plant' and player_team == 'T':
                reward += self.w_bomb_plant

            # Dense rewards
            elif event['type'] == 'damage_dealt':
                damage = event['damage']
                reward += self.w_damage * damage
                damage_dealt_total += damage

            elif event['type'] == 'damage_taken':
                reward -= self.w_damage * event['damage']

            # Penalties
            elif event['type'] == 'player_blind':
                reward += self.w_blind_penalty * event.get('blind_duration', 1.0)

            # Tactical tracking
            elif event['type'] == 'flashbang_detonate':
                flash_ticks.append(tick)

            elif event['type'] == 'weapon_fire':
                weapon_fires += 1

            # Check flash effectiveness
            if event['type'] == 'kill':
                for flash_tick in flash_ticks:
                    if 0 < tick - flash_tick < 3 * 64:
                        reward += self.w_successful_flash
                        flash_ticks.remove(flash_tick)
                        break

            if tick not in tick_rewards:
                tick_rewards[tick] = 0.0
            tick_rewards[tick] += reward

        # Accuracy bonus
        if weapon_fires > 0:
            accuracy = damage_dealt_total / (weapon_fires * 10)
            if accuracy > 0.3:
                last_tick = max(tick_rewards.keys()) if tick_rewards else 0
                tick_rewards[last_tick] += self.w_accuracy_bonus * accuracy

        # Convert to array aligned with states
        rewards = np.zeros(len(states))
        for i, state in enumerate(states):
            state_tick = state['tick']
            if state_tick in tick_rewards:
                rewards[i] = tick_rewards[state_tick]

        return rewards


# Использование:
calc = CompleteRewardCalculator()

for round_data in demo['rounds']:
    rewards = calc.compute_rewards(
        events=round_data['events'],
        player_team=round_data['player_team'],
        states=round_data['states']
    )

    # rewards теперь можно использовать для вычисления returns
    returns = compute_monte_carlo_returns(rewards, gamma=0.99)
```

---

## Сводная таблица весов наград

Рекомендуемые веса для offline RL:

| Категория | Событие | Вес | Обоснование |
|-----------|---------|-----|-------------|
| **Sparse** | kill | +1.0 | Главная цель |
| | death | -1.0 | Главный штраф |
| | headshot_bonus | +0.5 | Более эффективно |
| | round_win | +2.0 | Итоговая цель |
| | round_loss | -2.0 | Итоговый штраф |
| | bomb_plant (T) | +0.5 | Тактическая цель |
| | bomb_defuse (CT) | +2.0 | Критическая цель |
| **Dense** | damage_dealt | +0.01 per HP | Прогресс к kill |
| | damage_taken | -0.01 per HP | Избегать урона |
| **Penalty** | player_blind | -0.5 | Критичный штраф! |
| | bad_reload | -0.1 | Неудачный тайминг |
| **Tactical** | successful_flash | +0.3 | Эффективная утилита |
| | successful_HE | +0.2 | Эффективная утилита |
| | smoke_usage | +0.1 | Тактическое применение |
| **Bonus** | accuracy_bonus | +0.5 | За точность >30% |

---

## Интеграция с обучением

### Шаг 1: Вычислить награды

```python
from REWARD_EXAMPLES import CompleteRewardCalculator

reward_calc = CompleteRewardCalculator()

# Для каждого раунда
for round_data in demo['rounds']:
    rewards = reward_calc.compute_rewards(
        events=round_data['events'],
        player_team=round_data['player_team'],
        states=round_data['states']
    )

    # Сохранить rewards обратно в round_data
    round_data['rewards'] = rewards.tolist()
```

### Шаг 2: Вычислить returns (Monte Carlo)

```python
def compute_returns(rewards, gamma=0.99):
    """Вычислить discounted returns."""
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Для каждого раунда
for round_data in demo['rounds']:
    rewards = np.array(round_data['rewards'])
    returns = compute_returns(rewards, gamma=0.99)
    round_data['returns'] = returns.tolist()
```

### Шаг 3: Использовать в IQL loss

```python
# В Train.py (Фаза 2.1.3)
def train_step(batch):
    # ... forward pass

    # IQL value loss
    returns = batch['returns'].to(device)  # Ground truth returns
    value_pred = value.squeeze(-1)

    diff = returns - value_pred
    weight = torch.where(diff > 0, TAU, 1 - TAU)  # Expectile regression
    loss_value = (weight * diff ** 2).mean()

    # Total loss
    loss = loss_mouse + loss_keys + ALPHA_VALUE * loss_value
```

---

## Важные замечания

### 1. Ослепление (player_blind) - КРИТИЧНО!

Ослепление флешкой **очень важно** для RL:
- Если игрок ослеплен → **штраф -0.5**
- Модель должна научиться **избегать** флешек
- Или научиться **отворачиваться** при флешке

### 2. Accuracy tracking

Accuracy (точность стрельбы) важна:
- Accuracy = damage_dealt / (weapon_fire * avg_damage_per_shot)
- Если accuracy > 30% → **бонус +0.5**
- Модель научится **точнее целиться**

### 3. Tactical utility usage

Эффективное использование гранат:
- Flash → Kill (в течение 3 сек) → **+0.3 reward**
- HE → Damage → **+0.2 reward**
- Модель научится **правильно использовать** утилиту

### 4. Economic management

Покупки (`item_purchase`) можно использовать для:
- Tracking экономических решений
- Штраф за **плохие покупки** (например, AWP при 0-5 score)

---

## Следующие шаги

1. ✅ Парсинг событий завершен (universal_demo_parser.py)
2. ✅ Примеры вычисления наград созданы (этот файл)
3. ⏳ **TODO**: Реализовать `RewardCalculator.py` (Фаза 2.1.1)
4. ⏳ **TODO**: Интегрировать в `DatasetIntent.py`
5. ⏳ **TODO**: Добавить IQL loss в `Train.py` (Фаза 2.1.3)
