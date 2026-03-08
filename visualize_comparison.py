"""
Визуализация сравнения Perception vs Control параметров
"""
import matplotlib.pyplot as plt
import numpy as np

# Данные для сравнения
models = ['Ваша модель\n(текущая)', 'OpenAI VPT', 'AlphaStar', 'IMPALA', 'AlphaViT', 'Ваша модель\n(рекомендуемая)']

# Perception params (в миллионах)
perception = [3, 160, 80, 1.6, 86, 12]

# Control params (в миллионах)
control = [43, 90, 55, 0.8, 30, 24]

# Вычисляем ratios
ratios = [p/c for p, c in zip(perception, control)]

# === График 1: Stacked Bar Chart ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Stacked bars
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.6

p1 = ax1.bar(x, perception, width, label='Perception', color='#3b82f6', alpha=0.8)
p2 = ax1.bar(x, control, width, bottom=perception, label='Control', color='#ef4444', alpha=0.8)

ax1.set_ylabel('Parameters (millions)', fontsize=12, fontweight='bold')
ax1.set_title('Perception vs Control: Total Parameters', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15, ha='right')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Добавляем значения на bars
for i, (p, c) in enumerate(zip(perception, control)):
    total = p + c
    ax1.text(i, total + 5, f'{total:.1f}M', ha='center', fontsize=10, fontweight='bold')

# === График 2: Ratio comparison ===
ax2 = axes[0, 1]
colors = ['#dc2626' if r < 0.5 else '#facc15' if r < 1.5 else '#22c55e' for r in ratios]
bars = ax2.barh(models, ratios, color=colors, alpha=0.8)

# Добавляем reference lines
ax2.axvline(x=1.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Optimal min (1.5x)')
ax2.axvline(x=3.0, color='green', linestyle='--', linewidth=2, alpha=0.3, label='Optimal max (3.0x)')

ax2.set_xlabel('Perception / Control Ratio', fontsize=12, fontweight='bold')
ax2.set_title('Perception/Control Ratio Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='x', alpha=0.3)

# Добавляем значения
for i, (bar, ratio) in enumerate(zip(bars, ratios)):
    ax2.text(ratio + 0.1, i, f'{ratio:.2f}x', va='center', fontsize=10, fontweight='bold')

# === График 3: Pie charts для текущей и рекомендуемой моделей ===
ax3 = axes[1, 0]
# Текущая модель
current_perc = perception[0]
current_ctrl = control[0]
colors_pie = ['#3b82f6', '#ef4444']
explode = (0.05, 0)

wedges, texts, autotexts = ax3.pie(
    [current_perc, current_ctrl],
    labels=['Perception\n(3M)', 'Control\n(43M)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    explode=explode,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

ax3.set_title('Текущая модель (Ratio: 0.07x)', fontsize=14, fontweight='bold', color='#dc2626')

# === График 4: Рекомендуемая модель ===
ax4 = axes[1, 1]
rec_perc = perception[-1]
rec_ctrl = control[-1]

wedges, texts, autotexts = ax4.pie(
    [rec_perc, rec_ctrl],
    labels=['Perception\n(12M)', 'Control\n(24M)'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    explode=explode,
    textprops={'fontsize': 11, 'fontweight': 'bold'}
)

ax4.set_title('Рекомендуемая модель (Ratio: 0.5x)', fontsize=14, fontweight='bold', color='#facc15')

plt.tight_layout()
plt.savefig('perception_vs_control_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] График сохранен: perception_vs_control_comparison.png")

# === График 5: Детальный breakdown компонентов ===
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

components_current = ['RadarEncoder\n(EfficientNet-B0\n3 blocks)', 'YOLO\n(YOLOv11n\nfrozen)', 'YOLO Embeds', 'Temporal\nTransformer']
params_current = [0.348, 2.645, 0.021, 42.810]
colors_comp = ['#60a5fa', '#3b82f6', '#2563eb', '#ef4444']

components_rec = ['RadarEncoder\n(EfficientNet-B2\nfull)', 'YOLO\n(YOLOv11n\npartial unfreeze)', 'YOLO Embeds\n(upgraded)', 'Temporal\nTransformer\n(optimized)']
params_rec = [9.0, 3.0, 0.2, 24.0]

x_pos = np.arange(len(components_current))
width = 0.35

bars1 = ax.bar(x_pos - width/2, params_current, width, label='Текущая', color=colors_comp, alpha=0.7)
bars2 = ax.bar(x_pos + width/2, params_rec, width, label='Рекомендуемая', color=colors_comp, alpha=0.9)

ax.set_ylabel('Parameters (millions)', fontsize=12, fontweight='bold')
ax.set_title('Детальное сравнение компонентов', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(components_current, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Добавляем значения на bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('components_breakdown.png', dpi=300, bbox_inches='tight')
print("[OK] График сохранен: components_breakdown.png")

plt.show()

# === Печатаем summary ===
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nТекущая модель:")
print(f"  Perception:  {perception[0]:.1f}M params ({perception[0]/(perception[0]+control[0])*100:.1f}%)")
print(f"  Control:     {control[0]:.1f}M params ({control[0]/(perception[0]+control[0])*100:.1f}%)")
print(f"  Ratio:       {ratios[0]:.2f}x  {'[WARNING] KRITICHESKI NIZKO' if ratios[0] < 0.5 else '[OK] Normalno'}")

print(f"\nРекомендуемая модель:")
print(f"  Perception:  {perception[-1]:.1f}M params ({perception[-1]/(perception[-1]+control[-1])*100:.1f}%)")
print(f"  Control:     {control[-1]:.1f}M params ({control[-1]/(perception[-1]+control[-1])*100:.1f}%)")
print(f"  Ratio:       {ratios[-1]:.2f}x  {'[WARNING] Nizko' if ratios[-1] < 0.5 else '[OK] Horosho' if ratios[-1] < 1.5 else '[EXCELLENT] Otlichno'}")

print(f"\nIzmeneniya:")
print(f"  Perception:  {perception[0]:.1f}M -> {perception[-1]:.1f}M  (+{(perception[-1]/perception[0]-1)*100:.0f}%)")
print(f"  Control:     {control[0]:.1f}M -> {control[-1]:.1f}M  ({(control[-1]/control[0]-1)*100:.0f}%)")
total_change_pct = ((perception[-1]+control[-1])/(perception[0]+control[0])-1)*100
print(f"  Total:       {perception[0]+control[0]:.1f}M -> {perception[-1]+control[-1]:.1f}M  ({total_change_pct:.0f}%)")
print("="*70)
