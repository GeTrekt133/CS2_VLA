"""
Скрипт для подсчета параметров в модели CS2 NN Agent
Сравниваем perception (RadarEncoder + YOLO) vs control (TemporalTransformer)
"""
import torch
import sys
sys.path.append('src')

from RadarEncoder import RadarEncoderEffB0
from Yolo import DetectionModel
from TemporalTransformer import TemporalCrossTransformer


def count_params(model, name):
    """Подсчет параметров модели"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\n{name}:")
    print(f"  Total params:      {total:,}")
    print(f"  Trainable params:  {trainable:,}")
    print(f"  Frozen params:     {frozen:,}")

    return total, trainable, frozen


def main():
    print("=" * 70)
    print("CS2 NN Agent - Parameter Count Analysis")
    print("=" * 70)

    # === RadarEncoder ===
    radar_encoder = RadarEncoderEffB0(pretrained=False, in_ch=3, out_ch=64, embed_dim=512)
    radar_total, radar_train, radar_frozen = count_params(radar_encoder, "RadarEncoder (EfficientNet-B0)")

    # === YOLO DetectionModel ===
    yolo = DetectionModel(cfg='src/yolo11n.yaml', ch=3, nc=2)
    yolo_total, yolo_train, yolo_frozen = count_params(yolo, "YOLO Detection Model (YOLOv11n)")

    # YOLO embeds отдельно (только это обучается)
    yolo_embeds_params = sum(p.numel() for p in yolo.embeds.parameters())
    print(f"  YOLO embeds only:  {yolo_embeds_params:,}")

    # === TemporalTransformer ===
    temporal = TemporalCrossTransformer(
        radar_dim=512, radar_seq=129,
        scene_dim=2048, scene_seq=16,
        detection_dim=100, detection_seq=1,
        actions_dim=22, actions_seq=16,
        state_dim=95,
        d_model=512, num_heads=8, depth=6
    )
    temporal_total, temporal_train, temporal_frozen = count_params(temporal, "TemporalCrossTransformer (Control)")

    # === Анализ соотношений ===
    print("\n" + "=" * 70)
    print("PERCEPTION vs CONTROL Analysis")
    print("=" * 70)

    perception_total = radar_total + yolo_total
    control_total = temporal_total

    print(f"\nPerception Modules (RadarEncoder + YOLO):")
    print(f"  Total: {perception_total:,}")
    print(f"    - RadarEncoder:   {radar_total:,} ({radar_total/perception_total*100:.1f}%)")
    print(f"    - YOLO:           {yolo_total:,} ({yolo_total/perception_total*100:.1f}%)")

    print(f"\nControl Module (TemporalTransformer):")
    print(f"  Total: {control_total:,}")

    ratio = perception_total / control_total
    print(f"\n{'='*70}")
    print(f"PERCEPTION / CONTROL RATIO: {ratio:.2f}x")
    print(f"{'='*70}")

    print(f"\nВывод:")
    if ratio < 1.5:
        print(f"⚠️  Perception модули относительно слабые (ratio = {ratio:.2f})")
        print(f"    Рекомендация: увеличить размер perception моделей")
    elif ratio < 3:
        print(f"⚡ Сбалансированное соотношение (ratio = {ratio:.2f})")
    else:
        print(f"✅ Perception доминирует (ratio = {ratio:.2f})")

    # === Детальный breakdown по компонентам ===
    print(f"\n{'='*70}")
    print("Detailed Component Breakdown:")
    print(f"{'='*70}")

    total_model = perception_total + control_total
    print(f"\nRadarEncoder:          {radar_total:>12,} params ({radar_total/total_model*100:>5.1f}%)")
    print(f"YOLO (backbone+head):  {yolo_total:>12,} params ({yolo_total/total_model*100:>5.1f}%)")
    print(f"  └─ YOLO embeds:      {yolo_embeds_params:>12,} params ({yolo_embeds_params/total_model*100:>5.1f}%)")
    print(f"TemporalTransformer:   {temporal_total:>12,} params ({temporal_total/total_model*100:>5.1f}%)")
    print(f"{'-'*70}")
    print(f"TOTAL:                 {total_model:>12,} params")


if __name__ == "__main__":
    main()
