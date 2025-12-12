# RT-DETRv4 YAML設定ファイル完全ガイド

このドキュメントでは、RT-DETRv4の学習に使用する全YAMLファイルの構造と役割を詳細に説明します。

---

## 📁 ディレクトリ構造

```
configs/
├── runtime.yml                        # 実行時の基本設定
├── base/                              # 基本設定ファイル群
│   ├── dataloader.yml                # データローダーの設定
│   ├── optimizer.yml                 # オプティマイザーの設定
│   ├── dfine_hgnetv2.yml            # D-FINE + HGNetv2のモデル設定
│   ├── rtv4.yml                      # RT-DETRv4の基本設定（拡張版）
│   ├── deim.yml                      # DEIMの追加設定
│   ├── rtv2_r50vd.yml               # RT-DETRv2 + ResNet50の設定
│   ├── rtv2_optimizer.yml           # RT-DETRv2用オプティマイザー
│   └── rtv2_deim.yml                # RT-DETRv2 + DEIM設定
├── dataset/                          # データセット設定ファイル群
│   ├── coco_detection.yml           # COCO2017データセット設定
│   ├── custom_detection.yml         # カスタムデータセット設定
│   ├── voc_detection.yml            # PASCAL VOCデータセット設定
│   ├── obj365_detection.yml         # Objects365データセット設定
│   └── crowdhuman_detection.yml     # CrowdHumanデータセット設定
├── dfine/                            # D-FINEモデルの設定
│   ├── dfine_hgnetv2_n_coco.yml     # D-FINE Nanoモデル
│   ├── dfine_hgnetv2_s_coco.yml     # D-FINE Smallモデル
│   ├── dfine_hgnetv2_m_coco.yml     # D-FINE Mediumモデル
│   ├── dfine_hgnetv2_l_coco.yml     # D-FINE Largeモデル
│   ├── dfine_hgnetv2_x_coco.yml     # D-FINE Xtra-Largeモデル
│   └── object365/                   # Objects365用設定
├── rtv4/                             # RT-DETRv4モデルの設定
│   ├── rtv4_hgnetv2_s_coco.yml      # RT-DETRv4 Smallモデル
│   ├── rtv4_hgnetv2_s_coco_customed.yml  # カスタム設定版
│   ├── rtv4_hgnetv2_m_coco.yml      # RT-DETRv4 Mediumモデル
│   ├── rtv4_hgnetv2_l_coco.yml      # RT-DETRv4 Largeモデル
│   └── rtv4_hgnetv2_x_coco.yml      # RT-DETRv4 Xtra-Largeモデル
├── rtv2/                             # RT-DETRv2モデルの設定
│   ├── rtv2_r18vd_120e_coco.yml     # ResNet18ベース
│   ├── rtv2_r34vd_120e_coco.yml     # ResNet34ベース
│   ├── rtv2_r50vd_6x_coco.yml       # ResNet50ベース
│   ├── rtv2_r101vd_6x_coco.yml      # ResNet101ベース
│   └── rtv2_r50vd_m_7x_coco.yml     # ResNet50-M改良版
└── deim/                             # DEIMモデルの設定
    ├── deim_hgnetv2_n_coco.yml      # DEIM Nanoモデル
    ├── deim_hgnetv2_s_coco.yml      # DEIM Smallモデル
    ├── deim_hgnetv2_m_coco.yml      # DEIM Mediumモデル
    ├── deim_hgnetv2_l_coco.yml      # DEIM Largeモデル
    ├── deim_hgnetv2_x_coco.yml      # DEIM Xtra-Largeモデル
    └── object365/                   # Objects365用設定
```

---

## 🔧 1. ルートディレクトリの設定ファイル

### `configs/runtime.yml`

**役割**: 学習実行時の基本的なランタイム設定

**主な設定項目**:

| パラメータ | 説明 | デフォルト値 |
|-----------|------|------------|
| `print_freq` | ログ出力の間隔（イテレーション数） | `100` |
| `output_dir` | 出力ディレクトリのパス | `'./logs'` |
| `checkpoint_freq` | チェックポイント保存の間隔（エポック数） | `12` |
| `sync_bn` | 分散学習時のBatch Normalizationの同期 | `True` |
| `find_unused_parameters` | DDP使用時の未使用パラメータ検出 | `False` |
| `use_amp` | 自動混合精度（AMP）の使用 | `False` |
| `use_ema` | Exponential Moving Average（EMA）の使用 | `False` |
| `ema.decay` | EMAの減衰率 | `0.9999` |
| `ema.warmups` | EMAのウォームアップステップ数 | `1000` |
| `scaler.type` | AMPのスケーラー型 | `GradScaler` |

**設定例**:
```yaml
print_freq: 100
output_dir: './outputs/my_experiment'
checkpoint_freq: 4
use_amp: True
use_ema: True
```

---

## 📦 2. `configs/base/` - 基本設定ファイル群

### `configs/base/dataloader.yml`

**役割**: データ拡張、データローダーのパラメータ設定

**主な設定項目**:

#### 訓練用データローダー（`train_dataloader`）

| パラメータ | 説明 | 設定値例 |
|-----------|------|----------|
| `dataset.transforms.ops` | データ拡張の適用リスト | RandomFlip, Resize, Mosaic等 |
| `dataset.transforms.policy.epoch` | データ拡張を停止するエポック | `72` |
| `collate_fn.base_size` | 基本入力サイズ | `640` |
| `collate_fn.stop_epoch` | マルチスケール訓練を停止するエポック | `72` |
| `total_batch_size` | 全GPU合計のバッチサイズ | `32` |
| `num_workers` | データローダーのワーカー数 | `4` |
| `shuffle` | データのシャッフル有無 | `True` |

#### 検証用データローダー（`val_dataloader`）

| パラメータ | 説明 | 設定値例 |
|-----------|------|----------|
| `dataset.transforms.ops` | データ前処理（Resizeのみ等） | Resize等 |
| `total_batch_size` | バッチサイズ | `64` |
| `shuffle` | シャッフル有無 | `False` |

**データ拡張の種類**:
- `RandomPhotometricDistort`: ランダムな色調変換
- `RandomZoomOut`: ランダムズームアウト
- `RandomIoUCrop`: IoUベースのランダムクロップ
- `RandomHorizontalFlip`: 水平反転
- `Mosaic`: Mosaicデータ拡張
- `Resize`: リサイズ
- `SanitizeBoundingBoxes`: バウンディングボックスの検証

---

### `configs/base/optimizer.yml`

**役割**: オプティマイザー、学習率スケジューラーの設定

**主な設定項目**:

| パラメータ | 説明 | デフォルト値 |
|-----------|------|------------|
| `epoches` | 総エポック数 | `72` |
| `clip_max_norm` | 勾配クリッピングの閾値 | `0.1` |
| `optimizer.type` | オプティマイザーの種類 | `AdamW` |
| `optimizer.lr` | 基本学習率 | `0.00025` |
| `optimizer.betas` | AdamWのbetaパラメータ | `[0.9, 0.999]` |
| `optimizer.weight_decay` | 重み減衰係数 | `0.000125` |
| `optimizer.params` | レイヤー別の学習率設定 | 正規表現で指定 |
| `lr_scheduler.type` | 学習率スケジューラーの種類 | `MultiStepLR` |
| `lr_scheduler.milestones` | 学習率を下げるステップ | `[500]` |
| `lr_warmup_scheduler.warmup_duration` | ウォームアップの期間 | `500` |

**レイヤー別学習率の設定例**:
```yaml
optimizer:
  params:
    - params: '^(?=.*backbone)(?!.*norm).*$'  # バックボーン（norm以外）
      lr: 0.0000125
    - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'  # エンコーダー/デコーダーのnorm
      weight_decay: 0.
```

---

### `configs/base/dfine_hgnetv2.yml`

**役割**: D-FINE + HGNetv2バックボーンのモデルアーキテクチャ設定

**主な設定項目**:

#### タスク設定
- `task`: `detection`（物体検出）
- `model`: `RTv4`（モデル名）
- `criterion`: `RTv4Criterion`（損失関数）
- `postprocessor`: `PostProcessor`（後処理）

#### モデル構造（`RTv4`）
- `backbone`: `HGNetv2`（バックボーン）
- `encoder`: `HybridEncoder`（エンコーダー）
- `decoder`: `DFINETransformer`（デコーダー）

#### バックボーン（`HGNetv2`）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `pretrained` | 事前学習済み重みの使用 | `True` |
| `local_model_dir` | 事前学習モデルのパス | `./pretrain/hgnetv2/` |

#### エンコーダー（`HybridEncoder`）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `in_channels` | 入力チャネル数リスト | `[512, 1024, 2048]` |
| `feat_strides` | 特徴マップのストライドリスト | `[8, 16, 32]` |
| `hidden_dim` | 隠れ層の次元数 | `256` |
| `use_encoder_idx` | 使用するエンコーダー層のインデックス | `[2]` |
| `num_encoder_layers` | エンコーダー層の数 | `1` |
| `nhead` | アテンションヘッド数 | `8` |
| `dim_feedforward` | フィードフォワード層の次元数 | `1024` |
| `expansion` | チャネル拡張率 | `1.0` |
| `depth_mult` | 深さの倍率 | `1` |
| `act` | 活性化関数 | `'silu'` |

#### デコーダー（`DFINETransformer`）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `feat_channels` | 特徴チャネル数リスト | `[256, 256, 256]` |
| `feat_strides` | 特徴マップのストライドリスト | `[8, 16, 32]` |
| `hidden_dim` | 隠れ層の次元数 | `256` |
| `num_levels` | マルチスケールレベル数 | `3` |
| `num_layers` | デコーダー層の数 | `6` |
| `eval_idx` | 評価に使用する層のインデックス | `-1`（最終層） |
| `num_queries` | クエリ数 | `300` |
| `num_denoising` | デノイジングクエリ数 | `100` |
| `label_noise_ratio` | ラベルノイズの比率 | `0.5` |
| `box_noise_scale` | ボックスノイズのスケール | `1.0` |
| `reg_max` | 回帰の最大値 | `32` |
| `num_points` | サンプリングポイント数 | `[3, 6, 3]` |

#### 損失関数（`RTv4Criterion`）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `weight_dict` | 各損失の重み | `{loss_vfl: 1, loss_bbox: 5, loss_giou: 2, ...}` |
| `losses` | 使用する損失のリスト | `['vfl', 'boxes', 'local']` |
| `alpha` | Focal Lossのαパラメータ | `0.75` |
| `gamma` | Focal Lossのγパラメータ | `2.0` |
| `reg_max` | 回帰の最大値 | `32` |

---

### `configs/base/rtv4.yml`

**役割**: RT-DETRv4独自の拡張設定（知識蒸留、追加のデータ拡張等）

**主な設定項目**:

#### データ拡張の拡張

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `train_dataloader.dataset.transforms.ops` | Mosaic等の追加 | Mosaic拡張を含む |
| `train_dataloader.dataset.transforms.policy.epoch` | 複数エポックでの段階的停止 | `[4, 29, 50]` |
| `train_dataloader.dataset.transforms.mosaic_prob` | Mosaic適用確率 | `0.5` |
| `train_dataloader.collate_fn.mixup_prob` | MixUp適用確率 | `0.5` |
| `train_dataloader.collate_fn.mixup_epochs` | MixUp適用期間 | `[4, 29]` |

#### バックボーンの設定解除

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `HGNetv2.freeze_at` | 凍結するレイヤー（-1で凍結なし） | `-1` |
| `HGNetv2.freeze_norm` | Normalization層の凍結 | `False` |

#### デコーダーの活性化関数

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `DFINETransformer.activation` | アテンション活性化関数 | `silu` |
| `DFINETransformer.mlp_act` | MLP活性化関数 | `silu` |

#### 学習率スケジューラー（カスタム）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `lrsheduler` | スケジューラーの種類 | `flatcosine` |
| `lr_gamma` | 減衰率 | `0.5` |
| `warmup_iter` | ウォームアップステップ数 | `2000` |
| `flat_epoch` | フラット期間のエポック数 | `29` |
| `no_aug_epoch` | データ拡張停止前のエポック数 | `8` |

#### 損失関数の拡張

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `RTv4Criterion.weight_dict` | 損失の重み（知識蒸留を含む） | `{loss_mal: 1, ..., loss_distill: 10.0}` |
| `RTv4Criterion.losses` | 使用する損失 | `['mal', 'boxes', 'local', 'distill']` |
| `RTv4Criterion.gamma` | Focal Lossのγ | `1.5` |

---

### `configs/base/deim.yml`

**役割**: DEIM（DETR with Improved Matching）の追加設定

**内容**: `rtv4.yml`とほぼ同じ構造で、知識蒸留の損失重みが異なる場合があります。

**主な違い**:
- `loss_distill`の重みが調整されている場合がある
- モデルのマッチング戦略が異なる

---

### `configs/base/rtv2_r50vd.yml`

**役割**: RT-DETRv2 + ResNet50-D バックボーンの設定

**主な設定項目**:

#### バックボーン（`PResNet`）

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `depth` | ResNetの深さ | `50` |
| `variant` | 変種（a/b/c/d） | `d` |
| `freeze_at` | 凍結するステージ | `0` |
| `return_idx` | 返す特徴マップのインデックス | `[1, 2, 3]` |
| `freeze_norm` | Normalization層の凍結 | `True` |
| `pretrained` | 事前学習済み重みの使用 | `True` |

#### デコーダー（`RTDETRTransformerv2`）

- RT-DETRv2専用のTransformer
- `num_points`: `[4, 4, 4]`
- `cross_attn_method`: `default`
- `query_select_method`: `default`

---

### `configs/base/rtv2_optimizer.yml`

**役割**: RT-DETRv2用のオプティマイザー設定

**主な違い**:
- `total_batch_size`: `16`（RT-DETRv4より小さい）
- `lr`: `0.0001`（学習率が異なる）
- `ema.warmups`: `2000`

---

### `configs/base/rtv2_deim.yml`

**役割**: RT-DETRv2 + DEIMの追加設定

**内容**: RT-DETRv2用のデータ拡張とスケジューラー設定を含む。

---

## 📊 3. `configs/dataset/` - データセット設定ファイル群

### `configs/dataset/coco_detection.yml`

**役割**: COCO2017データセットの設定

**主な設定項目**:

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `task` | タスクの種類 | `detection` |
| `num_classes` | クラス数 | `80` |
| `remap_mscoco_category` | COCOカテゴリの再マッピング | `True` |
| `evaluator.type` | 評価器の種類 | `CocoEvaluator` |
| `evaluator.iou_types` | 評価するIoUタイプ | `['bbox']` |

#### 訓練データローダー

```yaml
train_dataloader:
  dataset:
    type: CocoDetection
    img_folder: /root/share/data/COCO2017/train2017/
    ann_file: /root/share/data/COCO2017/annotations/instances_train2017.json
    return_masks: False
```

#### 検証データローダー

```yaml
val_dataloader:
  dataset:
    img_folder: /root/share/data/COCO2017/val2017/
    ann_file: /root/share/data/COCO2017/annotations/instances_val2017.json
```

---

### `configs/dataset/custom_detection.yml`

**役割**: カスタムデータセット（COCO形式）の設定

**主な設定項目**:

| パラメータ | 説明 | 設定値例 |
|-----------|------|----------|
| `num_classes` | **カスタムデータセットのクラス数** | `1` |
| `remap_mscoco_category` | カテゴリ再マッピング（Falseに設定） | `False` |

#### データパス

```yaml
train_dataloader:
  dataset:
    img_folder: coco_dataset/images/train
    ann_file: coco_dataset/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: coco_dataset/images/val
    ann_file: coco_dataset/annotations/instances_val.json
```

**使用方法**: カスタムデータセットを使用する際は、このファイルを編集してパスとクラス数を変更します。

---

### `configs/dataset/voc_detection.yml`

**役割**: PASCAL VOCデータセットの設定

**主な設定項目**:

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `num_classes` | クラス数 | `20` |
| `dataset.type` | データセット型 | `VOCDetection` |
| `dataset.root` | VOCデータセットのルートディレクトリ | `./dataset/voc/` |
| `dataset.ann_file` | アノテーションファイル | `trainval.txt` / `test.txt` |
| `dataset.label_file` | ラベルファイル | `label_list.txt` |

---

### `configs/dataset/obj365_detection.yml`

**役割**: Objects365データセットの設定

**主な設定項目**:

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `num_classes` | クラス数 | `366` |
| `remap_mscoco_category` | カテゴリ再マッピング | `False` |

---

### `configs/dataset/crowdhuman_detection.yml`

**役割**: CrowdHumanデータセットの設定

**主な設定項目**:

| パラメータ | 説明 | 設定値 |
|-----------|------|--------|
| `num_classes` | クラス数 | `2`（person, ignore） |
| `remap_mscoco_category` | カテゴリ再マッピング | `False` |

---

## 🚀 4. `configs/rtv4/` - RT-DETRv4モデル設定ファイル群

### `configs/rtv4/rtv4_hgnetv2_s_coco.yml`

**役割**: RT-DETRv4 Smallモデルの完全な設定

**構造**: 以下のファイルをインクルード
```yaml
__include__: [
  '../dfine/dfine_hgnetv2_s_coco.yml',
  '../base/rtv4.yml'
]
```

**追加設定**:

#### 教師モデル（DINOv3）

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth
  patch_size: 16
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

#### エンコーダー設定

```yaml
HybridEncoder:
  distill_teacher_dim: 768  # DINOv3の出力次元に合わせる
```

#### 損失関数の知識蒸留

```yaml
RTv4Criterion:
  weight_dict:
    loss_distill: 5
  distill_adaptive_params:
    enabled: True
    rho: 11
    delta: 1
    default_weight: 20
```

#### オプティマイザー

```yaml
optimizer:
  type: AdamW
  params:
    - params: '^(?=.*backbone)(?!.*bn).*$'
      lr: 0.0002
    - params: '^(?=.*(?:norm|bn)).*$'
      weight_decay: 0.
  lr: 0.0004
  betas: [0.9, 0.999]
  weight_decay: 0.0001
```

#### 学習スケジュール

```yaml
epoches: 132
flat_epoch: 64
no_aug_epoch: 12
```

#### データ拡張ポリシー

```yaml
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: [4, 64, 120]
  collate_fn:
    mixup_epochs: [4, 64]
    stop_epoch: 120
```

---

### `configs/rtv4/rtv4_hgnetv2_s_coco_customed.yml`

**役割**: カスタムデータセット用のRT-DETRv4 Small設定

**違い**:
- `'../dataset/custom_detection.yml'`をインクルード（COCO検出の代わり）
- その他の設定は`rtv4_hgnetv2_s_coco.yml`と同じ

---

### `configs/rtv4/rtv4_hgnetv2_m_coco.yml`

**役割**: RT-DETRv4 Mediumモデルの設定

**モデルサイズの違い**:
- バックボーン: `HGNetv2.name: 'B1'`（Sより大きい）
- `hidden_dim`: より大きい次元数
- 学習率やバッチサイズの調整

---

### `configs/rtv4/rtv4_hgnetv2_l_coco.yml`

**役割**: RT-DETRv4 Largeモデルの設定

**モデルサイズの違い**:
- バックボーン: `HGNetv2.name: 'B2'`
- より大きな`hidden_dim`と`dim_feedforward`

---

### `configs/rtv4/rtv4_hgnetv2_x_coco.yml`

**役割**: RT-DETRv4 Xtra-Largeモデルの設定

**モデルサイズの違い**:
- バックボーン: `HGNetv2.name: 'B3'`（最大）
- 最大の`hidden_dim`と`dim_feedforward`

---

## 🔬 5. `configs/dfine/` - D-FINEモデル設定ファイル群

### `configs/dfine/dfine_hgnetv2_s_coco.yml`

**役割**: D-FINE Smallモデルの基本設定（知識蒸留なし）

**構造**: 以下のファイルをインクルード
```yaml
__include__: [
  '../dataset/custom_detection.yml',  # または coco_detection.yml
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/optimizer.yml',
  '../base/dfine_hgnetv2.yml',
]
```

**主な設定**:

#### バックボーン

```yaml
HGNetv2:
  name: 'B0'
  return_idx: [1, 2, 3]  # 3つの特徴マップを返す
  freeze_at: -1
  freeze_norm: False
  use_lab: True
```

#### エンコーダー

```yaml
HybridEncoder:
  in_channels: [256, 512, 1024]
  hidden_dim: 256
  depth_mult: 0.34
  expansion: 0.5
```

#### デコーダー

```yaml
DFINETransformer:
  num_layers: 3  # Smallモデルは3層
  eval_idx: -1
```

#### 学習スケジュール

```yaml
epoches: 132  # 120 + 4n
train_dataloader:
  dataset:
    transforms:
      policy:
        epoch: 120
  collate_fn:
    stop_epoch: 120
    ema_restart_decay: 0.9999
    base_size_repeat: 20
```

---

### `configs/dfine/dfine_hgnetv2_n_coco.yml`

**役割**: D-FINE Nanoモデル（最小サイズ）

**主な違い**:
- `return_idx: [2, 3]`（2つの特徴マップのみ）
- `in_channels: [512, 1024]`
- `hidden_dim: 128`（より小さい）
- `num_levels: 2`
- `epoches: 160`

---

### `configs/dfine/dfine_hgnetv2_m_coco.yml`

**役割**: D-FINE Mediumモデル

**主な違い**:
- `HGNetv2.name: 'B1'`
- より大きな`hidden_dim`
- `num_layers: 4`

---

### `configs/dfine/dfine_hgnetv2_l_coco.yml`

**役割**: D-FINE Largeモデル

**主な違い**:
- `HGNetv2.name: 'B2'`
- `num_layers: 5`

---

### `configs/dfine/dfine_hgnetv2_x_coco.yml`

**役割**: D-FINE Xtra-Largeモデル

**主な違い**:
- `HGNetv2.name: 'B3'`
- `num_layers: 6`

---

## 🔄 6. `configs/rtv2/` - RT-DETRv2モデル設定ファイル群

### `configs/rtv2/rtv2_r50vd_6x_coco.yml`

**役割**: RT-DETRv2 + ResNet50-D バックボーンの設定

**構造**: 以下のファイルをインクルード
```yaml
__include__: [
  '../dataset/coco_detection.yml',
  '../runtime.yml',
  '../base/dataloader.yml',
  '../base/rtv2_optimizer.yml',
  '../base/rtv2_r50vd.yml',
]
```

**主な設定**:
- バックボーン: `PResNet`（ResNet50-D）
- デコーダー: `RTDETRTransformerv2`
- 学習期間: `6x`スケジュール（72エポック）

---

### `configs/rtv2/rtv2_r18vd_120e_coco.yml`

**役割**: RT-DETRv2 + ResNet18-D（軽量版）

**主な違い**:
- `PResNet.depth: 18`
- より小さな`in_channels`
- 学習期間: 120エポック

---

### `configs/rtv2/rtv2_r34vd_120e_coco.yml`

**役割**: RT-DETRv2 + ResNet34-D

**主な違い**:
- `PResNet.depth: 34`

---

### `configs/rtv2/rtv2_r101vd_6x_coco.yml`

**役割**: RT-DETRv2 + ResNet101-D（大型版）

**主な違い**:
- `PResNet.depth: 101`
- より大きな`in_channels`

---

### `configs/rtv2/rtv2_r50vd_m_7x_coco.yml`

**役割**: RT-DETRv2 + ResNet50-D（改良版、7xスケジュール）

**主な違い**:
- 学習期間: 84エポック（7x）
- 調整された学習率スケジュール

---

## 🧪 7. `configs/deim/` - DEIMモデル設定ファイル群

### `configs/deim/deim_hgnetv2_s_coco.yml`

**役割**: DEIM + HGNetv2 Smallモデルの設定

**構造**: 以下のファイルをインクルード
```yaml
__include__: [
  '../dfine/dfine_hgnetv2_s_coco.yml',
  '../base/deim.yml'
]
```

**主な特徴**:
- D-FINEの基本構造にDEIMのマッチング改善を適用
- 知識蒸留なし（RT-DETRv4との違い）

---

### その他のDEIMモデル

- `deim_hgnetv2_n_coco.yml`: Nanoモデル
- `deim_hgnetv2_m_coco.yml`: Mediumモデル
- `deim_hgnetv2_l_coco.yml`: Largeモデル
- `deim_hgnetv2_x_coco.yml`: Xtra-Largeモデル
- `deim_rtv2_r18vd_120e_coco.yml`: ResNet18ベース
- `deim_rtv2_r34vd_120e_coco.yml`: ResNet34ベース
- `deim_rtv2_r50vd_60e_coco.yml`: ResNet50ベース
- `deim_rtv2_r101vd_60e_coco.yml`: ResNet101ベース

---

## 📝 8. 設定ファイルの階層構造と継承

### インクルード（`__include__`）の仕組み

RT-DETRv4の設定ファイルは階層的に構造化されており、複数のファイルを組み合わせて最終的な設定を作成します。

**例**: `rtv4_hgnetv2_s_coco.yml`の継承構造

```
rtv4_hgnetv2_s_coco.yml
├── dfine_hgnetv2_s_coco.yml
│   ├── custom_detection.yml          # データセット設定
│   ├── runtime.yml                   # ランタイム設定
│   ├── dataloader.yml                # データローダー設定
│   ├── optimizer.yml                 # オプティマイザー設定
│   └── dfine_hgnetv2.yml            # モデルアーキテクチャ
└── rtv4.yml                          # RT-DETRv4の拡張設定
```

**継承のルール**:
1. 後にインクルードされたファイルの設定が優先される
2. 同じキーが複数のファイルに存在する場合、後の値で上書きされる
3. 各ファイルで追加の設定項目を定義できる

---

## 🎯 9. 用途別の設定ファイル選択ガイド

### 9.1 モデルアーキテクチャで選ぶ

| モデル | 設定ファイル | 用途 |
|--------|------------|------|
| RT-DETRv4-S | `rtv4/rtv4_hgnetv2_s_coco.yml` | 高精度かつリアルタイム推論 |
| RT-DETRv4-M | `rtv4/rtv4_hgnetv2_m_coco.yml` | バランス型（精度重視） |
| RT-DETRv4-L | `rtv4/rtv4_hgnetv2_l_coco.yml` | 高精度（推論速度やや遅い） |
| RT-DETRv4-X | `rtv4/rtv4_hgnetv2_x_coco.yml` | 最高精度（推論速度遅い） |
| D-FINE-S | `dfine/dfine_hgnetv2_s_coco.yml` | 知識蒸留なしの軽量モデル |
| DEIM-S | `deim/deim_hgnetv2_s_coco.yml` | 改良マッチング付き |
| RT-DETRv2 (ResNet50) | `rtv2/rtv2_r50vd_6x_coco.yml` | ResNetバックボーン使用 |

### 9.2 データセットで選ぶ

| データセット | 設定ファイル（インクルード用） |
|-------------|------------------------|
| COCO2017 | `dataset/coco_detection.yml` |
| カスタム（COCO形式） | `dataset/custom_detection.yml` |
| PASCAL VOC | `dataset/voc_detection.yml` |
| Objects365 | `dataset/obj365_detection.yml` |
| CrowdHuman | `dataset/crowdhuman_detection.yml` |

### 9.3 学習戦略で選ぶ

| 学習戦略 | 追加インクルードファイル |
|---------|---------------------|
| 知識蒸留あり | `base/rtv4.yml` |
| 知識蒸留なし | `base/deim.yml` |
| Flat-Cosineスケジューラー | `base/rtv4.yml`または`base/deim.yml` |
| MultiStepスケジューラー | `base/optimizer.yml` |

---

## 🛠️ 10. カスタマイズ方法

### 10.1 新しいデータセット用の設定作成

1. `configs/dataset/custom_detection.yml`をコピー
2. 以下を編集:
   - `num_classes`: クラス数
   - `img_folder`: 画像フォルダのパス
   - `ann_file`: アノテーションファイルのパス

```yaml
num_classes: 10  # あなたのクラス数
remap_mscoco_category: False

train_dataloader:
  dataset:
    img_folder: path/to/your/train/images
    ann_file: path/to/your/train/annotations.json
```

### 10.2 学習ハイパーパラメータの調整

モデル設定ファイル（例: `rtv4_hgnetv2_s_coco.yml`）で上書き:

```yaml
epoches: 200  # エポック数を増やす

optimizer:
  lr: 0.0008  # 学習率を2倍に

train_dataloader:
  total_batch_size: 64  # バッチサイズを増やす
```

### 10.3 コマンドラインでの動的上書き

設定ファイルを直接編集せずに、コマンドラインで設定を上書き可能:

```powershell
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml `
  -u epoches=200 optimizer.lr=0.0008 train_dataloader.total_batch_size=64
```

---

## 📖 11. まとめ

### 設定ファイルの分類

| フォルダ | 役割 | 主な用途 |
|---------|------|---------|
| `configs/` ルート | 実行時の基本設定 | `runtime.yml` |
| `configs/base/` | 共通の基本設定 | モデル、オプティマイザー、データローダー |
| `configs/dataset/` | データセット固有の設定 | データパス、クラス数、評価器 |
| `configs/rtv4/` | RT-DETRv4モデル設定 | 知識蒸留を含む完全な設定 |
| `configs/dfine/` | D-FINEモデル設定 | 知識蒸留なしの基本モデル |
| `configs/deim/` | DEIMモデル設定 | 改良マッチング付きモデル |
| `configs/rtv2/` | RT-DETRv2モデル設定 | ResNetバックボーン使用 |

### 学習開始時のファイル選択フロー

1. **モデルを選ぶ**: `rtv4/`, `dfine/`, `deim/`, `rtv2/`から選択
2. **モデルサイズを選ぶ**: `n`, `s`, `m`, `l`, `x`から選択
3. **データセットを確認**: インクルードされている`dataset/*.yml`を確認
4. **必要に応じてカスタマイズ**: データパス、ハイパーパラメータを調整

### 推奨設定

- **初心者**: `rtv4/rtv4_hgnetv2_s_coco_customed.yml`（カスタムデータセット用）
- **高精度重視**: `rtv4/rtv4_hgnetv2_x_coco.yml`
- **軽量・高速**: `dfine/dfine_hgnetv2_n_coco.yml`
- **ResNetバックボーン**: `rtv2/rtv2_r50vd_6x_coco.yml`

---

## 🔗 参考情報

- 各設定ファイルの詳細な説明は、ファイル内のコメントも参照してください
- 学習手順については`README_for_train.md`を参照してください
- モデルアーキテクチャの詳細は論文を参照してください

---

**作成日**: 2025年12月4日  
**対象バージョン**: RT-DETRv4
