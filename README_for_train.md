# RT-DETRv4 学習手順書 (Training Guide)

このドキュメントでは、RT-DETRv4モデルの学習手順を詳しく説明します。

---

## 📋 目次

1. [環境構築](#1-環境構築)
2. [データセット準備](#2-データセット準備)
3. [教師モデル（DINOv3）の準備](#3-教師モデルdinov3の準備)
4. [設定ファイルの編集](#4-設定ファイルの編集)
5. [学習の実行](#5-学習の実行)
6. [学習の詳細設定](#6-学習の詳細設定)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 環境構築

### 1.1 Conda環境の作成

```powershell
conda create -n rtv4 python=3.11.9
conda activate rtv4
```

### 1.2 依存パッケージのインストール

```powershell
pip install -r requirements.txt
```

**必要なパッケージ:**
- torch
- torchvision
- faster-coco-eval
- PyYAML
- tensorboard
- scipy
- calflops
- transformers

---

## 2. データセット準備

RT-DETRv4は**COCO形式**のデータセットで学習します。

### 2.1 COCO2017データセットを使用する場合

#### ダウンロード
- [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017)
- [COCO公式サイト](https://cocodataset.org/#download)

#### 設定ファイルの編集

`configs/dataset/coco_detection.yml` を編集してパスを指定します：

```yaml
train_dataloader:
  dataset:
    img_folder: /data/COCO2017/train2017/
    ann_file: /data/COCO2017/annotations/instances_train2017.json

val_dataloader:
  dataset:
    img_folder: /data/COCO2017/val2017/
    ann_file: /data/COCO2017/annotations/instances_val2017.json
```

### 2.2 カスタムデータセットを使用する場合

#### 推奨ディレクトリ構造

```
coco_dataset/
├── annotations/
│   ├── instances_train.json    # 訓練用アノテーション
│   └── instances_val.json      # 検証用アノテーション
└── images/
    ├── train/                  # 訓練画像
    │   ├── image001.jpg
    │   └── ...
    └── val/                    # 検証画像
        ├── image001.jpg
        └── ...
```

#### COCO形式のアノテーション例

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "image001.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 200, 50, 80],
            "area": 4000,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "class_a"
        }
    ]
}
```

**注意:**
- `bbox` は `[左上x, 左上y, 幅, 高さ]` の形式（ピクセル単位）
- `area` はバウンディングボックスの面積（`width × height`）
- カスタムデータセットでは通常 `iscrowd: 0` を使用

#### YOLOフォーマットからの変換

YOLO形式のデータセットがある場合は、付属の変換スクリプトを使用できます：

```powershell
python yolo2coco.py -i ./YOLO_dataset -o ./coco_dataset
```

オプション:
- `-s 640 640`: 画像をリサイズ
- `-k`: アスペクト比を維持してリサイズ（パディング付き）

#### 設定ファイルの編集

`configs/dataset/custom_detection.yml` を編集します：

```yaml
num_classes: 2  # あなたのクラス数に変更
remap_mscoco_category: False  # カスタムデータセットではFalse

train_dataloader:
  dataset:
    img_folder: /path/to/coco_dataset/images/train
    ann_file: /path/to/coco_dataset/annotations/instances_train.json

val_dataloader:
  dataset:
    img_folder: /path/to/coco_dataset/images/val
    ann_file: /path/to/coco_dataset/annotations/instances_val.json
```

---

## 3. 教師モデル（DINOv3）の準備

RT-DETRv4は知識蒸留を使用するため、事前学習済みのDINOv3モデルが必要です。

### 3.1 DINOv3のダウンロード

1. **リポジトリのクローン:**

```powershell
git clone https://github.com/facebookresearch/dinov3.git
```

2. **重みファイルのダウンロード:**

[DINOv3公式ダウンロードページ](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)から **ViT-B/16-LVD-1689M** モデルをダウンロードします。

推奨ディレクトリ構造:

```
RT-DETRv4/
├── dinov3/                    # DINOv3リポジトリ
└── pretrain/
    └── dinov3_vitb16_pretrain_lvd1689m.pth  # 重みファイル
```

### 3.2 設定ファイルでのパス指定

`configs/rtv4/rtv4_hgnetv2_s_coco.yml` （または使用するモデルの設定ファイル）を編集：

```yaml
teacher_model:
  type: "DINOv3TeacherModel"
  dinov3_repo_path: dinov3/              # DINOv3リポジトリへの相対パス
  dinov3_weights_path: pretrain/dinov3_vitb16_pretrain_lvd1689m.pth  # 重みファイルへの相対パス
  patch_size: 16
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

---

## 4. 設定ファイルの編集

### 4.1 モデルサイズの選択

RT-DETRv4では以下のモデルサイズが利用可能です：

| モデル | AP | Latency (T4) | Config |
|--------|-----|--------------|--------|
| RT-DETRv4-S | 49.8 | 3.66 ms | `configs/rtv4/rtv4_hgnetv2_s_coco.yml` |
| RT-DETRv4-M | 53.7 | 5.91 ms | `configs/rtv4/rtv4_hgnetv2_m_coco.yml` |
| RT-DETRv4-L | 55.4 | 8.07 ms | `configs/rtv4/rtv4_hgnetv2_l_coco.yml` |
| RT-DETRv4-X | 57.0 | 12.90 ms | `configs/rtv4/rtv4_hgnetv2_x_coco.yml` |

### 4.2 出力ディレクトリの設定

設定ファイル内で出力ディレクトリを指定できます：

```yaml
output_dir: ./outputs/rtv4_hgnetv2_s_coco
```

---

## 5. 学習の実行

### 5.1 基本的な学習コマンド

#### シングルGPU学習

```powershell
python train.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco_customed.yml --use-amp -t .\weights\RTv4-S-hgnet.pth -d cpu
```

#### マルチGPU学習（4GPU）

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp --seed=0
```

**オプション説明:**
- `-c`: 設定ファイルのパス
- `--use-amp`: 自動混合精度（Automatic Mixed Precision）を有効化（メモリ節約、高速化）
- `--seed=0`: 再現性のためのランダムシード
- `--nproc_per_node=4`: 使用するGPU数

### 5.2 学習の再開（Resume）

チェックポイントから学習を再開する場合：

```powershell
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp -r outputs/rtv4_hgnetv2_s_coco/checkpoint.pth
```

### 5.3 ファインチューニング（Tuning）

事前学習済みモデルからファインチューニングする場合：

```powershell
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --use-amp -t pretrained_model.pth
```

### 5.4 モデルの評価のみ実行

```powershell
python train.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml --test-only -r model.pth
```

---

## 6. 学習の詳細設定

### 6.1 バッチサイズのカスタマイズ

`configs/base/dataloader.yml` を編集：

```yaml
train_dataloader:
  total_batch_size: 32  # 全GPU合計のバッチサイズ
```

**例:** 4GPUで `total_batch_size: 32` の場合、各GPUは8枚ずつ処理します。

#### バッチサイズを変更した場合の調整

バッチサイズを2倍にする場合、以下のパラメータも調整が必要です（モデル設定ファイル内）：

```yaml
optimizer:
  lr: 0.0005  # 学習率を2倍に（線形スケーリング則）
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.000025  # バックボーンの学習率も2倍

ema:
  decay: 0.9998  # 1 - (1 - decay) * 2 で調整
  warmups: 500   # 半分に

lr_warmup_scheduler:
  warmup_duration: 250  # 半分に
```

### 6.2 入力サイズのカスタマイズ

320x320で学習する場合の設定例：

#### `configs/base/dataloader.yml`:

```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [320, 320], }
  collate_fn:
    base_size: 320

val_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [320, 320], }
```

#### `configs/base/rtv4.yml`:

```yaml
eval_spatial_size: [320, 320]
```

### 6.3 学習のハイパーパラメータ

主要な設定ファイル内のハイパーパラメータ：

```yaml
epoches: 132           # 総エポック数
flat_epoch: 64         # フラット学習期間
no_aug_epoch: 12       # データ拡張を停止するエポック

optimizer:
  type: AdamW
  lr: 0.0004           # 基本学習率
  weight_decay: 0.0001 # 重み減衰

# 知識蒸留の設定
RTv4Criterion:
  weight_dict:
    loss_distill: 5    # 蒸留損失の重み
```

---

## 7. トラブルシューティング

### 7.1 よくあるエラーと対処法

#### エラー: CUDA out of memory

**対処法:**
- バッチサイズを小さくする（`configs/base/dataloader.yml`）
- `--use-amp` を使用してメモリ使用量を削減
- より小さいモデル（S または M）を使用

#### エラー: DINOv3が見つからない

**対処法:**
- `dinov3_repo_path` と `dinov3_weights_path` のパスを確認
- 相対パスまたは絶対パスで正しく指定されているか確認

#### エラー: データセットが読み込めない

**対処法:**
- アノテーションファイルがCOCO形式になっているか確認
- 画像パスとアノテーションファイルのパスが正しいか確認
- `remap_mscoco_category: False` をカスタムデータセットで設定

### 7.2 学習の監視

#### TensorBoardでの確認

学習中のログはTensorBoardで確認できます：

```powershell
tensorboard --logdir=outputs/rtv4_hgnetv2_s_coco
```

ブラウザで `http://localhost:6006` にアクセスしてログを確認できます。

### 7.3 学習時間の目安

- **RT-DETRv4-S**: 4x GPU (A100) で約24時間
- **RT-DETRv4-M**: 4x GPU (A100) で約30時間
- **RT-DETRv4-L**: 4x GPU (A100) で約36時間

（データセットサイズ: COCO2017、132エポック）

---

## 8. 参考資料

### 8.1 他のモデルの学習

このリポジトリでは以下のモデルも学習可能です：

- **D-FINE**: `configs/dfine/`
- **DEIM**: `configs/deim/`
- **RT-DETRv2**: `configs/rtv2/`

それぞれの設定ファイルを使用して同様に学習できます。

### 8.2 自動再開スクリプト

学習が中断された場合に自動で再開するスクリプト：

```powershell
bash tools/reference/safe_training.sh
```

### 8.3 モデル情報の確認

FLOPs、MACs、パラメータ数を確認：

```powershell
python tools/benchmark/get_info.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml
```

---

## 9. まとめ

RT-DETRv4の学習手順：

1. ✅ 環境構築（Conda + 依存パッケージ）
2. ✅ データセット準備（COCO形式）
3. ✅ DINOv3教師モデルの準備
4. ✅ 設定ファイルの編集（データパス、クラス数）
5. ✅ 学習の実行（`train.py`）
6. ✅ TensorBoardで学習監視

学習完了後は、[README_for_infer.md](./README_for_infer.md) を参照して推論を実行してください。

---

**問題が発生した場合は、[GitHub Issues](https://github.com/RT-DETRs/RT-DETRv4/issues)で質問してください。**
