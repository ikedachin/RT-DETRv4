# RT-DETRv4 推論手順書 (Inference Guide)

このドキュメントでは、RT-DETRv4モデルを使用した推論（物体検出）の手順を、プログラミング初心者の方にもわかりやすく説明します。

---

## 🚀 クイックスタート（5分で試す）

すぐに推論を試したい方は、以下の3ステップで実行できます：

### ステップ1: 環境を用意する
```powershell
# Python環境を作成して有効化
conda create -n rtv4 python=3.11.9
conda activate rtv4

# 必要なパッケージをインストール
pip install -r requirements.txt
```

### ステップ2: モデルをダウンロードする
- [RT-DETRv4-S モデル](https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z)をダウンロード
- ダウンロードしたファイル名を `RTv4-S-hgnet.pth` にリネーム
- プロジェクトフォルダ内の `weights` フォルダに配置

### ステップ3: 画像で推論を実行する
```powershell
# プロジェクトフォルダに移動（例）
cd C:\Users\あなたのユーザー名\Desktop\workspace\RT-DETRv4

# 推論を実行（your_image.jpgを検出したい画像のパスに置き換える）
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i your_image.jpg -d cpu
```

**実行後、`torch_results.jpg` という検出結果の画像が生成されます！**

> **💡 ヒント**: GPU（NVIDIA製）をお持ちの場合は、`-d cpu` を `-d cuda:0` に変更すると高速になります。

---

## 📋 目次

1. [環境構築](#1-環境構築)
2. [モデルの準備](#2-モデルの準備)
3. [PyTorchモデルでの推論](#3-pytorchモデルでの推論)
4. [ONNXモデルでの推論](#4-onnxモデルでの推論)
5. [TensorRTでの推論](#5-tensorrtでの推論)
6. [ベンチマーク](#6-ベンチマーク)
7. [可視化ツール](#7-可視化ツール)

---

## 1. 環境構築

### 1.1 基本環境のセットアップ

**Pythonとは？** プログラミング言語の1つで、機械学習やデータ分析でよく使われます。  
**Condaとは？** Pythonの環境を簡単に管理できるツールです。

学習環境が既にある場合はスキップできます：

```powershell
# 新しいPython環境を「rtv4」という名前で作成
conda create -n rtv4 python=3.11.9

# 作成した環境を有効化（使えるようにする）
conda activate rtv4

# 必要なライブラリ（パッケージ）を一括インストール
pip install -r requirements.txt
```

> **📝 補足**: `requirements.txt` には必要なライブラリのリストが書かれています。PyTorch（深層学習フレームワーク）など、推論に必要なものがインストールされます。

### 1.2 推論用の追加パッケージ（動画を扱う場合）

動画ファイルで推論を行いたい場合は、以下も実行してください：

```powershell
pip install opencv-python Pillow
```

**推論に必要なパッケージ:**
- `opencv-python`: 動画の読み込み・書き込み用
- `Pillow`: 画像の読み込み・編集用

### 1.3 GPUを使う場合の確認

NVIDIA製のGPU（グラフィックボード）がある場合、推論を高速化できます：

```powershell
# GPUが認識されているか確認
python -c "import torch; print(torch.cuda.is_available())"
```

- `True` と表示されれば、GPU使用可能です
- `False` の場合は、CPUで推論します（少し遅くなりますが問題ありません）

---

## 2. モデルの準備

### 2.1 事前学習済みモデルのダウンロード

**モデルとは？** 画像から物体を検出するための「学習済みの頭脳」です。既に大量の画像で学習済みなので、すぐに使えます。

公式の事前学習済みモデル（COCOデータセットで学習済み）を使用する場合：

| モデル | 精度 (AP) | 速度 (T4 GPU) | 用途 | ダウンロード |
|--------|----------|--------------|------|------------|
| RT-DETRv4-S | 49.8 | 3.66 ms | 軽量・高速 | [Google Drive](https://drive.google.com/file/d/1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z) |
| RT-DETRv4-M | 53.7 | 5.91 ms | バランス型 | [Google Drive](https://drive.google.com/file/d/1O-YpP4X-quuOXbi96y2TKkztbjroP5mX) |
| RT-DETRv4-L | 55.4 | 8.07 ms | 高精度 | [Google Drive](https://drive.google.com/file/d/1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8) |
| RT-DETRv4-X | 57.0 | 12.90 ms | 最高精度 | [Google Drive](https://drive.google.com/file/d/19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3) |

> **💡 初心者の方へ**: まずは **RT-DETRv4-S** から試すことをお勧めします。軽量で動作も速く、CPUでも比較的快適に動きます。

### 2.2 ダウンロードしたモデルの配置方法

1. 上記のリンクからモデルファイル（`.pth`）をダウンロード
2. ダウンロードしたファイルを `weights` フォルダに配置
3. ファイル名を以下のようにリネーム（推奨）：

**推奨ディレクトリ構造:**
```
RT-DETRv4/
└── weights/
    ├── RTv4-S-hgnet.pth    ← Sモデル
    ├── RTv4-M-hgnet.pth    ← Mモデル  
    ├── RTv4-L-hgnet.pth    ← Lモデル
    └── RTv4-X-hgnet.pth    ← Xモデル
```

> **📝 注意**: `weights` フォルダが存在しない場合は、プロジェクトフォルダ内に新規作成してください。

### 2.3 自分で学習したモデルを使用する場合

カスタムデータセットで学習を行った場合は、学習完了後のチェックポイントファイル（`.pth`）を使用します：

```
outputs/rtv4_hgnetv2_s_coco/checkpoint0029.pth
```

> **💡 ヒント**: 学習時に `outputs` フォルダ内に保存されたチェックポイントをそのまま使えます。

---

## 3. PyTorchモデルでの推論

**推論とは？** 学習済みモデルを使って、新しい画像から物体を検出することです。

### 3.1 画像に対する推論（基本）

まずは1枚の画像で試してみましょう。

#### 🔹 コマンドの構造を理解する

```powershell
python .\tools\inference\torch_inf.py -c [設定ファイル] -r [モデルファイル] -i [入力画像] -d [デバイス]
```

#### 🔹 実際の実行例（GPU使用）

```powershell
# プロジェクトのルートフォルダにいることを確認
cd C:\Users\あなたのユーザー名\Desktop\workspace\RT-DETRv4

# 推論を実行（room.pngという画像を検出）
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\room.png -d cuda:0
```

#### 🔹 実際の実行例（CPU使用 - GPUがない場合）

```powershell
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\room.png -d cpu
```

**各オプションの意味:**
- `-c`: 設定ファイル（モデルの構造やパラメータが書かれたファイル）
- `-r`: モデルの重みファイル（学習済みのパラメータが保存されたファイル）
- `-i`: 入力画像のパス（検出したい画像）
- `-d`: 使用デバイス
  - `cuda:0` = 1番目のGPUを使用（高速）
  - `cpu` = CPUを使用（GPUがない場合）

**実行後の出力:**
- `torch_results.jpg`: 検出結果が描画された画像ファイルが、プロジェクトフォルダに生成されます

> **📝 重要**: 入力画像のパスは、**プロジェクトフォルダからの相対パス**または**絶対パス**で指定します。
> 
> 例:
> - 相対パス: `.\images\test.jpg` （プロジェクトフォルダ内のimagesフォルダ内のtest.jpg）
> - 絶対パス: `C:\Users\YourName\Pictures\test.jpg`

### 3.2 動画に対する推論

動画ファイル（`.mp4`, `.avi` など）にも対応しています。

```powershell
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\your_video.mp4 -d cuda:0
```

**実行後の出力:**
- `torch_results.mp4`: 検出結果が描画された動画ファイルが生成されます
- コンソールに処理進捗が表示されます: `Processed 10 frames...`

> **⚠️ 注意**: 動画処理は画像より時間がかかります。最初は短い動画（10秒程度）で試すことをお勧めします。

### 3.3 異なるモデルサイズを使う

より高精度な検出が必要な場合は、大きいモデルを使用します：

#### RT-DETRv4-M（中サイズ）を使う例
```powershell
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_m_coco.yml -r .\weights\RTv4-M-hgnet.pth -i .\your_image.jpg -d cuda:0
```

#### RT-DETRv4-L（大サイズ）を使う例
```powershell
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_l_coco.yml -r .\weights\RTv4-L-hgnet.pth -i .\your_image.jpg -d cuda:0
```

> **💡 モデル選択のヒント**:
> - **S (Small)**: 速度重視、CPUでも動く、精度は標準
> - **M (Medium)**: バランス型、GPUがあればお勧め
> - **L (Large) / X (Extra Large)**: 精度重視、GPU必須、処理に時間がかかる

---

## 4. ONNXモデルでの推論

ONNXモデルはプラットフォーム非依存で高速な推論が可能です。

### 4.1 ONNXモデルのエクスポート

```powershell
python tools/deployment/export_onnx.py `
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml `
  -r weights/rtv4_hgnetv2_s.pth `
  --check `
  --simplify
```

**オプション:**
- `--check`: エクスポートしたモデルの検証
- `--simplify`: ONNXモデルの最適化（推奨）

**出力:**
- `weights/rtv4_hgnetv2_s.onnx`: エクスポートされたONNXモデル

### 4.2 ONNX推論の実行

#### 必要なパッケージのインストール

```powershell
pip install onnx onnxruntime
```

GPUを使用する場合:

```powershell
pip install onnxruntime-gpu
```

#### 画像に対する推論

```powershell
python tools/inference/onnx_inf.py `
  --onnx weights/rtv4_hgnetv2_s.onnx `
  --input path/to/image.jpg
```

**出力:**
- `onnx_result.jpg`: 検出結果が描画された画像

#### 動画に対する推論

```powershell
python tools/inference/onnx_inf.py `
  --onnx weights/rtv4_hgnetv2_s.onnx `
  --input path/to/video.mp4
```

**出力:**
- `onnx_result.mp4`: 検出結果が描画された動画

---

## 5. TensorRTでの推論

TensorRTは最高速度の推論エンジンです（NVIDIA GPUのみ）。

### 5.1 TensorRTのインストール

TensorRTのインストールは環境によって異なります。詳細は[公式ドキュメント](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)を参照してください。

### 5.2 TensorRTエンジンのビルド

まずONNXモデルをエクスポート（上記参照）してから、TensorRTエンジンをビルドします：

```powershell
trtexec --onnx="weights/rtv4_hgnetv2_s.onnx" `
        --saveEngine="weights/rtv4_hgnetv2_s.engine" `
        --fp16
```

**オプション:**
- `--fp16`: FP16精度を使用（高速化、推奨）
- `--int8`: INT8精度を使用（さらに高速化、要キャリブレーション）

**出力:**
- `weights/rtv4_hgnetv2_s.engine`: TensorRTエンジンファイル

### 5.3 TensorRT推論の実行

```powershell
python tools/inference/trt_inf.py `
  --trt weights/rtv4_hgnetv2_s.engine `
  --input path/to/image.jpg
```

**動画の場合:**

```powershell
python tools/inference/trt_inf.py `
  --trt weights/rtv4_hgnetv2_s.engine `
  --input path/to/video.mp4
```

---

## 6. ベンチマーク

### 6.1 モデル情報の確認

FLOPs、MACs、パラメータ数を確認：

```powershell
python tools/benchmark/get_info.py -c configs/rtv4/rtv4_hgnetv2_s_coco.yml
```

**出力例:**
```
Model FLOPs: 26.8 G   MACs: 13.4 G   Params: 9876543
```

### 6.2 TensorRTレイテンシのベンチマーク

TensorRTエンジンの実測レイテンシを計測：

```powershell
pip install -r tools/benchmark/requirements.txt
```

```powershell
python tools/benchmark/trt_benchmark.py `
  --COCO_dir path/to/COCO2017 `
  --engine_dir weights/rtv4_hgnetv2_s.engine
```

このツールはCOCO2017検証セットを使用してレイテンシを計測します。

---

## 7. 可視化ツール

### 7.1 Fiftyone可視化

[Fiftyone](https://github.com/voxel51/fiftyone)を使用した高度な可視化：

#### インストール

```powershell
pip install fiftyone
```

#### 実行

```powershell
python tools/visualization/fiftyone_vis.py `
  -c configs/rtv4/rtv4_hgnetv2_s_coco.yml `
  -r weights/rtv4_hgnetv2_s.pth
```

このツールを実行すると、ブラウザが開いてインタラクティブな可視化インターフェースが表示されます。

### 7.2 検出結果のカスタマイズ

推論スクリプト内の閾値を調整できます：

`tools/inference/torch_inf.py` の `draw()` 関数内：

```python
def draw(images, labels, boxes, scores, thrh=0.4):  # thrh: 信頼度閾値
```

閾値を上げると（例: `0.6`）、より確信度の高い検出のみ表示されます。

---

## 8. バッチ推論

### 8.1 複数画像の推論

複数の画像に対してバッチ推論を行う場合、スクリプトを修正してループ処理を追加します：

```python
import glob

image_paths = glob.glob("path/to/images/*.jpg")
for img_path in image_paths:
    # 推論処理
    pass
```

### 8.2 ディレクトリ処理の例

```powershell
# PowerShellでの例
Get-ChildItem -Path "path/to/images" -Filter *.jpg | ForEach-Object {
    python tools/inference/torch_inf.py `
      -c configs/rtv4/rtv4_hgnetv2_s_coco.yml `
      -r weights/rtv4_hgnetv2_s.pth `
      -i $_.FullName `
      -d cuda:0
}
```

---

## 9. 推論パフォーマンス比較

| 推論方法 | レイテンシ | セットアップ難易度 | 互換性 |
|---------|-----------|------------------|--------|
| PyTorch | 普通 | 簡単 | 高 |
| ONNX Runtime | やや速い | 簡単 | 高 |
| TensorRT | 最速 | やや難 | NVIDIA GPUのみ |

**推奨:**
- **開発・デバッグ**: PyTorch
- **本番環境**: TensorRT（NVIDIA GPU）またはONNX Runtime（その他）

---

## 10. トラブルシューティング

### 10.1 よくあるエラーと対処法

#### ❌ エラー: `FileNotFoundError: [Errno 2] No such file or directory`

**原因:** ファイルパスが間違っているか、ファイルが存在しません。

**対処法:**
1. ファイルパスを確認（スペルミスがないか）
2. ファイルが実際に存在するか確認
3. 絶対パスで指定してみる
   ```powershell
   # 例: 絶対パスで指定
   python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i C:\Users\YourName\Pictures\test.jpg -d cpu
   ```

#### ❌ エラー: `RuntimeError: No such file or directory: 'weights/rtv4_hgnetv2_s.pth'`

**原因:** モデルファイルが見つかりません。

**対処法:**
1. `weights` フォルダが存在するか確認
2. モデルファイルをダウンロードして `weights` フォルダに配置
3. ファイル名が正確に一致しているか確認（`-r` オプションで指定した名前と実際のファイル名）

#### ❌ エラー: `CUDA out of memory`（推論時）

**原因:** GPUのメモリ不足です。

**対処法（優先順）:**
1. **CPUで推論する**: `-d cuda:0` を `-d cpu` に変更
   ```powershell
   python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\your_image.jpg -d cpu
   ```
2. より小さいモデル（S）を使用
3. 画像サイズを小さくする（Photoshopなどで事前にリサイズ）

#### ❌ エラー: `ModuleNotFoundError: No module named 'cv2'`

**原因:** OpenCVがインストールされていません。

**対処法:**
```powershell
pip install opencv-python
```

#### ❌ エラー: `ImportError: cannot import name 'YAMLConfig'`

**原因:** プロジェクトフォルダ以外の場所からスクリプトを実行しています。

**対処法:**
```powershell
# プロジェクトのルートフォルダに移動
cd C:\Users\あなたのユーザー名\Desktop\workspace\RT-DETRv4

# 再度実行
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\your_image.jpg -d cpu
```

### 10.2 推論結果が表示されない・物体が検出されない

#### 🔹 検出結果のファイルが生成されない

**対処法:**
1. エラーメッセージを確認（赤文字で表示されます）
2. 入力画像が正しく読み込めているか確認
3. プロジェクトフォルダに書き込み権限があるか確認

#### 🔹 画像に何も検出されない（四角が描画されない）

**原因:** 検出信頼度が閾値（しきい値）以下です。

**対処法:**
1. **閾値を下げる**: `tools/inference/torch_inf.py` を編集
   ```python
   # 17行目付近を探す
   def draw(images, labels, boxes, scores, thrh=0.4):  # 0.4 → 0.2 に変更
   ```
2. より大きいモデル（M, L, X）を試す
3. 画像に検出対象物体（人、車、動物など）が写っているか確認

> **💡 補足**: COCOデータセットで学習されたモデルは、80種類の物体（人、車、犬、猫、椅子など）を検出できます。それ以外の物体は検出できません。

### 10.3 推論速度が遅い場合

**対処法（効果が高い順）:**

1. **GPUを使用する**: `-d cpu` を `-d cuda:0` に変更
   ```powershell
   python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i .\your_image.jpg -d cuda:0
   ```

2. **小さいモデルを使う**: L/X → M → S の順に軽量化
   
3. **ONNX Runtime を使用する**: セクション4を参照（1.5～2倍高速化）

4. **TensorRT を使用する**: セクション5を参照（最大5倍高速化、NVIDIA GPU必須）

### 10.4 検出精度を上げたい場合

**対処法:**

1. **より大きいモデルを使う**: S → M → L → X の順に高精度化
   
2. **閾値を適切に調整**: 高すぎると検出漏れ、低すぎると誤検出が増える

3. **カスタムデータセットで再学習**: 独自の物体を検出したい場合は必須（`README_for_train.md` を参照）

### 10.5 コマンドが長くて打ちにくい

**対処法: バッチファイルを作成する**

`inference.bat` という名前でテキストファイルを作成し、以下を記述：

```batch
@echo off
python .\tools\inference\torch_inf.py -c .\configs\rtv4\rtv4_hgnetv2_s_coco.yml -r .\weights\RTv4-S-hgnet.pth -i %1 -d cpu
pause
```

使い方:
```powershell
# バッチファイルをダブルクリック、または
.\inference.bat your_image.jpg
```

これで毎回長いコマンドを打つ必要がなくなります！

---

## 11. APIとしての使用（プログラマー向け）

### 11.1 Pythonスクリプトでの使用例

推論機能を自分のPythonプログラムに組み込むことができます。

```python
import torch
from engine.core import YAMLConfig
from PIL import Image
import torchvision.transforms as T

# === 1. モデルの読み込み ===
cfg = YAMLConfig('configs/rtv4/rtv4_hgnetv2_s_coco.yml')
checkpoint = torch.load('weights/RTv4-S-hgnet.pth', map_location='cpu')

# チェックポイント形式に応じて読み込み
if 'ema' in checkpoint:
    state = checkpoint['ema']['module']
else:
    state = checkpoint['model']

cfg.model.load_state_dict(state)
model = cfg.model.deploy().cuda()  # GPUを使う場合
# model = cfg.model.deploy()  # CPUを使う場合（.cuda()を削除）
postprocessor = cfg.postprocessor.deploy()

# === 2. 画像の読み込みと前処理 ===
image = Image.open('your_image.jpg').convert('RGB')
w, h = image.size  # 元の画像サイズを保存

# 画像を640x640にリサイズしてテンソルに変換
transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
])
image_tensor = transforms(image).unsqueeze(0).cuda()  # バッチ次元を追加
orig_size = torch.tensor([[w, h]]).cuda()  # 元のサイズ情報

# === 3. 推論実行 ===
model.eval()  # 評価モードに設定
with torch.no_grad():  # 勾配計算を無効化（推論時は不要）
    outputs = model(image_tensor)
    labels, boxes, scores = postprocessor(outputs, orig_size)

# === 4. 結果の処理 ===
threshold = 0.4  # 信頼度の閾値
for label, box, score in zip(labels[0], boxes[0], scores[0]):
    if score > threshold:
        class_id = label.item()
        confidence = score.item()
        x1, y1, x2, y2 = box.tolist()
        
        print(f"検出: クラスID={class_id}, 信頼度={confidence:.2f}, "
              f"位置=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
```

**出力例:**
```
検出: クラスID=0, 信頼度=0.95, 位置=(120.5, 80.3, 350.2, 450.8)
検出: クラスID=2, 信頼度=0.87, 位置=(400.1, 200.5, 550.3, 380.7)
```

### 11.2 検出結果の意味

- **クラスID**: COCOデータセットのクラス番号
  - 0 = 人 (person)
  - 1 = 自転車 (bicycle)
  - 2 = 車 (car)
  - など（[全80クラスのリスト](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)）

- **信頼度 (confidence)**: 0.0～1.0の値。高いほど確信度が高い

- **位置 (box)**: 境界ボックスの座標 `(x1, y1, x2, y2)`
  - `(x1, y1)` = 左上の座標
  - `(x2, y2)` = 右下の座標

### 11.3 複数画像のバッチ処理例

```python
import glob
from pathlib import Path

# 画像フォルダ内の全JPGファイルを取得
image_folder = Path("./images")
image_paths = list(image_folder.glob("*.jpg"))

print(f"{len(image_paths)}枚の画像を処理します...")

for i, img_path in enumerate(image_paths):
    print(f"[{i+1}/{len(image_paths)}] {img_path.name} を処理中...")
    
    # 画像を読み込んで推論
    image = Image.open(img_path).convert('RGB')
    w, h = image.size
    image_tensor = transforms(image).unsqueeze(0).cuda()
    orig_size = torch.tensor([[w, h]]).cuda()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        labels, boxes, scores = postprocessor(outputs, orig_size)
    
    # 結果を保存する処理など...
    print(f"  → {len(scores[0][scores[0] > 0.4])}個の物体を検出")

print("全ての処理が完了しました！")
```

---

## 12. よくある質問（FAQ）

### Q1. どんな物体が検出できますか？

**A:** COCOデータセットで学習されたモデルは、以下の80種類の物体を検出できます：

**人・動物**: 人、犬、猫、鳥、馬、牛、羊、象、クマ、シマウマ、キリンなど  
**乗り物**: 自転車、車、バイク、飛行機、バス、電車、トラック、船など  
**家具・電化製品**: 椅子、ソファ、テーブル、ベッド、テレビ、ラップトップ、冷蔵庫など  
**食べ物**: りんご、バナナ、ピザ、ケーキ、ホットドッグなど  
**スポーツ用品**: ボール、野球バット、テニスラケットなど  
**日用品**: 本、時計、傘、ハサミ、歯ブラシなど

[COCOデータセット全80クラスの一覧はこちら](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)

> **💡 重要**: これら以外の物体（例: アニメキャラクター、特定の製品など）を検出したい場合は、**カスタムデータセットで再学習**が必要です。

### Q2. 検出精度を上げるにはどうすればいいですか？

**A:** 以下の方法を試してください：

1. **より大きなモデルを使う**: S → M → L → X の順に精度が向上
2. **閾値を調整**: デフォルトは0.4。用途に応じて0.3～0.6の範囲で調整
3. **高解像度の画像を使う**: ぼやけた画像では検出精度が下がります
4. **カスタムデータで再学習**: 特定の物体や環境に特化したモデルを作成

### Q3. 商用利用は可能ですか？

**A:** ライセンスファイル（`LICENSE`）を確認してください。一般的に、学術研究やオープンソースプロジェクトでは自由に使えますが、商用利用には制限がある場合があります。

### Q4. リアルタイムで動作しますか？

**A:** はい。以下の環境で実現可能です：

- **GPU使用時**: RT-DETRv4-Sで約3.66ms/フレーム（≈273 FPS）@T4 GPU
- **CPU使用時**: フレームレートは低下しますが、動画処理は可能

### Q5. Webカメラからのリアルタイム検出はできますか？

**A:** はい、可能です。`tools/inference/torch_inf.py` を改造してWebカメラ入力に対応できます：

```python
import cv2

cap = cv2.VideoCapture(0)  # Webカメラを開く (0は通常1番目のカメラ)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # フレームをPIL画像に変換して推論
    # ... 推論処理 ...
    
    # 結果を表示
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーで終了
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 13. まとめ

### ✅ RT-DETRv4の推論手順（おさらい）

1. **環境構築** - Python環境と必要なライブラリをインストール
2. **モデルの準備** - 学習済みモデルをダウンロードして配置
3. **推論実行** - コマンドで画像/動画から物体を検出
4. **結果確認** - 生成された画像/動画で検出結果を確認

### 📊 推論方法の選択ガイド

| 用途 | 推奨方法 | 特徴 |
|------|----------|------|
| **まず試したい** | PyTorch推論 | セットアップ簡単、CPU/GPU両対応 |
| **本番環境で使う** | ONNX Runtime | 高速、環境依存が少ない |
| **最高速度が必要** | TensorRT | 最速、NVIDIA GPU専用 |
| **開発・デバッグ** | PyTorch推論 | エラー情報が詳しい、柔軟性が高い |

### 🎯 次のステップ

- **独自データで学習したい** → `README_for_train.md` を参照
- **モデルを改造したい** → `engine/` フォルダのコードを確認
- **エクスポート・デプロイ** → `tools/deployment/` を参照

### 🆘 サポート

- **バグ報告・質問**: [GitHub Issues](https://github.com/RT-DETRs/RT-DETRv4/issues)
- **最新情報**: 公式GitHubリポジトリをチェック

---

**🎉 これで推論の準備は完了です！実際に画像や動画で試してみましょう！**
