# QUVI (QUality VIsion)

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.md)
[![English](https://img.shields.io/badge/lang-English-red)](README.en.md)
[![日本語](https://img.shields.io/badge/lang-日本語-white)](README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/MCU-ESP32--S3-E7352C)
![Docker](https://img.shields.io/badge/Docker-quvi--dev-2496ED?logo=docker&logoColor=white)
[![Web Demo](https://img.shields.io/badge/Web_Demo-Live-brightgreen)](https://seongjun-k.github.io/QUVI/)

**AIビジョンロボットによる3Dプリンター造形物の自動良否判定・仕分けシステム**

> 「見ることこそ品質である (Seeing is Quality)」

▶ **[Webデモを試す](https://seongjun-k.github.io/QUVI/)** — ハードウェアなしで、実機録画をそのまま再生した良品・不良品の検査サイクルを体験できます。

---

## 課題定義

3Dプリンティングでは、造形後に**人が結果物をベッドから取り外し、目視で不良(反り・層間剥離・糸引き・造形不足)を確認して仕分ける**必要があります。この後処理工程は、プリントファームの規模が大きくなるほど、繰り返し労働と検査のばらつき(人によって異なる基準、疲労による見落とし)が蓄積するボトルネックになります。

QUVIはこの課題を、**把持(模倣学習ロボットアーム) → 検査(マシンビジョン) → 仕分け(自動積載)** の全工程を無人化した自動仕分けシステムで解決します。造形が完了した製品をロボットアームが自動で把持し、検査チャンバーのターンテーブル上で4方向から撮影して品質を分析(良否判定)し、結果に応じて合格(PASS)と不良(FAIL)のステーションへ仕分けて積載します。

全工程は有限状態機械(FSM)オーケストレーターが自律制御し、把持はLeRobot ACTによる模倣学習、検査は表面特徴のルール判定とPatchCore異常検知を組み合わせたハイブリッド方式で行います。

---

## アーキテクチャ

### システム構成

| 構成要素 | 技術仕様 | 役割・特徴 |
| :--- | :--- | :--- |
| **メイン制御** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | 全ノードのオーケストレーションおよび状態機械(FSM)制御 |
| **下位制御** | ESP32-S3 + TB6600 (micro-ROS) | リニアレール(ステッピングモーター)・ターンテーブル角度・照明LEDの駆動制御 |
| **ロボットアーム** | ROBOTIS OMX マニピュレーター (フォロワー) | Dynamixel (XL430/XL330) ベース、リーダー・フォロワー方式のテレオペ対応 |
| **カメラ** | USB UVCカメラ × 2 | サイドカメラ(Zone 1: 把持エリア)、検査カメラ(Zone 2: 品質検査チャンバー) |
| **AI・アルゴリズム** | LeRobot ACT + OpenCV + PatchCore | 模倣学習による把持制御、表面特徴のルール判定、ML異常検知(ハイブリッド結合) |

### 検査方式

検査カメラがターンテーブルの4方向(0°/90°/180°/270°)の画像をキャプチャして2系統で分析し、両者の結果を**ハイブリッドに結合**して最終判定します。

1. **表面特徴のルール判定** — 角度別のworst-caseでPASS/FAILを決定
   * Solidity (凸包に対する輪郭面積比 — 反りの検出)
   * 面積比 (基準画像との比較 — 造形不足/過剰の検出)。ターンテーブルの偏心による物体-カメラ間距離の変化を相殺するため、**面積/幅² の距離不変正規化**で比較
   * 穴の個数・穴面積比 (層間剥離の検出) — 穴1個からFAIL
   * テクスチャ分散 (ラプラシアン — 糸引きの検出)

   判定しきい値は `src/quvi_inspect/config/inspect_params.yaml` で管理し、HMI表示基準(`dashboard.js` のTHRESHOLDS)との同期を維持します。
2. **PatchCore異常検知** — WideResNet50バックボーンによる角度別メモリバンクで異常スコアを計算。正常品の画像のみで学習するため、不良サンプルの収集・ラベリングが不要です。

**ハイブリッド判定ルール** — ルールとMLの両方がPASSのときのみ最終PASSです。ルールがPASSと判断した製品でも、MLが明示的にFAILを出せば最終FAILに覆します。逆に、MLがルールのFAILをPASSに戻すことはありません。つまりMLは判定を保守的に強化する方向にのみ作用し、不良品が良品として流出するケース(false-accept)を増やしません。MLモデルがロードされていない環境では、ルール単独の判定に自動フォールバックします。

判定結果には `anomaly_score_worst`(4方向のうち最悪の異常スコア)と `ml_passed`(-1=未使用 / 0=FAIL / 1=PASS)が併せて配信され、HMIダッシュボードにML異常スコアとして表示されます。

基準画像はHMIの基準画像キャプチャモードで正常品を実撮影して作成します。ルール判定の面積比項目が基準画像を使用するため現時点では依然として必要であり、ルール依存を完全に取り除くことは今後の課題です。

### ROS 2 パッケージ・ノード構成

| パッケージ名 | 実行ノード名 | 主な役割 |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | 全体自律シーケンスのFSM制御 (把持 → チャンバー安着 → 検査 → 仕分け → ホーム復帰) |
| | `robot_control_node` | ロボットアームのDynamixel制御、LeRobot ACT把持推論、レール/ターンテーブル指令の中継、E-STOP処理 |
| **`quvi_inspect`** | `inspect_node` | 4方向表面特徴分析 + PatchCore異常検知のハイブリッド良否判定、検査ログ保存、基準画像・MLデータセットのキャプチャモード |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIOベースのリアルタイムWebダッシュボード** (状態モニタリング、MJPEGストリーミング、手動制御) |
| **`quvi_msgs`** | - | カスタムメッセージ (`SystemStatus`, `InspectionResult`, `GraspGoal`, `MotorStatus` など) |
| **`quvi_bringup`** | - | システムランチファイル (`full_system.launch.py`, `vision_pipeline.launch.py`) |

トピック名は `quvi_robot_control/topics.py` で一元管理します。

### Web HMI 主要機能 (ダッシュボード)

* **リアルタイムシステム状態モニタリング**
  * ロボット関節角度の可視化 (`/robot/joint_states` リアルタイムゲージ)
  * リニアレールのトラックモーション (ステーションマップ: INSPECT / PASS / FAIL / BED)
  * ターンテーブルのコンパスダイヤル、FSM段階別フロー図の可視化
  * 検査履歴・統計 (PASS/FAILカウント)
* **リアルタイムMJPEGビデオストリーミング** — サイドカメラ、検査カメラ、検査デバッグビュー(4方向タイル + 判定オーバーレイ)
* **手動制御**
  * 自律シーケンスの開始/停止、**非常停止(E-STOP)** およびリセット
  * リーダー・フォロワーのテレオペレーショントグル、レール/ターンテーブル/LEDの手動駆動
  * ACTモデルのスキャン・選択(ホットスワップ)、デバイスマッピング(カメラ・シリアルポート)の設定と再起動
  * 基準画像キャプチャモード、ML正常品データセット撮影モード

### プロジェクトフォルダ構成

```
QUVI/
├── docker/                  # Docker 開発環境 (Dockerfile, compose)
├── firmware/                # ESP32-S3 レール・ターンテーブル ファームウェア (PlatformIO, micro-ROS)
├── lerobot/                 # LeRobot サブモジュール (OMX 対応ブランチ)
├── src/                     # ROS 2 ソース
│   ├── quvi_msgs/           # カスタムメッセージ定義
│   ├── quvi_bringup/        # ローンチファイル
│   ├── quvi_robot_control/  # ロボットアーム・FSM オーケストレーター・共通ユーティリティ/トピック
│   ├── quvi_inspect/        # 良否判定 + PatchCore 異常検知パッケージ
│   └── quvi_hmi/            # Flask + SocketIO ウェブダッシュボード
├── data/                    # 基準画像、検査ログ、ML データセット・モデル、デバイス設定
├── scripts/                 # ACT 記録/学習、異常検知学習、キャリブレーション・診断スクリプト
├── tests/                   # pytest ロジックテスト
└── docs/                    # 技術設計ドキュメント
```

---

## 使用スタック

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy <2 固定), Hugging Face LeRobot (ACT), PatchCore (WideResNet50)
* **Web HMI**: Flask, Flask-SocketIO (threadingモード), Vanilla JS, HTML5/CSS3 (Industrial Dark Theme)
* **Embedded**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0), PlatformIO

---

## 実行方法 (Docker Environment)

### 1. リポジトリのクローンとサブモジュールの初期化
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
git submodule update --init --recursive
```

### 2. Docker環境の構築
```bash
cd docker
docker compose build
docker compose up -d
```

### 3. ビルドと実行 (ホスト側で)
```bash
./build.sh   # コンテナ起動 + colcon build --symlink-install
./run.sh     # full_system.launch.py 実行
```
* Webブラウザで `http://localhost:5000` にアクセス → HMIダッシュボード。
* 手動実行の場合: コンテナ内で `ros2 launch quvi_bringup full_system.launch.py`

### 4. テスト
```bash
docker exec quvi-dev bash -c "cd /workspace && python3 -m pytest tests/ -q"
```

---

## LeRobot ACT 模倣学習ガイド

ロボットアームの把持(Zone 1)は、LeRobot ACT (Action Chunking with Transformers) 模倣学習に基づくビジュオモーター制御で行います。

### 1. テレオペレーションによるデータ収集
リーダー・フォロワー方式の実演データをヘルパースクリプトで記録します(ホスト/コンテナのどちらでも実行可能)。
```bash
./scripts/act_record.sh <HF_USER> <エピソード数> <エピソード時間(秒)>
```

### 2. ACTモデルの学習
```bash
./scripts/act_train.sh <HF_USER>
```
CUDAが利用できない場合はCPUにフォールバックし、警告を出力します。

### 3. 推論とデプロイ
HMIダッシュボードでモデルのスキャン・選択によるランタイム切り替えが可能で、**最後に選択したモデルは `data/act_last_model.json` に保存され、再起動時に自動ロード・有効化**されます。保存された選択がない場合は `act_model_path` パラメータのデフォルトパスを使用します。

---

## PatchCore 異常検知の学習パイプライン

```bash
# 0. HMI データセット撮影モードまたは既存 PASS 検査ログから正常品画像を収集
python3 scripts/build_anomaly_dataset.py     # PASS ログ → 角度別 raw/ 整理 + レビューシート生成
# (人が review_sheet_{angle}.png を確認し、不良混入画像を raw/ から削除)

# 1. 角度別メモリバンク学習 + しきい値算定
python3 scripts/train_anomaly_bank.py        # → data/models/bank_{angle}.pt, thresholds.json

# 2. ルール vs ML 一致率レポート (判定信頼度の点検)
python3 scripts/shadow_report.py
```

ランチ引数 `anomaly_enabled`(デフォルト true)でオン/オフを切り替え、モデルファイルが存在しないかロードに失敗した場合は自動的に無効化され、ルール判定のみを使用します。

---

## ESP32-S3 ファームウェアのビルド・書き込み

リニアレール・ターンテーブル・LEDを担当するESP32-S3ファームウェアはPlatformIOプロジェクトです (`firmware/quvi_esp32_firmware/`)。ESP32-S3はCH340ブリッジ経由(`/dev/ttyESP32` udevシンボリックリンク、`scripts/99-esp32.rules`)で接続され、ブートボタンの操作なしで自動リセット書き込みが可能です。

```bash
# ホスト側で実行。micro-ROS agent がポートを掴んでいる場合は先に終了すること。
cd firmware/quvi_esp32_firmware
pio run                                        # コンパイル
pio run -t upload --upload-port /dev/ttyESP32  # 書き込み
```

ホーミング(3段階: 粗探索 → バックオフ → 精密探索)・レール座標系・ソフトリミットなどのハードウェアキャリブレーション定数は `Config.h` で管理します。

---

## チーム情報

- **ソウルロボット高等学校 卒業制作 オールラウンダーチーム**

---

## AI使用内訳 (外部利用内訳の公開)

透明性の原則に基づき、本プロジェクトの制作に活用したAI、オープンソース、外部の支援を公開します。

- **開発支援AI**: Anthropic **Claude Code** (Claude Opus・Sonnet・Haikuモデル)、OpenAI **Codex**、Google **Antigravity** — コード作成・レビュー・デバッグの補助。最終的な設計判断と実機検証はチームが直接実施
- **製品に搭載したAIモデル**: **ACT** 把持ポリシー (LeRobotで自前のテレオペレーションデータを収集して直接学習)、**PatchCore** 異常検知 (WideResNet50バックボーン、自前の正常品画像でメモリバンクを構築)
- **オープンソース**: ROS 2 Jazzy, micro-ROS, LeRobot, OpenCV, PyTorch, Flask/Flask-SocketIO, Dynamixel SDK, PlatformIO など — 詳細な出典は下記[リファレンス](#リファレンス)を参照
- **外部アドバイザー**: なし (指導教員の指導以外に外部機関・企業の助言なし)

---

## リファレンス

- **LeRobot** — Hugging Faceのロボット模倣学習フレームワーク。本プロジェクトはOMX対応ブランチである [ROBOTIS-GIT/lerobot](https://github.com/ROBOTIS-GIT/lerobot) フォークをサブモジュールとして使用 (オリジナル: [huggingface/lerobot](https://github.com/huggingface/lerobot))
- **ACT** — Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705) (RSS 2023) — 把持模倣学習ポリシー
- **ROBOTIS OpenMANIPULATOR-X (OMX)** — [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) — ロボットアームハードウェアおよび [DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- **PatchCore** — Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (CVPR 2022) — 異常検知アルゴリズム
- **micro-ROS** — [micro-ROS-Agent](https://github.com/micro-ROS/micro-ROS-Agent), [micro_ros_platformio](https://github.com/micro-ROS/micro_ros_platformio) — ESP32-S3 ↔ ROS 2 通信

---

## ライセンス

本プロジェクトは [MIT License](LICENSE) の下でライセンスされています。
