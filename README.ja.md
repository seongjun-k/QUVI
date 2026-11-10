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

**AIビジョンロボットアームによる3Dプリンター造形物の無人検査・仕分けシステム**

> 「見ることこそ品質である (Seeing is Quality)」

▶ **[Webデモを試す](https://seongjun-k.github.io/QUVI/)** — ハードウェア不要。良品・不良品の検査サイクルを、実機の録画そのままにブラウザで再生できます。スマートフォンでも見られます。

---

## ミッションと成果指標

**ミッション** — 私たちは、長時間・夜通しプリンターを稼働させるメイカーや小規模工房・出力事業者が、造形物を見張り続けられずに不良を手遅れで見つけ、材料と時間を無駄にする問題を、出力直後にロボットが自動撮影しAIが不良を判定する方式でなくします。

**北極星指標** — 出力が終わった瞬間から判定・分類が終わるまでにかかる時間です。以前の審査フィードバックを受けて北極星を測定可能な区間として再定義し、実測しました。人が確認しに来るまで数十分〜数時間放置されていたものを、**現在の中央値64秒**(実機23回、9月の71秒比 -10%)まで短縮しました。次の目標は、最も長い検査区間(24秒)を短縮して60秒以下にすることです。

大きな失敗(フィラメント切れ・衝突)はプリンターが自ら停止し、最後まで出力された不良(層ずれ・反り・糸引き)をQUVIが検出します。QUVIはKlipperの状態が `complete` のときにのみ検査を開始します。

---

## 本選期間の改善 (2026-10-21 ~ 11-10)

実機ログ(ROSログ4,356ファイル、把持軌跡203件、検査記録268件)と測定スクリプトで数え直した値です。修正前後の基準時点は、把持・学習が新しいVLAモデルに切り替わった11-08 19:10、検査が品種の自動選択を適用した11-09 18:55です。詳細な数値と原データの一覧は [`docs/metrics_20261110.md`](docs/metrics_20261110.md) にあります。

| 指標 | 前 | 後 | 変化 |
| :--- | ---: | ---: | ---: |
| **北極星: 出力完了 → 分類完了の時間(中央値)** | 71秒 (66回) | **64秒** (23回) | **-7秒 (-10%)** |
| 把持動作の完了率 | 53% (37/70) | **77%** (34/44) | **+24pt** |
| エラー・非常停止の割合 | 26% (18/70) | **5%** (2/44) | **-21pt** |
| 空手運搬の割合 | 20% (1/5) | **0%** (0/16) | **-20pt** |
| VLA学習の予測誤差(関節の平均絶対誤差) | 0.1457 | **0.0409** | **-72%** |
| 検査判定の完了率 | 30% (43/143) | **100%** (7/7) | **+70pt** |
| 検査判定不能の件数 | 100件 | **0件** | **-100件** |
| 良品の誤検出率(明るさ1.3倍、box) | 46% | **6%** (2/32) | **-40pt** |
| 新規品目の検査準備時間 | 測定なし | **約15分** | 初測定 |
| 実ユーザーテスト | 0人 | **1人** (Geekble(긱블)のメイカー スドラゴン氏、11-10) | +1人 |

**実ユーザーテスト — Geekble(긱블)のメイカー スドラゴン氏 (2026-11-10)**

QUVIのプロトタイプを実際に使っていただき、インタビューしました。商用化されたら使いたいとの回答でした。(以下の引用は韓国語からの翻訳)

* 「既存のソリューションの多くはオープン型チャンバーでベッドを交換するアプローチでしたが、チャンバー型でも露出型でも適用できるソリューションを考えた点が非常に印象的です。特に、同じ部品を何度も出力しなければならない場面で期待しています。」
* 「実際にソリューションが商用化されてほしい。とても便利そうです。」
* 「メイカーの視点から見て、出力物に生じる問題を素早く解決できる点が良かったです。」

| フィードバック | 反映計画 |
| :--- | :--- |
| チャンバー型・露出型の両方に適用可能 | ベッド交換方式はチャンバー型には使えないという差別化点として確認、比較の根拠として提示 |
| 同じ部品の繰り返し出力に期待 | 反復生産の現場を中核の用途とし、連続サイクル64秒・新規品目の準備約15分を根拠として提示 |
| 出力物の問題を素早く解決 | 北極星指標(出力完了 → 分類完了 64秒)で継続して測定 |

**何を変えたか**

* **把持モデルの再学習** — デモンストレーション115件を新たに収集して112件に精選(9,296フレーム)、SmolVLAを15,000 step学習(`smolvla_1108_box2c_new15k`、loss 0.542 → 0.079)。ホーム姿勢を学習開始姿勢に合わせ、VLAの実行時間をデモの長さと同じ5.5秒に合わせた
* **自前の収集・学習パイプライン** — 外部のCycloスタックへの依存なしに、Web収集ツール(録画・再生・検収) → LeRobot v3.0変換 → 学習スクリプトに一本化。収集と推論が同じカメラの露出・ホワイトバランスプロファイルを使用
* **安定性** — 物理非常停止ボタン(押している間は停止、離すと復帰)、ロボットアームのUSB切断時の自動再接続、検査カメラノードの自動再起動、カメラ3台のUSB帯域の時分割とハブ再接続時の自動復旧
* **検査** — 多品種の検査アセット(box・cube)の登録と、開始時の品種自動選択により、判定不能(品種未選択70件・基準画像不足30件)を0件に。新規品目は良品160枚の撮影(13分)と学習(1~2分)で準備でき、不良サンプル・ラベリングは不要
* **ハードウェア補正** — ターンテーブルへの置き姿勢を +8mm 上方・前方 +10mm 移動、グリッパーの把持力を50%低減

**試したが採用しなかったもの** — 検査の明るさ正規化(11-02)は、box の明るさ1.3倍での誤検出を46% → 4%に減らしましたが、引っかき傷の検出が92% → 50%、角欠けの検出が54% → 8%と見逃しが増え、cube の切れ誤検出が0% → 54%に増えたため却下しました(合成欠陥、品種あたり24枚)。再生周期の30→15Hz(9月の実験)は、震えが2倍に増えたため却下しました。震えの原因は再生周期ではなくポリシー出力自体の振動であることを確認し、次の段階はアクションスムージングです。

**測定の限界**

* 把持動作の完了率は、把持シーケンスを最後まで実行した割合です。グリッパーの空手判定はまだ shadow モードのため反映されておらず、実際に運搬できたかどうかは検査結果で別途確認しました(判定記録のある16サイクルすべてで造形物あり)。
* 学習の予測誤差は、現在のモデルの学習データを基準としたオープンループ評価の値です。
* 検査判定完了率の修正後のサンプルは7件です。北極星の9月の値は以前の集計で、デモではHMIの開始ボタンが出力完了信号の代わりになります。
* 検査の誤検出率は照明の明るさの影響を受けるため、検査チャンバーの照明を一定に保つ必要があります。

---

## 課題定義

3Dプリンターの価格が下がったことで、出力代行や試作、少量生産で事業を営む小規模事業者が増えました。しかしFDM出力において失敗はなくせる事故ではなく、確率として残り続ける定数です。共用出力環境におけるデスクトップFDMの失敗率は41.1%と報告されており(Song & Telenko, *Material and energy loss due to human and machine error in commercial FDM printers*, Journal of Cleaner Production, 2017)、熟練者やプリントファームでも2~8%まで下げるのが限界で、ゼロにはできません。

本当の損失は失敗そのものではなく、**出力が終わった後に誰も確認しない時間**です。最後まで出力された不良は人が見なければ分からず、夜通しプリンターを稼働させる事業者にとって、明け方に終わった出力は朝までベッドを占有し続けることを意味し、その数時間が1日分のサイクルを丸ごと失わせます。損失は検査の人件費ではなく、プリンターの稼働率から生じています。

産業用ビジョン検査装置は数千万ウォン規模の投資が必要で、製品が変わるたびに図面やラベリングデータを作り直さなければならず、注文ごとに違うものを出力する現場には合いません。QUVIは**把持(模倣学習ロボットアーム) → 検査(マシンビジョン) → 仕分け(自動積載)** の全工程を、60万ウォン程度の部品で無人化したシステムです。出力が終わったベッドが投入されると、ロボットアームが造形物を把持し、検査チャンバーのターンテーブル上で4方向から撮影して良否を判定し、結果に応じて合格(PASS)・不良(FAIL)のステーションへ仕分けて積載します。その間、人の手は一切介在しません。

全工程は有限状態機械(FSM)オーケストレーターが自律制御し、把持はSmolVLA(ビジョン・言語・行動モデル)による模倣学習、検査は表面特徴のルール判定とPatchCore異常検知を組み合わせたハイブリッド方式で行います。Klipperプリンターの出力完了信号を受けてシーケンスを開始できます。

---

## 主要な数値 (第1次集計、2026-07-30時点)

本選期間の改善数値は、上記の[本選期間の改善](#本選期間の改善-2026-10-21--11-10)の表を参照してください。以下の把持成功率は以前のACTポリシー・別の集計基準によるもので、上記の表の把持動作の完了率とは直接比較できません。

| 項目 | 値 |
| :--- | :--- |
| 全部品コスト | 約60万ウォン — 数千万ウォン規模の産業用ビジョン検査装置の1/50以下 |
| サイクルタイム | 把持開始から仕分け積載まで約1分10秒(実機録画ベース、良品67秒・不良品64秒) |
| 実機検査サイクル | 累計133回完了(検査ログ基準、2026-06-26~07-30 20:43 KST) |
| 把持成功率 | 実機13回中11回成功、84.6%(2026-07-30実測、人が試験を手動停止した4件を除く。含めると11/17 = 64.7%) |
| ルール・ML判定一致率 | 直近13件(2026-07-15以降)で100% — 検証初期を含む全103件では88.3%(ML判定のない07-08以前の30件を除く) |

集計範囲以降の記録を含む原データ全体では137回、ルール・ML 95/107 = 88.8%、把持 15/17 = 88.2%です。原データは [`docs/metrics/`](docs/metrics/) にあり、`python3 scripts/recount_metrics.py` で再計算できます。Webデモの指標は [W6 GA4 レポート](docs/metrics/QUVI_W6_실측지표.pdf)(2026-08-31~09-27)に基づきます。

---

## アーキテクチャ

### システム構成

| 構成要素 | 技術仕様 | 役割・特徴 |
| :--- | :--- | :--- |
| **メイン制御** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | 全ノードのオーケストレーションおよび状態機械(FSM)制御 |
| **下位制御** | ESP32-S3 + TB6600 (micro-ROS) | リニアレール(ステッピングモーター)・ターンテーブル角度・照明LEDの駆動制御、物理非常停止ボタン |
| **ロボットアーム** | ROBOTIS OMX マニピュレーター (フォロワー) | Dynamixel (XL430/XL330) ベース、リーダー・フォロワー方式のテレオペ対応 |
| **カメラ** | USB UVCカメラ × 3 | サイドカメラ・トップビューカメラ(Zone 1: 把持エリア、VLA入力)、検査カメラ(Zone 2: 品質検査チャンバー)。同じUSB 2.0バスのため帯域を時分割 |
| **プリンター連携** | Klipper + Moonraker (KP3S) | 出力完了の検知 → 自動開始(オプション)、出力終了マクロでベッドを排出 |
| **AI・アルゴリズム** | LeRobot SmolVLA + OpenCV + PatchCore | 模倣学習による把持制御、表面特徴のルール判定、ML異常検知(ハイブリッド結合) |

### 検査方式

検査カメラがターンテーブルの4方向(0°/90°/180°/270°)の画像をキャプチャして2系統で分析し、両者の結果を**組み合わせて**最終判定します。検査基準(基準画像・MLモデル)は品種ごとに登録し、開始時に品種が自動選択されます(現在 box・cube の2種)。

1. **表面特徴のルール判定** — 角度別のworst-caseでPASS/FAILを決定
   * Solidity (凸包に対する輪郭面積比 — 反りの検出)
   * 面積比 (基準画像との比較 — 造形不足/過剰の検出)。ターンテーブルの偏心による物体-カメラ間距離の変化を相殺するため、**面積/幅² の距離不変正規化**で比較
   * 穴の個数・穴面積比 (層間剥離の検出) — 穴1個からFAIL
   * テクスチャ分散 (ラプラシアン — 糸引きの検出)

   判定しきい値は `src/quvi_inspect/config/inspect_params.yaml` で管理し、HMI表示基準(`dashboard.js` のTHRESHOLDS)との同期を維持します。
2. **PatchCore異常検知** — WideResNet50バックボーンによる角度別メモリバンクで異常スコアを計算。正常品の画像のみで学習するため、不良サンプルの収集・ラベリングが不要です。

**ハイブリッド判定ルール** — ルールとMLの両方がPASSのときのみ最終PASSです。ルールがPASSと判断した製品でも、MLが明示的にFAILと判定すれば最終判定はFAILに覆ります。逆に、MLがルールのFAILをPASSに戻すことはありません。つまりMLは判定を保守的に強化する方向にのみ作用し、不良品が良品として流出するケース(false-accept)を増やしません。MLモデルがロードされていない環境では、ルール単独の判定に自動フォールバックします。

判定結果には `anomaly_score_worst`(4方向のうち最悪の異常スコア)と `ml_passed`(-1=未使用 / 0=FAIL / 1=PASS)が併せて配信され、HMIダッシュボードにML異常スコアとして表示されます。

基準画像は、HMIの基準画像キャプチャモードで正常品を実際に撮影して作成します。ルール判定の面積比項目が基準画像を使用するため現時点では欠かせません。ただし、同じ製品を繰り返し出力する少量生産ではその基準画像自体が正解となるため制約にはならず、基準画像への依存を完全になくすことが今後の課題です。

### ROS 2 パッケージ・ノード構成

| パッケージ名 | 実行ノード名 | 主な役割 |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | 全体自律シーケンスのFSM制御 (把持 → チャンバーへの設置 → 検査 → 仕分け → ホーム復帰) |
| | `robot_control_node` | ロボットアームのDynamixel制御、SmolVLA把持推論(別サイドカープロセス)、レール/ターンテーブル指令の中継、E-STOP処理 |
| | `printer_monitor_node` | Klipper/Moonrakerの出力状態の監視、出力完了信号の発行 (何も動かさない) |
| **`quvi_inspect`** | `inspect_node` | 4方向表面特徴分析 + PatchCore異常検知のハイブリッド良否判定、品種ごとの検査アセット管理、検査ログ保存、基準画像・MLデータセットのキャプチャモード |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIOベースのリアルタイムWebダッシュボード** (状態モニタリング、MJPEGストリーミング、手動制御) |
| **`quvi_msgs`** | - | カスタムメッセージ (`SystemStatus`, `InspectionResult`, `GraspGoal`, `MotorStatus` など) |
| **`quvi_bringup`** | - | システムのlaunchファイル (`full_system.launch.py`, `vision_pipeline.launch.py`) |

トピック名は `quvi_robot_control/topics.py` で一元管理します。

### Web HMI 主要機能 (ダッシュボード)

* **リアルタイムシステム状態モニタリング**
  * ロボット関節角度の可視化 (`/robot/joint_states` リアルタイムゲージ)
  * リニアレールのトラックモーション (ステーションマップ: INSPECT / PASS / FAIL / BED)
  * ターンテーブルのコンパスダイヤル、FSM段階別フロー図の可視化
  * 検査履歴・統計 (PASS/FAILカウント)
* **リアルタイムMJPEGビデオストリーミング** — サイドカメラ、トップビューカメラ、検査カメラ、検査デバッグビュー(4方向タイル + 判定オーバーレイ)
* **手動制御**
  * 自律シーケンスの開始/停止、**非常停止(E-STOP)** およびリセット (物理ボタンは押している間は停止、離すと自動復帰)
  * リーダー・フォロワーのテレオペレーショントグル、レール/ターンテーブル/LEDの手動駆動
  * VLAモデルのスキャン・選択(ホットスワップ)、デバイスマッピング(カメラ・シリアルポート)の設定と再起動
  * 検査品種の選択・登録、基準画像キャプチャモード、ML正常品データセット撮影モード

### プロジェクトフォルダ構成

```
QUVI/
├── docker/                  # Docker 開発環境 (Dockerfile, compose)
├── firmware/                # ESP32-S3 レール・ターンテーブル ファームウェア (PlatformIO, micro-ROS)
├── lerobot/                 # LeRobot サブモジュール (OMX 対応ブランチ)
├── src/                     # ROS 2 ソース
│   ├── quvi_msgs/           # カスタムメッセージ定義
│   ├── quvi_bringup/        # launchファイル
│   ├── quvi_robot_control/  # ロボットアーム・FSM オーケストレーター・共通ユーティリティ/トピック
│   ├── quvi_inspect/        # 良否判定 + PatchCore 異常検知パッケージ
│   └── quvi_hmi/            # Flask + SocketIO ウェブダッシュボード
├── data/                    # 基準画像、検査ログ、ML データセット・モデル、デバイス設定
├── scripts/                 # VLA 学習・評価、異常検知学習、指標集計、キャリブレーション・診断スクリプト
│   ├── collect/             # VLA デモ収集 Web アプリ (録画・再生・検収、ポート 5100)
│   └── vla_sidecar/         # SmolVLA 推論サイドカー (lerobot 0.6.1 の別 venv)
├── tests/                   # pytest ロジックテスト
└── docs/                    # 技術設計ドキュメント
```

---

## 使用スタック

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy <2 固定), Hugging Face LeRobot (SmolVLA), PatchCore (WideResNet50)
* **Web HMI**: Flask, Flask-SocketIO (threadingモード), Vanilla JS, HTML5/CSS3 (Industrial Dark Theme)
* **Embedded**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0), PlatformIO

---

## 実行方法 (Docker Environment)

> **ハードウェアをお持ちでない場合**は、以下の手順の代わりに[Web デモ](https://seongjun-k.github.io/QUVI/)をご利用ください。インストール不要で、実機の録画による検査サイクル全体をブラウザでご覧いただけます。
> 以下の手順はロボットアーム・レール・カメラの実機がある環境を前提とし、ホストは Ubuntu 24.04 + Docker を想定しています。

### 1. リポジトリのクローンとサブモジュールの初期化
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
git submodule update --init --recursive
```

### 2. デバイス udev ルールの導入 (実機接続時に一度)
ESP32-S3 とカメラ3台は固定シンボリックリンク (`/dev/ttyESP32`, `/dev/sidecam`, `/dev/topcam`, `/dev/fixed_cam`) に割り当てないと、launch ファイルの既定デバイスパスと一致しません。
```bash
sudo cp scripts/99-esp32.rules scripts/99-uvc-cameras.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
> カメラのシンボリックリンクは USB ハブの物理ポート位置を基準とするため、ハブを別のポートに挿すと生成されません。その場合は HMI のデバイス設定で raw ノードを直接指定してください。

### 3. Docker環境の構築
```bash
cd docker
docker compose build
docker compose up -d
```

### 4. ビルドと実行 (ホスト側で)
```bash
./build.sh   # コンテナ起動 + colcon build --symlink-install
./run.sh     # full_system.launch.py 実行
```
* Webブラウザで `http://localhost:5000` にアクセス → HMIダッシュボード。
* 手動実行の場合: コンテナ内で `ros2 launch quvi_bringup full_system.launch.py`

### 5. テスト
```bash
docker exec quvi-dev bash -c "cd /workspace && python3 -m pytest tests/ -q"
```

---

## SmolVLA 模倣学習ガイド

ロボットアームの把持(Zone 1)は、LeRobot SmolVLA 模倣学習に基づくビジュオモーター制御で行います。サイドカメラとトップビューカメラの2枚の画像と関節値を入力として受け取ります。

> **模倣学習とは?** 動作を座標やルールでプログラミングする代わりに、人がリーダーアームを持って把持動作を何度か教示し、そのときのカメラ映像とロボット関節値のペアをニューラルネットワークが学習する方式です。学習後のモデルはカメラ映像だけから次の関節動作を自ら生成するため、造形物の位置が多少ずれていても再ティーチングなしで把持できます。SmolVLA は事前学習済みのビジョン・言語モデルの上に行動出力部を載せた小型のVLAポリシーで、少ないデモデータでファインチューニングできます。

quvi-dev のメインPythonは lerobot 0.3.4 / numpy<2 の環境のため、SmolVLA (lerobot 0.6.1) は別 venv のサイドカープロセス (`scripts/vla_sidecar/`) で推論します。

### 1. デモデータの収集 (Web収集ツール)
HMIでテレオペをオンにした後、収集Webアプリ(`http://localhost:5100`)でエピソードを録画・再生し、使用可否を検収します。録画開始姿勢は学習データの平均姿勢に合わせます。
```bash
# ホスト側で: 変換ツール venv の作成 (初回のみ)
docker exec quvi-dev bash /workspace/scripts/collect/setup_tools_venv.sh
# コンテナ quvi-dev 内で: 収集アプリの実行
source /opt/ros/jazzy/setup.bash
PYTHONPATH=/workspace/src/quvi_robot_control:$PYTHONPATH python3 /workspace/scripts/collect/app.py
```

### 2. LeRobot v3.0 データセットへの変換
収集した元データ(`data/episodes_raw/<セッション>`)を15fpsにリサンプリングして LeRobot v3.0 データセットに変換し、品質を点検します。収集アプリから実行するか、直接実行します。
```bash
# コンテナ quvi-dev 内で実行
/workspace/data/collect_tools/venv/bin/python /workspace/scripts/build_lerobot_dataset.py \
    /workspace/data/episodes_raw/<セッション> --out /workspace/data/datasets/<名前>
```

### 3. SmolVLA の学習
```bash
# ホスト側で実行 (8GB VRAMをほぼ使い切るため、先にQUVIを停止すること)
./scripts/smolvla_train.sh <データセット名> <出力名> [steps] [batch]
# 学習済みモデルから続けて学習: BASE_MODEL=<data/models 配下のパス> ./scripts/smolvla_train.sh ...
```
画像拡張はデフォルトで有効で、`scripts/eval_vla_open_loop.py` で録画データに対する予測誤差と震え(反転率)を評価します。

### 4. 推論とデプロイ
HMIダッシュボードでモデルのスキャン・選択によるランタイム切り替えが可能で、**最後に選択したモデルは `data/act_last_model.json` に保存され、再起動時に自動ロード・有効化**されます。保存された選択がない場合は `act_model_path` パラメータのデフォルトパスを使用します。以前のACTポリシー用スクリプト(`act_record.sh`, `act_train.sh`)も残っています。

---

## PatchCore 異常検知の学習パイプライン

```bash
# 0. HMI データセット撮影モードまたは既存 PASS 検査ログから正常品画像を収集
python3 scripts/build_anomaly_dataset.py     # PASS ログ → 角度別 raw/ 整理 + レビューシート生成
# (人が review_sheet_{angle}.png を確認し、不良混入画像を raw/ から削除)

# 1. 角度別メモリバンク学習 + しきい値算定
python3 scripts/train_anomaly_bank.py --product <品種>   # → data/inspection_products/<品種>/models/bank_{angle}.pt, thresholds.json

# 2. ルール vs ML 一致率レポート (判定信頼度の点検)
python3 scripts/shadow_report.py
```

launch引数 `anomaly_enabled`でオン/オフを切り替え、モデルファイルが存在しないかロードに失敗した場合は自動的に無効化され、ルール判定のみを使用します。

---

## ESP32-S3 ファームウェアのビルド・書き込み

リニアレール・ターンテーブル・LEDを担当するESP32-S3ファームウェアはPlatformIOプロジェクトです (`firmware/quvi_esp32_firmware/`)。ESP32-S3はCH340ブリッジ経由(`/dev/ttyESP32` udevシンボリックリンク、`scripts/99-esp32.rules`)で接続され、ブートボタンの操作なしで自動リセット書き込みが可能です。

```bash
# ホスト側で実行。micro-ROS agent がポートを使用している場合は先に終了すること。
cd firmware/quvi_esp32_firmware
pio run                                        # コンパイル
pio run -t upload --upload-port /dev/ttyESP32  # 書き込み
```

ホーミング(3段階: 粗探索 → バックオフ → 精密探索)・レール座標系・ソフトリミットなどのハードウェアキャリブレーション定数は `Config.h` で管理します。

---

## 今後の計画

次の目標は検査の自動化だけではなく、**出力→回収→検査→再出力を人手なしで回す無人ループ**です。プリンター連携はコードとして組み込み済みですが、実際のプリンターを動かす動作であるため、デフォルトはオフにしています。

| 段階 | 状況 |
| :--- | :--- |
| 出力完了の検知 → 自動開始 | 実装済み (`printer_monitor_node` + Moonraker、`auto_start_on_print_done` デフォルト false) |
| 再出力の指示 | 実装済み — 不良判定後に最後のgcodeの再出力を要求 (`auto_reprint_on_fail` デフォルト false) |
| ベッドの着脱 | 計画 — 毎回同じ位置・同じ軌道の繰り返しのため、ロボットアームではなくレール程度の専用機構で処理 |
| 不良品の回収 | 計画 — 選別した不良品を粉砕してフィラメントに再生 |

把持側の次の課題は、ポリシー出力の震えを減らすアクションスムージングと、グリッパーの空手判定の実機投入(現在は shadow モード)です。ハードウェア拡張としては、マグネット式ツールチェンジャー(グリッパーの自動交換)、接近検知による停止(E-STOPに加える安全層)を計画しています。

---

## チーム情報

- **ソウルロボット高等学校 オールラウンダー** — コンテスト出場を目標に活動する部活動です。QUVIは卒業制作として始まり、全国マイスター高校スタープロジェクトへの出品まで続いているプロジェクトです。

---

## AI使用内訳 (外部利用内訳の公開)

透明性の原則に基づき、本プロジェクトの制作に活用したAI、オープンソース、外部の支援を公開します。

- **開発支援AI**: Anthropic **Claude Code** (Claude Fable 5・Opus・Sonnet・Haikuモデル)、OpenAI **Codex**、Google **Antigravity** — コード作成・レビュー・デバッグの補助。最終的な設計判断と実機検証はチームが直接実施
- **製品に搭載したAIモデル**: **SmolVLA** 把持ポリシー (事前学習モデル `lerobot/smolvla_base` を自前のテレオペレーション・デモデータでファインチューニング。以前のバージョンでは **ACT** ポリシーを直接学習)、**PatchCore** 異常検知 (WideResNet50バックボーン、自前の正常品画像でメモリバンクを構築)
- **オープンソース**（ライセンス種別を含む）: ROS 2 Jazzy (Apache-2.0), micro-ROS (Apache-2.0), LeRobot (Apache-2.0), OpenCV (Apache-2.0), PyTorch (BSD-3-Clause), Flask (BSD-3-Clause), Flask-SocketIO (MIT), NumPy (BSD-3-Clause), DYNAMIXEL SDK (Apache-2.0), PlatformIO (Apache-2.0), ROBOTIS cyclo_intelligence (Apache-2.0、一部モジュールを同梱) — 詳細な出典は下記[リファレンス](#リファレンス)を参照
- **発表・デモ動画の音声 (AI TTS)**: 8週目の3分デモ動画は **edge-tts** (Microsoft Edge ニューラル音声、`ko-KR-HyunsuMultilingualNeural`)、ピッチ動画と4週目の3分デモ動画 (第1回デモデイ) は **Typecast** (typecast.ai) で合成した AI 音声を使用
- **Web デモ計測**: Google Analytics 4 — 公開 Web デモの訪問数と検査サイクル完了数を集計。個人を識別する情報は収集しない
- **外部アドバイザー**: なし (指導教員による指導のほか、外部機関・企業からの助言は受けていません)

---

## リファレンス

- **LeRobot** — Hugging Faceのロボット模倣学習フレームワーク。本プロジェクトはOMX対応ブランチである [ROBOTIS-GIT/lerobot](https://github.com/ROBOTIS-GIT/lerobot) フォークをサブモジュールとして使用 (オリジナル: [huggingface/lerobot](https://github.com/huggingface/lerobot))
- **SmolVLA** — Shukor et al., ["SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"](https://arxiv.org/abs/2506.01844) (2025) — 現在の把持ポリシー、ベースモデル [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **ACT** — Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705) (RSS 2023) — 以前の把持模倣学習ポリシー
- **ROBOTIS OpenMANIPULATOR-X (OMX)** — [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) — ロボットアームハードウェアおよび [DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- **PatchCore** — Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (CVPR 2022) — 異常検知アルゴリズム
- **ROBOTIS cyclo_intelligence** — [ROBOTIS-GIT/cyclo_intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence) (Apache-2.0)。`cyclo_brain/policy/lerobot/lerobot_engine/` の推論用モジュール4つ(constants, image_preprocessing, loading, prediction)を2026-09-08時点の原本のまま `scripts/vla_sidecar/lerobot_engine/` に同梱。各ファイルのROBOTIS著作権・Apache-2.0ヘッダーを保持。その後 `image_preprocessing.py` のみ、重複定数を `constants.py` からのimportに置き換える2行の修正(2026-09-17、コミット 5465f84)
- **micro-ROS** — [micro-ROS-Agent](https://github.com/micro-ROS/micro-ROS-Agent), [micro_ros_platformio](https://github.com/micro-ROS/micro_ros_platformio) — ESP32-S3 ↔ ROS 2 通信

---

## ライセンス

本プロジェクトは [MIT License](LICENSE) の下でライセンスされています。ただし `scripts/vla_sidecar/lerobot_engine/` のROBOTIS原本ファイルは [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) に従います。
