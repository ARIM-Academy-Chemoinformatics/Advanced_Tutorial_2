# Advanced_Tutorial_3_Chemoinformatics

ARIMデータポータル会員向けセミナー・ワークショップで提供する、ケモインフォマティクス（RDKit・scikit-learn・pymatgen等）を使った材料設計・機械学習の教育用Jupyter Notebook教材シリーズです。分子記述子の計算から、化学空間の可視化、水溶解度予測モデルの構築、仮想化合物ライブラリの生成・スクリーニング、触媒探索まで、一連のマテリアルズインフォマティクス（MI）ワークフローを一通り体験できる構成になっています。


## ノートブック一覧

| No. | ファイル | 内容 | Colabで開く | 使用ライブラリ | データセット |
| --- | --- | --- | --- | --- | --- |
| 1 | [`1_Descriptor.ipynb`](./1_Descriptor.ipynb) | 分子記述子編：RDKitによる分子記述子（物理化学的性質・トポロジカル指標・VSA系記述子等）の計算とMorganフィンガープリントの基礎 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/1_Descriptor.ipynb) | RDKit, pandas, numpy, matplotlib | Delaney水溶解度データセット（1,128化合物） |
| 2 | [`2_Dimensionality_Reduction.ipynb`](./2_Dimensionality_Reduction.ipynb) | 次元削減・化学空間の可視化編：タニモト係数による分子類似性評価、PCA・t-SNE・UMAPによる化学空間の可視化 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/2_Dimensionality_Reduction.ipynb) | RDKit, scikit-learn, umap-learn | Delaney水溶解度データセット（同上） |
| 3 | [`2-2_Fragment_Count.ipynb`](./2-2_Fragment_Count.ipynb) | フラグメントカウント編：部分構造（フラグメント）の出現頻度を特徴量とする手法、Mordred記述子との比較 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/2-2_Fragment_Count.ipynb) | RDKit, Mordred, pandas, scikit-learn | Delaney水溶解度データセット（同上） |
| 4 | [`3_Solubility.ipynb`](./3_Solubility.ipynb) | 溶解度予測モデル構築編：記述子・フィンガープリントを用いたMLR/PLS/RFによる水溶解度予測モデルの構築・比較 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/3_Solubility.ipynb) | RDKit, scikit-learn, cheminfo（同梱パッケージ） | Delaney水溶解度データセット（同上） |
| 5 | [`4_Structure_Generation.ipynb`](./4_Structure_Generation.ipynb) | 構造生成編：BRICSによるフラグメントの組み換えで仮想化合物ライブラリを生成 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/4_Structure_Generation.ipynb) | RDKit（BRICS） | Delaney水溶解度データセット（同上） |
| 6 | [`5_Screening.ipynb`](./5_Screening.ipynb) | スクリーニング編：T2指標・Q値による適用範囲（AD）の設定と、仮想ライブラリの予測モデルによるスクリーニング | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/5_Screening.ipynb) | RDKit, scikit-learn, cheminfo | Delaney水溶解度データセット＋4の仮想ライブラリ |
| 7 | [`6_Catalyst.ipynb`](./6_Catalyst.ipynb) | 触媒探索編：プロピレンオキシド製造用触媒データにpymatgenの元素記述子を組み合わせ、PLS・ランダムフォレストで複数モデルを比較 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-Chemoinformatics/Advanced_Tutorial_2/blob/main/6_Catalyst.ipynb) | pymatgen, scikit-learn, seaborn | プロピレンオキシド製造用触媒データ（75件） |

各ノートブックの冒頭に「対象読者・前提知識・動作環境・版とライセンス」ブロック、章末に「まとめ」「本ノートブックで扱っていないこと（今後の課題）」「演習問題」を掲載しています。`3_Solubility.ipynb`で構築したモデル（`models/`）と`4_Structure_Generation.ipynb`で生成した仮想ライブラリは、`5_Screening.ipynb`に引き継がれます。


## 対象読者・前提知識
- Pythonの基礎文法は理解しているが、RDKit・pandas・scikit-learn・pymatgenなどのライブラリには初めて触れる方を想定
- 統計学・機械学習の予備知識は前提としない

## 動作環境

Google Colab、または以下のバージョンで動作確認しています。

| ライブラリ | バージョン |
| --- | --- |
| Python | 3.10 |
| rdkit | 2026.3.4 |
| scikit-learn | 1.7.2 |
| pandas / numpy | 2.3.3 / 1.26.4 |
| pymatgen | 2025.10.7 |
| umap-learn | 0.5.12 |
| mordred | 1.2.0 |
| seaborn / matplotlib | 0.13.2 / 3.10.9 |

### 実行方法

**ローカル環境の場合**：このフォルダをそのまま開き、各ノートブックを先頭セルから順に実行してください。`data/`・`img/`・`cheminfo/`・`models/`フォルダは本リポジトリに同梱されています。

**Google Colabの場合**：上表の「Colabで開く」バッジから直接開けます。バッジ経由で開いた場合は、各ノートブック冒頭の「Google Colabにおける環境設定」セルを実行すると、必要なデータ・パッケージ一式を含む本リポジトリを自動的にクローンします。ローカル実行時はこのセルの実行は不要です。

## フォルダ構成

```
.
├── 1_Descriptor.ipynb               # 分子記述子編
├── 2_Dimensionality_Reduction.ipynb # 次元削減・化学空間の可視化編
├── 2-2_Fragment_Count.ipynb         # フラグメントカウント編
├── 3_Solubility.ipynb               # 溶解度予測モデル構築編
├── 4_Structure_Generation.ipynb     # 構造生成編（.pyはColab外の補助スクリプト）
├── 5_Screening.ipynb                # スクリーニング編（.pyはColab外の補助スクリプト）
├── 6_Catalyst.ipynb                 # 触媒探索編
├── cheminfo/                        # 3_Solubility・5_Screeningが参照する自作パッケージ
├── data/                            # Delaney水溶解度データセット・触媒データ等
├── img/                             # マークダウン中の図版
├── models/                          # 3_Solubility.ipynbが生成する学習済みモデル
└── LICENSE                          # MIT License
```

## データの出典・ライセンス

- **本リポジトリのコード**：MIT Licenseで公開しています（詳細は[`LICENSE`](./LICENSE)参照）。
- **RDKit本体**：BSD-3-Clauseライセンスで公開されているオープンソースソフトウェアです。
- **Delaney水溶解度データセット**（`1_Descriptor.ipynb`〜`5_Screening.ipynb`で使用）：John S. Delaney, "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure", *J. Chem. Inf. Comput. Sci.*, **44**, 1000–1005 (2004). DOI: [10.1021/ci034243x](https://doi.org/10.1021/ci034243x)
- **プロピレンオキシド製造用触媒データ**（`6_Catalyst.ipynb`で使用）：植村 圭祐、荒川 正幹、船津 公人 "ケモインフォマティックス手法による新規触媒候補の提案", *Journal of Computer Aided Chemistry*, **7**, 69–77 (2006). DOI: [10.2751/jcac.7.69](https://doi.org/10.2751/jcac.7.69)

## 執筆者・レビュー担当者向け

このシリーズのノートブックを追加・修正する場合は、ブラッシュアップ用プロジェクトの`CLAUDE.md`に記載のコーディング規約・レビュー手順（`skills/edu-notebook-brushup/SKILL.md`）に従ってください。
