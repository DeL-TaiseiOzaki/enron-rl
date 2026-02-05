# Claude Code Orchestra

**マルチエージェント協調フレームワーク**

Claude Code が Codex CLI（深い推論）と Gemini CLI（大規模リサーチ）を統合し、各エージェントの強みを活かして開発を加速する。

---

## Why This Exists

| Agent | Strength | Use For |
|-------|----------|---------|
| **Claude Code** | オーケストレーション、ユーザー対話 | 全体統括、タスク管理 |
| **Codex CLI** | 深い推論、設計判断、デバッグ | 設計相談、エラー分析、トレードオフ評価 |
| **Gemini CLI** | 1Mトークン、マルチモーダル、Web検索 | コードベース全体分析、ライブラリ調査、PDF/動画処理 |

**IMPORTANT**: 単体では難しいタスクも、3エージェントの協調で解決できる。

---

## Context Management (CRITICAL)

Claude Code のコンテキストは **200k トークン** だが、ツール定義等で **実質 70-100k** に縮小する。

**YOU MUST** サブエージェント経由で Codex/Gemini を呼び出す（出力が10行以上の場合）。

| 出力サイズ | 方法 | 理由 |
|-----------|------|------|
| 1-2文 | 直接呼び出しOK | オーバーヘッド不要 |
| 10行以上 | **サブエージェント経由** | メインコンテキスト保護 |
| 分析レポート | サブエージェント → ファイル保存 | 詳細は `.claude/docs/` に永続化 |

```
# MUST: サブエージェント経由（大きな出力）
Task(subagent_type="general-purpose", prompt="Codexに設計を相談し、要約を返して")

# OK: 直接呼び出し（小さな出力のみ）
Bash("codex exec ... '1文で答えて'")
```

---

## Quick Reference

### Codex を使う時

- 設計判断（「どう実装？」「どのパターン？」）
- デバッグ（「なぜ動かない？」「エラーの原因は？」）
- 比較検討（「AとBどちらがいい？」）

→ 詳細: `.claude/rules/codex-delegation.md`

### Gemini を使う時

- リサーチ（「調べて」「最新の情報は？」）
- 大規模分析（「コードベース全体を理解して」）
- マルチモーダル（「このPDF/動画を見て」）

→ 詳細: `.claude/rules/gemini-delegation.md`

---

## Workflow

```
/startproject <機能名>
```

1. Gemini がリポジトリ分析（サブエージェント経由）
2. Claude が要件ヒアリング・計画作成
3. Codex が計画レビュー（サブエージェント経由）
4. Claude がタスクリスト作成
5. **別セッションで実装後レビュー**（推奨）

→ 詳細: `/startproject`, `/plan`, `/tdd` skills

---

## Tech Stack

- **Python** / **uv** (pip禁止)
- **ruff** (lint/format) / **ty** (type check) / **pytest**
- `poe lint` / `poe test` / `poe all`

→ 詳細: `.claude/rules/dev-environment.md`

---

## Documentation

| Location | Content |
|----------|---------|
| `.claude/rules/` | コーディング・セキュリティ・言語ルール |
| `.claude/docs/DESIGN.md` | 設計決定の記録 |
| `.claude/docs/research/` | Gemini調査結果 |
| `.claude/logs/cli-tools.jsonl` | Codex/Gemini入出力ログ |

---

## Language Protocol

- **思考・コード**: 英語
- **ユーザー対話**: 日本語

---

## Current Project: ART-e RL Ablation Experiments

### Context
- Goal: Run ablation experiments for a paper comparing ART-e RL, Multi-hop RL, CLI Agent, and No RL baselines across different model sizes and evaluation types
- Key files: `configs/art_e_rl/` (8b.toml, 4b.toml, 1.7b.toml)
- Environment: `art_e` (Enron email search QA with tool use)
- GPU setup: 2 GPUs (inference=[0], trainer=[1])

### Experiment Matrix (20 experiments total)

| Training | Sizes | Eval |
|----------|-------|------|
| ART-e RL | 1.7B, 4B, 8B | Multi-hop, Single-hop |
| Multi-hop RL | 1.7B, 4B, 8B | Multi-hop, Single-hop |
| CLI Agent | - | Multi-hop, Single-hop |
| No RL (GPT-5, Opus 4.5, GPT-5-mini) | - | Multi-hop, Single-hop |

### Training Dataset
- Source: `corbt/enron_emails_sample_questions` filtered to `vince.kaminski@enron.com`
- Local path: `data/art_e_vince_kaminski/` (train: 2,510, eval: 376)
- art_e environment modified: `dataset_path` parameter added to `load_environment()`

### Decisions
- Model family: Qwen3 Base (not Instruct)
- Hyperparameters: Unified across all model sizes (lr=5e-6, LoRA rank=16, batch_size=128)
- Multi-hop QA dataset: User-created (not yet integrated)

### Run Commands
```bash
uv run rl @ configs/art_e_rl/8b.toml
uv run rl @ configs/art_e_rl/4b.toml
uv run rl @ configs/art_e_rl/1.7b.toml
```
