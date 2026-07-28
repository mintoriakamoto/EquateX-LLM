# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

**EquateX** is the product brand for this project: a fast LLM fine-tuning/RL library plus a local web UI (**EquateX Studio**). It is built on — and is a rebranded fork of — [Unsloth](https://github.com/unslothai/unsloth); the internal Python package, imports, and `unsloth_zoo` dependency keep the `unsloth` name (renaming them is load-bearing), so "unsloth" in code/paths is expected and correct. EquateX is developed by Cook Labs Inc. Upstream Unsloth copyright, `LICENSE`/`COPYING`/SPDX headers are retained per Apache-2.0/AGPL-3.0 (see `NOTICE`); EquateX changes are © 2026 Cook Labs Inc.

Three main pieces:

- `unsloth/` — the core Python library (Triton kernels, model patching, training/saving). Apache-2.0, except `unsloth/utils/` (LGPL) and `unsloth/kernels/moe/` (AGPL).
- `studio/` — EquateX Studio: FastAPI backend + React frontend + optional Tauri desktop app. AGPL-3.0.
- `unsloth_cli/` — the Typer CLI, invoked as `equatex` (brand) or `unsloth` (alias): `equatex studio`, `equatex start <agent>`, `equatex train`, ... AGPL-3.0.

The console entrypoints are `equatex` and `unsloth`, both `= "unsloth_cli:app"` (pyproject.toml). Root `unsloth-cli.py` is an unrelated legacy standalone fine-tuning example script. Version lives at `unsloth/models/_utils.py` (`__version__`).

## Commands

### Lint / format

```bash
pre-commit run --all-files                                          # canonical
ruff check unsloth unsloth_cli studio tests cli.py unsloth-cli.py   # what Lint CI blocks on
python scripts/run_ruff_format.py <files...>                        # canonical formatter
```

- Do NOT run plain `ruff format` — the canonical formatter is `scripts/run_ruff_format.py`, which runs `ruff format` plus `scripts/enforce_kwargs_spacing.py` to apply the house style of **spaces around `=` in keyword arguments**: `foo(x = 1)`, `load_in_4bit = True`. Match this style in all Python you write here.
- Ruff lint is deliberately a syntax/safety net only (`select = ["E9","F63","F7","F82"]`, long ignore list); it is not a style gate. `line-length = 100`, target py311.
- Ruff and the formatter exclude `*chat_templates.py`, `*ollama_template_mappers.py`, `*_auto_install.py`, `*mapper.py` — don't reformat those.

### Tests

Bare `pytest` at repo root deliberately runs ONLY `tests/security` (see `[tool.pytest.ini_options]`). Everything else must be targeted explicitly.

```bash
# Single test file / case
python -m pytest tests/test_model_registry.py -v
python -m pytest tests/test_model_registry.py::test_model_registration -v

# Full CPU repo suite, as Backend CI runs it
PYTHONPATH=$PWD/studio UNSLOTH_COMPILE_DISABLE=1 python -m pytest tests/ -q --tb=short \
  --ignore=tests/qlora --ignore=tests/saving --ignore=tests/utils --ignore=tests/sh \
  --ignore=tests/vllm_compat --ignore=tests/version_compat \
  --ignore=tests/studio/test_hardware_dispatch_matrix.py \
  --ignore=tests/studio/test_is_mlx_dispatch_gate.py \
  --ignore=tests/studio/test_xpu_spoof_pipeline.py \
  -m 'not server and not e2e'

# Hardware-spoof tests mutate globals — run in their own invocation
python -m pytest -q tests/studio/test_hardware_dispatch_matrix.py \
  tests/studio/test_is_mlx_dispatch_gate.py tests/studio/test_xpu_spoof_pipeline.py

# Studio backend suite (large; imports rooted at studio/backend, so run from there)
cd studio/backend && python -m pytest tests/ -q --tb=short --ignore=tests/test_studio_api.py

# Installer shell tests (use bash, not sh — some files use bashisms)
bash tests/sh/test_<name>.sh
sh tests/run_all.sh          # installer-focused runner: auto-discovers tests/sh + key python tests
```

Notes:
- `tests/conftest.py` auto-spoofs a CUDA GPU on CPU-only machines so `import unsloth` succeeds; `tests/_zoo_aggressive_cuda_spoof.py` / `_zoo_rocm_spoof.py` are deeper opt-in spoofs that must be imported before any unsloth/transformers import.
- Markers: `server` (needs studio venv) and `e2e` (needs network) are deselected in CI. `tests/security` has an autouse network blocker (non-loopback connects raise).
- GPU-only, not run in CI's CPU jobs: `tests/qlora/` (run as scripts, e.g. `python tests/qlora/test_unsloth_qlora_train_and_merge.py`), `tests/saving/`, `tests/fast_inference/`, GPU files in `tests/utils/`.
- `tests/version_compat/` and `tests/vllm_compat/` are pinned-symbol drift canaries with their own CI job and dep matrix.
- Meta-tests that police the suite itself: `tests/studio/test_ci_shell_suite_coverage.py` (fails if `run_all.sh`/CI stop discovering `tests/sh/` or skip a file without a documented reason) and `tests/test_enforce_kwargs_spacing.py`.

### Studio frontend (`studio/frontend/`)

```bash
npm run dev          # Vite dev server on 5173; proxies /api and /v1 to backend on 8888
npm run typecheck    # tsc -b
npm run build        # tsc -b && vite build
npm run lint         # eslint
npm run biome:check  # biome (biome:fix to autofix)
npm run i18n:check   # locale parity vs locales/en.ts — run before committing i18n changes
```

There is no JS unit-test runner; frontend verification is typecheck + build + Biome + Python-driven Playwright E2E (`tests/studio/playwright_*.py`, run by the `studio-*-ui-smoke` workflows).

### Running Studio locally

```bash
bash studio/setup.sh --local                    # one-time dev setup (venv, prebuilts, frontend build)
python studio/backend/run.py --port 8888        # backend alone (FastAPI/uvicorn)
cd studio/frontend && npm run dev               # hot-reload frontend against that backend
```

Full packaged builds go through `./build.sh` (frontend build → wheel/sdist; `./build.sh publish` for releases). Desktop app: `npx tauri dev` / `npx tauri build` from `studio/src-tauri/`.

## Architecture

### Core library (`unsloth/`)

**Import order is load-bearing.** `import unsloth` must come before `transformers`, `trl`, and `peft` — `unsloth/_gpu_init.py` warns if they're already in `sys.modules`. `unsloth/__init__.py` is a dispatcher that forks into two mutually exclusive backends before torch is ever imported:

- **MLX branch** (Apple Silicon, torch-free): shims `FastLanguageModel`/`FastModel` onto `FastMLXModel`, installs fake `trl`/`unsloth.trainer` modules into `sys.modules`, and loads select files by `importlib` file spec to dodge torch-pulling `__init__`s. `UNSLOTH_FORCE_GPU_PATH=1` opts out.
- **GPU branch** (`_gpu_init.py`): applies pre-import shims from `import_fixes.py`, version-gates `unsloth_zoo`, imports torch/triton/bitsandbytes in a specific order, star-imports the public surface, then patches TRL trainers.

Key modules:
- `models/loader.py` — dispatch layer. `FastLanguageModel` maps HF `model_type` → per-arch class (`llama.py` is the reference implementation others derive from); `FastModel` (built on `vision.py`'s `FastBaseModel`) is the modern universal path for multimodal/8-bit/full-FT. `models/_utils.py` holds `__version__` and shared patch helpers; `mapper.py` holds 4bit↔16bit model-name tables.
- `models/rl.py` + `rl_replacements.py` — table-driven source rewriting (`RL_*` tables) that regenerates TRL trainers for RL.
- `kernels/` — Triton kernels (cross-entropy, RMS layernorm, RoPE, swiglu/geglu, fused LoRA, FP8) with patch/unpatch functions. `kernels/moe/` is a self-contained fused grouped-GEMM MoE subproject.
- `import_fixes.py` — ~120 compatibility shims for third-party breakage, including `sys.meta_path` finders that stub or block broken optional deps (vllm, causal_conv1d, ...) at import time.
- `registry/` — declarative model catalog: one `_<family>.py` per family with `ModelMeta` + `register_<family>_models()`; see `registry/REGISTRY.md` before adding a family.
- `save.py` (GGUF export, drives llama.cpp), `chat_templates.py`, `trainer.py`, `tokenizer_utils.py`, `device_type.py` (cached `DEVICE_TYPE`: cuda/hip/xpu/mlx).
- Env escape hatches: `UNSLOTH_ALLOW_CPU=1` (CPU-only CI; skips TRL patching so `inspect.getsource`-based drift tests stay valid), `UNSLOTH_COMPILE_DISABLE=1`.

Monkey-patching conventions: idempotency sentinels on rebound functions, meta-path import blockers, and inline transformers-version gates. `_gpu_init.py` `del`s each fix function after use; most modules declare `__all__` (`tests/test_public_api_surface.py` guards the public surface).

### Studio (`studio/`)

- **Backend** (`studio/backend/`, FastAPI): `run.py` is the launcher, `main.py` builds the app, mounts routers, and serves the built frontend. `routes/` = one module per API surface (`/api/train`, `/api/chat`, `/api/models`, OpenAI-compat at `/v1`, ...); `core/` = business logic (`core/inference/llama_cpp.py` manages a `llama-server` subprocess and proxies chat through it; `core/training/`, `core/export/`, `core/data_recipe/`); `models/` = Pydantic schemas; `storage/` = SQLite; `utils/hardware/` = per-vendor GPU detection + the VRAM formula documented in `utils/hardware/VRAM_ESTIMATION.md`.
- **MCP**: `backend/mcp_server.py` exposes Studio's own opt-in MCP control endpoint (see `studio/MCP.md`, enabled via `UNSLOTH_STUDIO_ENABLE_MCP=1` + token). Distinct from `routes/mcp_servers.py`, where Studio is an MCP *client*.
- **Frontend** (`studio/frontend/`, React 19 + Vite + TS + Tailwind v4): routes hand-registered in `src/app/router.tsx`; code organized by feature under `src/features/<feature>/{api,components,hooks,stores,...}` with feature-local Zustand stores; talks to the backend via plain `fetch` through `src/lib/api-base.ts`. i18n: `src/i18n/locales/en.ts` is the complete baseline; other locales may be partial and must fall back to English; preserve interpolation vars; see `src/i18n/README.md`.
- **Desktop** (`studio/src-tauri/`, Tauri v2): wraps the web UI and supervises the local Python backend (start/stop/health, single-owner lease, bundled installers). `studio/package.json` at the studio root is only a lockfile holder for the Tauri CLI.
- **Prebuilt installers** (`studio/install_*_prebuilt.py`, `prebuilt_core.py`): sha256-verified, lock-guarded installs of llama.cpp / whisper.cpp / Node into an isolated `UNSLOTH_HOME` — never system toolchains. `node_prebuilt_pins.json` is the Node trust anchor.

### CLI (`unsloth_cli/`)

`commands/start.py` implements `unsloth start <agent>` (claude, codex, hermes, openclaw, opencode, pi): passthrough Typer commands that find/launch a local Studio server, resolve a model, write a per-agent config into a session-scoped fake HOME, and exec the agent pointed at the local OpenAI-compatible endpoint. `claude_subagent_mcp.py` / `codex_subagent_mcp.py` / `pi_subagent.ts` are the `--as-subagent` stdio-MCP bridges. `_tool_policy.py` is a deliberately tiny pure resolver whose `_LOOPBACK_HOSTS` set is intentionally duplicated in `studio/backend/utils/host_policy.py` — keep them in sync.

## CI expectations

- **Lint CI** runs on every PR: `compileall` on all `.py`, the `ruff check` above, `scripts/verify_import_hoist.py` on changed files, no leftover `breakpoint()`/`pdb.set_trace()`, `bash -n` on all `.sh`, YAML/JSON parse checks. `ruff format --check` is informational only (the kwarg-spacing style intentionally differs from plain ruff).
- **Supply-chain guards** run before any `npm ci`/`cargo fetch`: `scripts/lockfile_supply_chain_audit.py` (lockfile `resolved` URLs must be the npm registry, integrity required), `scripts/check_new_install_scripts.py` (flags new `hasInstallScript` deps), `scripts/scan_packages.py`/`scan_npm_packages.py` against baselines, and `scripts/lint_workflow_triggers.py` (bans `pull_request_target` etc.). If you touch lockfiles, package.json, or workflows, expect these gates.
- pre-commit.ci auto-pushes fixes (ruff, format, allowScripts pin sync) onto PR branches.
- Notebook changes are validated by `scripts/notebook_validator.py` (`drift`, `lint`, `convert`, ...) in `notebooks-ci.yml`.
