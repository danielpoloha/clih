# clih

An agentic coding assistant for the terminal, in the style of Claude Code, that works with any
LLM behind an OpenAI-compatible `/v1/chat/completions` API: Ollama, llama.cpp, vLLM, LM Studio,
OpenAI, OpenRouter, and others.

## Build

```sh
# Dynamically linked against glibc
cargo build --release                      # -> target/release/clih

# Fully static, runs on any x86_64 Linux
rustup target add x86_64-unknown-linux-musl
CC_x86_64_unknown_linux_musl=gcc cargo build --release --target x86_64-unknown-linux-musl
# -> target/x86_64-unknown-linux-musl/release/clih
```

## Setup

```sh
clih init          # writes ~/.config/clih/config.toml (mode 600)
$EDITOR ~/.config/clih/config.toml
```

```toml
default_model = "local"
auto_approve = []            # e.g. ["edit_file", "write_file", "bash"]

[models.local]
base_url = "http://192.168.1.50:11434/v1"
model = "qwen2.5-coder:14b"
api_key = ""

[models.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"   # or: api_key = "sk-..."
```

Optional fields for each profile: `temperature`, `max_tokens`, `stream` (default true),
`tools` (default true; set it to false for models without function calling), and `timeout_secs`.
Set `CLIH_CONFIG_DIR` to use a different config directory.

## Usage

```sh
clih                         # interactive REPL with the default model
clih -m openai               # choose a profile
clih -p "explain src/main.rs"          # one-shot
git diff | clih -p "review this" -y    # piped stdin is appended to the prompt
clih models                  # list profiles
```

REPL commands: `/help`, `/models`, `/model <name>`, `/new` (or `/clear`), `/sessions`,
`/resume <n|id>`, `/delete <n|id>`, `/yes`, `/tools`, `/exit`.

## Sessions

Every conversation is saved automatically, separately for each project (the directory you start
`clih` in), so several instances can work on different projects, or on the same one, at once.

```sh
clih sessions        # list this project's sessions (1 = most recent)
clih -c              # continue the most recent session
clih -r              # pick a session from a list
clih -r 3            # resume by number, or by id or id prefix (e.g. -r 20261105-1430)
clih -c -p "now add tests"   # continue non-interactively
```

Sessions are saved after each turn and each tool round, so an interrupted run loses little.
When `clih` exits, including after `-p` and a forced double Ctrl-C, it prints the exact command to
resume the session (`clih -r <id>`). `/new` prints the same for the session you're leaving. The hint
goes to stderr, so it doesn't mix with piped output.
Resuming switches back to the session's model unless `-m` is given. Files are stored under
`~/.config/clih/projects/<project>/`: `sessions/<id>.json`, plus `history.txt` for up-arrow input
history, which is appended so that concurrent instances don't overwrite each other's history.
Ctrl-C stops the current generation or command. Press Ctrl-C twice to force quit.

**Tools available to the model:** `read_file`, `write_file`, `edit_file`, `bash`, `list_dir`,
`grep`, and `find_files`. `write_file`, `edit_file`, and `bash` ask for permission first unless
you pass `-y` or list them in `auto_approve`.

If the current directory contains `CLIH.md` or `AGENTS.md`, it is added to the system prompt as
project instructions.
