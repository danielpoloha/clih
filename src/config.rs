use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const APP: &str = "clih";

fn default_true() -> bool {
    true
}
fn default_timeout() -> u64 {
    600
}

/// One model definition (a "profile") from the config file.
#[derive(Debug, Deserialize, Clone)]
pub struct Profile {
    /// Base URL of the OpenAI-compatible API, e.g. http://192.168.1.10:11434/v1
    pub base_url: String,
    /// Model name sent in the request body.
    pub model: String,
    /// API token (literal). Optional for local servers.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Name of an environment variable holding the token (used if api_key is unset).
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Use SSE streaming. Disable for servers that can't stream tool calls.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Whether the model supports tool/function calling.
    #[serde(default = "default_true")]
    pub tools: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

impl Profile {
    pub fn resolved_key(&self) -> Option<String> {
        if let Some(k) = &self.api_key {
            if !k.is_empty() {
                return Some(k.clone());
            }
        }
        self.api_key_env
            .as_ref()
            .and_then(|v| std::env::var(v).ok())
            .filter(|k| !k.is_empty())
    }
}

#[derive(Debug, Deserialize)]
pub struct Config {
    /// Profile used when --model is not given.
    pub default_model: Option<String>,
    /// Tools that never ask for permission (e.g. ["edit_file", "write_file"]).
    #[serde(default)]
    pub auto_approve: Vec<String>,
    /// Extra text appended to the system prompt.
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub models: BTreeMap<String, Profile>,
}

impl Config {
    pub fn profile(&self, name: Option<&str>) -> Result<(String, Profile)> {
        let name = match name.or(self.default_model.as_deref()) {
            Some(n) => n.to_string(),
            None => match self.models.keys().next() {
                Some(n) => n.clone(),
                None => bail!("No models defined in {}", config_path().display()),
            },
        };
        match self.models.get(&name) {
            Some(p) => Ok((name, p.clone())),
            None => bail!(
                "Model profile '{}' not found. Available: {}",
                name,
                self.models.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

/// ~/.config/clih on Linux (override with CLIH_CONFIG_DIR).
pub fn config_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CLIH_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    dirs::config_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join(APP)
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn load() -> Result<Config> {
    let path = config_path();
    if !path.exists() {
        bail!(
            "No config found at {}.\nRun `{APP} init` to create one, then edit it.",
            path.display()
        );
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn init(force: bool) -> Result<PathBuf> {
    let dir = config_dir();
    let path = config_path();
    if path.exists() && !force {
        bail!("{} already exists (use --force to overwrite)", path.display());
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(&path, TEMPLATE)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(path)
}

const TEMPLATE: &str = r#"# clih configuration
# Each [models.<name>] block defines an OpenAI-compatible endpoint.
# Select one with `clih -m <name>` or `/model <name>` inside the REPL.

default_model = "local"

# Tools that run without asking: read_file, list_dir, grep, find_files are always allowed.
# Add "edit_file", "write_file" or "bash" here to skip their prompts.
auto_approve = []

# Extra instructions appended to the system prompt (optional).
# system_prompt = "Prefer concise answers."

[models.local]
base_url = "http://localhost:11434/v1"   # Ollama; llama.cpp server: http://HOST:8080/v1, LM Studio: http://HOST:1234/v1
model = "qwen2.5-coder:14b"
api_key = ""                              # usually not needed for local servers
temperature = 0.2
# max_tokens = 4096
# stream = true        # set false if the server can't stream tool calls
# tools = true         # set false for models without function calling
# timeout_secs = 600

[models.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
api_key_env = "OPENAI_API_KEY"            # read the token from an env var...
# api_key = "sk-..."                      # ...or store it here directly

[models.openrouter]
base_url = "https://openrouter.ai/api/v1"
model = "deepseek/deepseek-chat"
api_key_env = "OPENROUTER_API_KEY"
"#;
