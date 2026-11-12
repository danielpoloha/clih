mod api;
mod config;
mod session;
mod tools;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAX_STEPS: usize = 50;

/// Id of the session most recently saved by this process, for the exit hint
/// (also read from the Ctrl-C handler, hence a global).
static SAVED_SESSION: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Print how to resume the saved session (to stderr, so piped stdout stays clean).
fn print_resume_hint(prefix: &str) {
    if let Some(id) = SAVED_SESSION.lock().ok().and_then(|g| g.clone()) {
        eprintln!("{DIM}{prefix}{RESET}clih -r {id}");
    }
}

// ANSI styles
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

#[derive(Parser)]
#[command(name = "clih", version, about = "Agentic coding assistant for any OpenAI-compatible LLM (local or remote)")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Model profile name from the config file
    #[arg(short, long)]
    model: Option<String>,
    /// Run a single prompt non-interactively and exit (stdin is appended if piped)
    #[arg(short, long)]
    prompt: Option<String>,
    /// Auto-approve all tool calls (no permission prompts)
    #[arg(short = 'y', long)]
    yes: bool,
    /// Disable tools (plain chat)
    #[arg(long)]
    no_tools: bool,
    /// Continue the most recent session in this project
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    cont: bool,
    /// Resume a saved session by number or id (no value: pick from a list)
    #[arg(short = 'r', long, num_args = 0..=1, default_missing_value = "")]
    resume: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a config template at ~/.config/clih/config.toml
    Init {
        #[arg(long)]
        force: bool,
    },
    /// List configured model profiles
    Models,
    /// Print the config file path
    ConfigPath,
    /// List saved sessions for the current project
    Sessions,
}

struct Agent {
    cfg: config::Config,
    client: api::Client,
    profile_name: String,
    messages: Vec<Value>,
    auto_yes: bool,
    use_tools: bool,
    always_allowed: HashSet<String>,
    cancel: Arc<AtomicBool>,
    tool_defs: Value,
    session: session::Session,
}

impl Agent {
    fn new(cfg: config::Config, model: Option<&str>, auto_yes: bool, no_tools: bool, cancel: Arc<AtomicBool>) -> Result<Self> {
        let (name, profile) = cfg.profile(model)?;
        let use_tools = profile.tools && !no_tools;
        let always_allowed = cfg.auto_approve.iter().cloned().collect();
        let mut a = Agent {
            client: api::Client::new(profile)?,
            profile_name: name,
            messages: Vec::new(),
            auto_yes,
            use_tools,
            always_allowed,
            cancel,
            tool_defs: tools::definitions(),
            session: session::Session::new(""),
            cfg,
        };
        a.reset();
        Ok(a)
    }

    /// Start a fresh conversation (a new session; previous ones stay on disk).
    fn reset(&mut self) {
        self.messages = vec![json!({"role": "system", "content": system_prompt(&self.cfg, self.use_tools)})];
        self.session = session::Session::new(&self.profile_name);
    }

    /// Load a saved session; switches to its model profile if that still exists.
    fn resume(&mut self, s: session::Session) {
        if s.model != self.profile_name && self.cfg.models.contains_key(&s.model) {
            let _ = self.switch_model(&s.model.clone());
        }
        self.messages = vec![json!({"role": "system", "content": system_prompt(&self.cfg, self.use_tools)})];
        self.messages.extend(s.messages.iter().cloned());
        self.session = s;
        *SAVED_SESSION.lock().unwrap() = Some(self.session.id.clone());
    }

    fn persist(&mut self) {
        self.session.messages = self.messages[1..].to_vec();
        self.session.model = self.profile_name.clone();
        match self.session.save() {
            Ok(()) if self.session.user_turns() > 0 => {
                if let Ok(mut g) = SAVED_SESSION.lock() {
                    *g = Some(self.session.id.clone());
                }
            }
            Ok(()) => {}
            Err(e) => println!("{RED}warning: could not save session: {e:#}{RESET}"),
        }
    }

    fn run_turn(&mut self, user: &str) -> Result<()> {
        let r = self.run_turn_inner(user);
        self.persist();
        r
    }

    fn switch_model(&mut self, name: &str) -> Result<()> {
        let (name, profile) = self.cfg.profile(Some(name))?;
        self.use_tools = profile.tools;
        self.client = api::Client::new(profile)?;
        self.profile_name = name;
        // Refresh the system prompt (tool availability may differ), keep the conversation.
        self.messages[0] = json!({"role": "system", "content": system_prompt(&self.cfg, self.use_tools)});
        Ok(())
    }

    fn run_turn_inner(&mut self, user: &str) -> Result<()> {
        self.messages.push(json!({"role": "user", "content": user}));
        for _ in 0..MAX_STEPS {
            self.cancel.store(false, Ordering::SeqCst);
            let mut in_reasoning = false;
            let mut printed_any = false;
            let tools = if self.use_tools { Some(&self.tool_defs) } else { None };
            let turn = self.client.chat(&self.messages, tools, &self.cancel, &mut |ev| {
                let mut out = std::io::stdout();
                match ev {
                    api::Event::Reasoning(t) => {
                        if !in_reasoning {
                            let _ = write!(out, "{DIM}");
                            in_reasoning = true;
                        }
                        let _ = write!(out, "{t}");
                    }
                    api::Event::Text(t) => {
                        if in_reasoning {
                            let _ = write!(out, "{RESET}\n\n");
                            in_reasoning = false;
                        }
                        let _ = write!(out, "{t}");
                    }
                }
                printed_any = true;
                let _ = out.flush();
            })?;
            if in_reasoning {
                print!("{RESET}");
            }
            if printed_any {
                println!();
            }

            if turn.cancelled {
                self.messages.push(json!({"role": "assistant", "content": turn.content}));
                println!("{YELLOW}(interrupted){RESET}");
                return Ok(());
            }

            let mut msg = json!({"role": "assistant", "content": turn.content});
            if !turn.tool_calls.is_empty() {
                msg["tool_calls"] = Value::Array(turn.tool_calls.iter().map(|c| c.to_json()).collect());
            }
            self.messages.push(msg);
            if turn.tool_calls.is_empty() {
                return Ok(());
            }

            for call in &turn.tool_calls {
                let result = self.handle_call(call);
                self.messages.push(json!({"role": "tool", "tool_call_id": call.id, "content": result}));
            }
            self.persist();
            if self.cancel.load(Ordering::SeqCst) {
                println!("{YELLOW}(interrupted){RESET}");
                return Ok(());
            }
        }
        println!("{YELLOW}(stopped after {MAX_STEPS} steps){RESET}");
        Ok(())
    }

    fn handle_call(&mut self, call: &api::ToolCall) -> String {
        let args: Value = if call.arguments.trim().is_empty() {
            json!({})
        } else {
            match serde_json::from_str(&call.arguments) {
                Ok(v) => v,
                Err(e) => {
                    println!("{RED}⏺ {}: invalid arguments{RESET}", call.name);
                    return format!("Error: tool arguments are not valid JSON ({e}). Retry with valid JSON.");
                }
            }
        };
        println!("{CYAN}⏺ {BOLD}{}{RESET}{CYAN}({}){RESET}", call.name, tools::truncate(&tools::summary(&call.name, &args), 200));
        preview_change(&call.name, &args);

        if !self.permitted(&call.name) {
            println!("  {RED}✗ denied{RESET}");
            return "The user denied this tool call. Ask them how to proceed.".into();
        }
        let result = tools::execute(&call.name, &args, &self.cancel);
        let lines: Vec<&str> = result.lines().collect();
        for l in lines.iter().take(6) {
            println!("  {DIM}{}{RESET}", tools::truncate(l, 160));
        }
        if lines.len() > 6 {
            println!("  {DIM}… ({} more lines){RESET}", lines.len() - 6);
        }
        result
    }

    fn permitted(&mut self, name: &str) -> bool {
        if self.auto_yes || !tools::needs_permission(name) || self.always_allowed.contains(name) {
            return true;
        }
        if !std::io::stdin().is_terminal() {
            println!("  {DIM}(no TTY for permission prompt; use --yes to allow){RESET}");
            return false;
        }
        print!("  {YELLOW}Allow {name}? [y]es / [n]o / [a]lways this session: {RESET}");
        let _ = std::io::stdout().flush();
        let mut s = String::new();
        if std::io::stdin().read_line(&mut s).is_err() {
            return false;
        }
        match s.trim().to_lowercase().as_str() {
            "y" | "yes" => true,
            "a" | "always" => {
                self.always_allowed.insert(name.to_string());
                true
            }
            _ => false,
        }
    }
}

fn preview_change(name: &str, args: &Value) {
    let show = |prefix: &str, color: &str, text: &str| {
        let lines: Vec<&str> = text.lines().collect();
        for l in lines.iter().take(15) {
            println!("  {color}{prefix} {l}{RESET}");
        }
        if lines.len() > 15 {
            println!("  {DIM}  … ({} more lines){RESET}", lines.len() - 15);
        }
    };
    match name {
        "edit_file" => {
            show("-", RED, args["old_string"].as_str().unwrap_or(""));
            show("+", GREEN, args["new_string"].as_str().unwrap_or(""));
        }
        "write_file" => show("+", GREEN, args["content"].as_str().unwrap_or("")),
        _ => {}
    }
}

fn system_prompt(cfg: &config::Config, use_tools: bool) -> String {
    let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default();
    let mut s = String::from(
        "You are clih, an interactive command-line coding assistant. You help the user with software \
         engineering tasks in their current project: reading and editing code, running commands, \
         debugging, and explaining.\n\n\
         Guidelines:\n\
         - Be concise; output is shown in a terminal.\n\
         - Investigate before acting: read relevant files before editing them.\n\
         - Prefer edit_file for small changes to existing files; use write_file for new files.\n\
         - Verify your work (build, run tests) with bash when appropriate.\n\
         - Do not run destructive commands unless the user clearly asked for them.\n\
         - When done, briefly summarize what you changed.\n",
    );
    if !use_tools {
        s.push_str("- Tools are not available in this session; answer directly.\n");
    }
    s.push_str(&format!("\nEnvironment:\n- Working directory: {cwd}\n- OS: {}\n", std::env::consts::OS));
    for f in ["CLIH.md", "AGENTS.md"] {
        if let Ok(t) = std::fs::read_to_string(f) {
            s.push_str(&format!("\nProject instructions from {f}:\n{}\n", tools::truncate(&t, 20_000)));
            break;
        }
    }
    if let Some(extra) = &cfg.system_prompt {
        s.push_str(&format!("\n{extra}\n"));
    }
    s
}

fn print_models(cfg: &config::Config, current: Option<&str>) {
    let default = cfg.default_model.as_deref();
    for (name, p) in &cfg.models {
        let mark = if Some(name.as_str()) == current { "*" } else { " " };
        let def = if Some(name.as_str()) == default { " (default)" } else { "" };
        println!("{mark} {BOLD}{name}{RESET}{def}  {DIM}{} @ {}{RESET}", p.model, p.base_url);
    }
}

const HELP: &str = "Commands:
  /help              Show this help
  /models            List model profiles
  /model <name>      Switch model profile (keeps conversation)
  /clear, /new       Start a new conversation (the current one stays saved)
  /sessions          List saved sessions for this project
  /resume <n|id>     Resume a saved session
  /delete <n|id>     Delete a saved session
  /yes               Toggle auto-approve of tool calls
  /tools             Toggle tool use on/off
  /exit, /quit       Exit
Tips: end a line with \\ for multi-line input. Ctrl-C interrupts generation; Ctrl-D exits.";

fn repl(agent: &mut Agent) -> Result<()> {
    let mut rl = rustyline::DefaultEditor::new()?;
    // Per-project input history; appended (not overwritten) so concurrent instances don't clobber it.
    let hist = session::history_file();
    let _ = rl.load_history(&hist);
    println!(
        "{BOLD}clih{RESET} {DIM}v{} — model: {} ({}) — /help for commands{RESET}",
        env!("CARGO_PKG_VERSION"),
        agent.profile_name,
        agent.client.profile.model
    );
    if agent.session.user_turns() > 0 {
        print_recap(agent);
    }

    loop {
        let mut input = String::new();
        let mut prompt = format!("{BOLD}>{RESET} ");
        loop {
            match rl.readline(&prompt) {
                Ok(line) => {
                    if let Some(stripped) = line.strip_suffix('\\') {
                        input.push_str(stripped);
                        input.push('\n');
                        prompt = "… ".into();
                        continue;
                    }
                    input.push_str(&line);
                    break;
                }
                Err(rustyline::error::ReadlineError::Interrupted) => {
                    input.clear();
                    break;
                }
                Err(rustyline::error::ReadlineError::Eof) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(input);
        let _ = rl.append_history(&hist);

        if let Some(cmd) = input.strip_prefix('/') {
            let mut parts = cmd.split_whitespace();
            match (parts.next().unwrap_or(""), parts.next()) {
                ("exit" | "quit", _) => break,
                ("help", _) => println!("{HELP}"),
                ("clear" | "new", _) => {
                    print_resume_hint("Previous session saved. Resume with: ");
                    agent.reset();
                    *SAVED_SESSION.lock().unwrap() = None;
                    println!("{DIM}New conversation started.{RESET}");
                }
                ("sessions", _) | ("resume", None) => session::print_list(&session::list(), Some(&agent.session.id)),
                ("resume", Some(key)) => match session::find(key) {
                    Ok(s) => {
                        agent.resume(s);
                        print_recap(agent);
                    }
                    Err(e) => println!("{RED}{e}{RESET}"),
                },
                ("delete", Some(key)) => match session::find(key) {
                    Ok(s) if s.id == agent.session.id => println!("{RED}Can't delete the active session; /new first.{RESET}"),
                    Ok(s) => match session::delete(&s) {
                        Ok(()) => println!("{DIM}Deleted {}{RESET}", s.id),
                        Err(e) => println!("{RED}{e:#}{RESET}"),
                    },
                    Err(e) => println!("{RED}{e}{RESET}"),
                },
                ("models", _) | ("model", None) => print_models(&agent.cfg, Some(&agent.profile_name)),
                ("model", Some(name)) => match agent.switch_model(name) {
                    Ok(()) => println!("{DIM}Switched to {} ({}){RESET}", agent.profile_name, agent.client.profile.model),
                    Err(e) => println!("{RED}{e}{RESET}"),
                },
                ("yes", _) => {
                    agent.auto_yes = !agent.auto_yes;
                    println!("{DIM}Auto-approve: {}{RESET}", if agent.auto_yes { "on" } else { "off" });
                }
                ("tools", _) => {
                    agent.use_tools = !agent.use_tools;
                    agent.messages[0] = json!({"role": "system", "content": system_prompt(&agent.cfg, agent.use_tools)});
                    println!("{DIM}Tools: {}{RESET}", if agent.use_tools { "on" } else { "off" });
                }
                _ => println!("{RED}Unknown command. /help for a list.{RESET}"),
            }
            continue;
        }

        if let Err(e) = agent.run_turn(input) {
            println!("{RED}Error: {e:#}{RESET}");
        }
        println!();
    }
    Ok(())
}

/// Show which session is active and the last exchange.
fn print_recap(agent: &Agent) {
    let s = &agent.session;
    let n = s.user_turns();
    println!("{DIM}Resumed session {} — {n} turn{} — model {}{RESET}", s.id, if n == 1 { "" } else { "s" }, agent.profile_name);
    let last = |role: &str| {
        agent.messages.iter().rev()
            .find(|m| m["role"] == role && m["content"].as_str().is_some_and(|c| !c.trim().is_empty()))
            .and_then(|m| m["content"].as_str())
    };
    if let Some(u) = last("user") {
        println!("{BOLD}>{RESET} {}", tools::truncate(u.trim(), 300));
    }
    if let Some(a) = last("assistant") {
        println!("{DIM}{}{RESET}", tools::truncate(a.trim(), 600));
    }
    println!();
}

/// Interactive picker for `clih -r` with no argument.
fn pick_session() -> Result<Option<session::Session>> {
    let all = session::list();
    session::print_list(&all, None);
    if all.is_empty() {
        return Ok(None);
    }
    print!("Resume which session? [number or id, Enter = 1]: ");
    std::io::stdout().flush()?;
    let mut s = String::new();
    std::io::stdin().read_line(&mut s)?;
    let key = s.trim();
    session::find(if key.is_empty() { "1" } else { key }).map(Some)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{RED}error:{RESET} {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Some(Cmd::Init { force }) => {
            let p = config::init(force)?;
            println!("Created {}\nEdit it to add your model endpoints, then run `clih`.", p.display());
            return Ok(());
        }
        Some(Cmd::ConfigPath) => {
            println!("{}", config::config_path().display());
            return Ok(());
        }
        Some(Cmd::Models) => {
            print_models(&config::load()?, None);
            return Ok(());
        }
        Some(Cmd::Sessions) => {
            session::print_list(&session::list(), None);
            return Ok(());
        }
        None => {}
    }

    let cfg = config::load()?;
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let c = cancel.clone();
        ctrlc::set_handler(move || {
            // Second Ctrl-C while already cancelling exits hard.
            if c.swap(true, Ordering::SeqCst) {
                print!("{RESET}");
                eprintln!();
                print_resume_hint("Resume this session with: ");
                std::process::exit(130);
            }
        })?;
    }
    let mut agent = Agent::new(cfg, cli.model.as_deref(), cli.yes, cli.no_tools, cancel)?;

    let to_resume = if cli.cont {
        match session::latest() {
            Some(s) => Some(s),
            None => anyhow::bail!("No saved sessions for this project."),
        }
    } else {
        match cli.resume.as_deref() {
            Some("") => pick_session()?,
            Some(key) => Some(session::find(key)?),
            None => None,
        }
    };
    if let Some(s) = to_resume {
        agent.resume(s);
        // An explicit --model overrides the session's saved model.
        if let Some(m) = &cli.model {
            agent.switch_model(m)?;
        }
        if cli.no_tools {
            agent.use_tools = false;
            agent.messages[0] = json!({"role": "system", "content": system_prompt(&agent.cfg, false)});
        }
    }

    if let Some(mut prompt) = cli.prompt {
        if !std::io::stdin().is_terminal() {
            let mut piped = String::new();
            std::io::stdin().read_to_string(&mut piped)?;
            if !piped.trim().is_empty() {
                prompt = format!("{prompt}\n\n<stdin>\n{piped}\n</stdin>");
            }
        }
        let r = agent.run_turn(&prompt);
        print_resume_hint("Continue with: ");
        return r;
    }
    let r = repl(&mut agent);
    print_resume_hint("Resume this session with: ");
    r
}
