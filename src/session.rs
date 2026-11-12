//! Saved conversations, stored per project:
//!   ~/.config/clih/projects/<project-key>/sessions/<id>.json
//!   ~/.config/clih/projects/<project-key>/history.txt   (input line history)
//! The project key is derived from the absolute working directory.

use crate::config;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Clone)]
pub struct Session {
    pub id: String,
    pub cwd: String,
    pub model: String,
    pub created: u64,
    pub updated: u64,
    pub title: String,
    /// Conversation without the system prompt (it is regenerated on resume).
    pub messages: Vec<Value>,
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn cwd() -> PathBuf {
    std::env::current_dir()
        .and_then(|p| p.canonicalize())
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn project_key(p: &Path) -> String {
    p.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '_' { c } else { '-' })
        .collect()
}

/// Per-project data directory (created on demand).
pub fn project_dir() -> PathBuf {
    config::config_dir().join("projects").join(project_key(&cwd()))
}

fn sessions_dir() -> PathBuf {
    project_dir().join("sessions")
}

pub fn history_file() -> PathBuf {
    let d = project_dir();
    let _ = std::fs::create_dir_all(&d);
    d.join("history.txt")
}

impl Session {
    pub fn new(model: &str) -> Self {
        let t = now();
        Session {
            id: new_id(t),
            cwd: cwd().display().to_string(),
            model: model.to_string(),
            created: t,
            updated: t,
            title: String::new(),
            messages: Vec::new(),
        }
    }

    fn path(&self) -> PathBuf {
        sessions_dir().join(format!("{}.json", self.id))
    }

    pub fn user_turns(&self) -> usize {
        self.messages.iter().filter(|m| m["role"] == "user").count()
    }

    /// Persist atomically (write temp file, then rename). Empty sessions are not saved.
    pub fn save(&mut self) -> Result<()> {
        if self.user_turns() == 0 {
            return Ok(());
        }
        if self.title.is_empty() {
            if let Some(first) = self.messages.iter().find(|m| m["role"] == "user") {
                let line = first["content"].as_str().unwrap_or("").lines().next().unwrap_or("");
                self.title = line.chars().take(70).collect();
            }
        }
        self.updated = now();
        let dir = sessions_dir();
        std::fs::create_dir_all(&dir)?;
        let tmp = dir.join(format!(".{}.tmp", self.id));
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, self.path())?;
        Ok(())
    }
}

/// All sessions for the current project, most recently updated first.
pub fn list() -> Vec<Session> {
    let Ok(rd) = std::fs::read_dir(sessions_dir()) else { return Vec::new() };
    let mut v: Vec<Session> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| std::fs::read(e.path()).ok())
        .filter_map(|b| serde_json::from_slice(&b).ok())
        .collect();
    v.sort_by(|a, b| (b.updated, b.created).cmp(&(a.updated, a.created)));
    v
}

/// Resolve a session by list number (1 = most recent), full id, or unique id prefix.
pub fn find(key: &str) -> Result<Session> {
    let all = list();
    if all.is_empty() {
        bail!("No saved sessions for this project.");
    }
    if let Ok(n) = key.parse::<usize>() {
        if n >= 1 && n <= all.len() && key.len() < 6 {
            return Ok(all[n - 1].clone());
        }
    }
    let matches: Vec<&Session> = all.iter().filter(|s| s.id.starts_with(key)).collect();
    match matches.len() {
        1 => Ok(matches[0].clone()),
        0 => bail!("No session matching '{key}'. Use `clih sessions` to list them."),
        n => bail!("'{key}' matches {n} sessions; be more specific."),
    }
}

pub fn latest() -> Option<Session> {
    list().into_iter().next()
}

pub fn delete(s: &Session) -> Result<()> {
    std::fs::remove_file(s.path()).with_context(|| format!("deleting session {}", s.id))
}

pub fn print_list(sessions: &[Session], current: Option<&str>) {
    const DIM: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";
    if sessions.is_empty() {
        println!("{DIM}No saved sessions for this project.{RESET}");
        return;
    }
    let t = now();
    for (i, s) in sessions.iter().enumerate().take(30) {
        let mark = if Some(s.id.as_str()) == current { "*" } else { " " };
        println!(
            "{mark}{:>3}  {}  {DIM}{:>8}  {:>9}  {:<10}{RESET} {}",
            i + 1,
            s.id,
            ago(t.saturating_sub(s.updated)),
            format!("{} turn{}", s.user_turns(), if s.user_turns() == 1 { "" } else { "s" }),
            s.model,
            s.title
        );
    }
    if sessions.len() > 30 {
        println!("{DIM}  … {} older sessions{RESET}", sessions.len() - 30);
    }
}

fn ago(secs: u64) -> String {
    match secs {
        0..=59 => "just now".into(),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86399 => format!("{}h ago", secs / 3600),
        _ => format!("{}d ago", secs / 86400),
    }
}

/// Id like 20261105-143012-a3f1 (UTC timestamp + random suffix), sortable and unique
/// across concurrent instances.
fn new_id(t: u64) -> String {
    let (y, mo, d) = civil_from_days((t / 86400) as i64);
    let s = t % 86400;
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    let rnd = (nanos ^ std::process::id().wrapping_mul(2654435761)) & 0xffff;
    format!("{y:04}{mo:02}{d:02}-{:02}{:02}{:02}-{rnd:04x}", s / 3600, (s / 60) % 60, s % 60)
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}
