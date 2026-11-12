use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const MAX_OUTPUT: usize = 30_000;
const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", ".venv", "__pycache__", "dist", "build"];

pub fn definitions() -> Value {
    json!([
        tool("read_file", "Read a text file. Returns lines prefixed with line numbers.", json!({
            "path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
            "offset": {"type": "integer", "description": "1-based line to start from (optional)"},
            "limit": {"type": "integer", "description": "Max number of lines (default 2000)"}
        }), &["path"]),
        tool("write_file", "Create or overwrite a file with the given content.", json!({
            "path": {"type": "string"},
            "content": {"type": "string"}
        }), &["path", "content"]),
        tool("edit_file", "Replace an exact string in a file. old_string must match exactly (including whitespace) and be unique unless replace_all is true. Read the file first.", json!({
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"}
        }), &["path", "old_string", "new_string"]),
        tool("bash", "Run a shell command with bash -c in the current working directory. Returns stdout, stderr and exit code.", json!({
            "command": {"type": "string"},
            "timeout_secs": {"type": "integer", "description": "Default 120, max 600"}
        }), &["command"]),
        tool("list_dir", "List entries of a directory (directories end with '/').", json!({
            "path": {"type": "string", "description": "Default: current directory"}
        }), &[]),
        tool("grep", "Search file contents with a regular expression, recursively. Returns path:line:text.", json!({
            "pattern": {"type": "string", "description": "Rust regex syntax"},
            "path": {"type": "string", "description": "File or directory (default: cwd)"},
            "include": {"type": "string", "description": "Glob filter for files, e.g. *.rs"}
        }), &["pattern"]),
        tool("find_files", "Find files whose relative path matches a glob, e.g. **/*.toml or src/*.rs.", json!({
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Root directory (default: cwd)"}
        }), &["pattern"]),
    ])
}

fn tool(name: &str, desc: &str, props: Value, required: &[&str]) -> Value {
    json!({"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}
    }})
}

pub fn needs_permission(name: &str) -> bool {
    matches!(name, "write_file" | "edit_file" | "bash")
}

/// Short human-readable summary of a call, for display.
pub fn summary(name: &str, args: &Value) -> String {
    let s = |k: &str| args[k].as_str().unwrap_or("").to_string();
    match name {
        "bash" => s("command"),
        "grep" | "find_files" => {
            let p = s("path");
            if p.is_empty() { s("pattern") } else { format!("{} in {}", s("pattern"), p) }
        }
        "list_dir" => { let p = s("path"); if p.is_empty() { ".".into() } else { p } }
        _ => s("path"),
    }
}

pub fn execute(name: &str, args: &Value, cancel: &AtomicBool) -> String {
    let r = match name {
        "read_file" => read_file(args),
        "write_file" => write_file(args),
        "edit_file" => edit_file(args),
        "bash" => bash(args, cancel),
        "list_dir" => list_dir(args),
        "grep" => grep(args),
        "find_files" => find_files(args),
        _ => Err(anyhow::anyhow!("unknown tool '{name}'")),
    };
    match r {
        Ok(s) => truncate(&s, MAX_OUTPUT),
        Err(e) => format!("Error: {e:#}"),
    }
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args[key].as_str().with_context(|| format!("missing string argument '{key}'"))
}

fn resolve(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = dirs::home_dir() {
            return h.join(rest);
        }
    }
    PathBuf::from(p)
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n... [truncated, {} bytes total]", &s[..end], s.len())
}

fn read_file(args: &Value) -> Result<String> {
    let path = resolve(str_arg(args, "path")?);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let offset = args["offset"].as_u64().unwrap_or(1).max(1) as usize;
    let limit = args["limit"].as_u64().unwrap_or(2000) as usize;
    let mut out = String::new();
    let total = text.lines().count();
    for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
        let line = if line.len() > 2000 { truncate(line, 2000) } else { line.to_string() };
        out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
    }
    if total == 0 {
        return Ok("(empty file)".into());
    }
    if offset - 1 + limit < total {
        out.push_str(&format!("... ({} more lines)\n", total - (offset - 1 + limit)));
    }
    Ok(out)
}

fn write_file(args: &Value) -> Result<String> {
    let path = resolve(str_arg(args, "path")?);
    let content = str_arg(args, "content")?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(&path, content).with_context(|| format!("writing {}", path.display()))?;
    Ok(format!("Wrote {} bytes to {}", content.len(), path.display()))
}

fn edit_file(args: &Value) -> Result<String> {
    let path = resolve(str_arg(args, "path")?);
    let old = str_arg(args, "old_string")?;
    let new = str_arg(args, "new_string")?;
    let all = args["replace_all"].as_bool().unwrap_or(false);
    if old.is_empty() {
        bail!("old_string must not be empty (use write_file to create files)");
    }
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let count = text.matches(old).count();
    if count == 0 {
        bail!("old_string not found in {}", path.display());
    }
    if count > 1 && !all {
        bail!("old_string occurs {count} times; add more context or set replace_all");
    }
    let updated = if all { text.replace(old, new) } else { text.replacen(old, new, 1) };
    std::fs::write(&path, updated)?;
    Ok(format!("Edited {} ({} replacement{})", path.display(), count, if count == 1 { "" } else { "s" }))
}

fn bash(args: &Value, cancel: &AtomicBool) -> Result<String> {
    let cmd = str_arg(args, "command")?;
    let timeout = Duration::from_secs(args["timeout_secs"].as_u64().unwrap_or(120).clamp(1, 600));
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn bash")?;

    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let t_out = std::thread::spawn(move || { let mut b = Vec::new(); let _ = so.read_to_end(&mut b); b });
    let t_err = std::thread::spawn(move || { let mut b = Vec::new(); let _ = se.read_to_end(&mut b); b });

    let start = Instant::now();
    let mut note = String::new();
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break Some(st);
        }
        if cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            note = "\n[interrupted by user]".into();
            break child.wait().ok();
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            note = format!("\n[killed after {}s timeout]", timeout.as_secs());
            break child.wait().ok();
        }
        std::thread::sleep(Duration::from_millis(30));
    };
    let stdout = String::from_utf8_lossy(&t_out.join().unwrap_or_default()).to_string();
    let stderr = String::from_utf8_lossy(&t_err.join().unwrap_or_default()).to_string();

    let mut out = String::new();
    if !stdout.is_empty() {
        out.push_str(&truncate(&stdout, MAX_OUTPUT / 2));
    }
    if !stderr.is_empty() {
        if !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
        out.push_str("[stderr]\n");
        out.push_str(&truncate(&stderr, MAX_OUTPUT / 2));
    }
    let code = status.and_then(|s| s.code()).map(|c| c.to_string()).unwrap_or_else(|| "signal".into());
    if !out.is_empty() && !out.ends_with('\n') { out.push('\n'); }
    out.push_str(&format!("[exit code: {code}]{note}"));
    Ok(out)
}

fn list_dir(args: &Value) -> Result<String> {
    let path = resolve(args["path"].as_str().unwrap_or("."));
    let mut entries: Vec<String> = std::fs::read_dir(&path)
        .with_context(|| format!("listing {}", path.display()))?
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) { format!("{name}/") } else { name }
        })
        .collect();
    entries.sort();
    if entries.is_empty() {
        return Ok("(empty directory)".into());
    }
    Ok(entries.join("\n"))
}

fn walk(root: &Path) -> impl Iterator<Item = walkdir::DirEntry> {
    walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            !(e.depth() > 0 && e.file_type().is_dir()
                && SKIP_DIRS.contains(&e.file_name().to_string_lossy().as_ref()))
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().to_string()
}

fn grep(args: &Value) -> Result<String> {
    let re = regex::Regex::new(str_arg(args, "pattern")?).context("invalid regex")?;
    let root = resolve(args["path"].as_str().unwrap_or("."));
    let include = match args["include"].as_str() {
        Some(g) if !g.is_empty() => Some(globset::Glob::new(g)?.compile_matcher()),
        _ => None,
    };
    let mut out = Vec::new();
    let files: Box<dyn Iterator<Item = PathBuf>> = if root.is_file() {
        Box::new(std::iter::once(root.clone()))
    } else {
        Box::new(walk(&root).map(|e| e.into_path()))
    };
    'files: for path in files {
        if let Some(m) = &include {
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if !m.is_match(&name) && !m.is_match(rel(&root, &path)) {
                continue;
            }
        }
        let Ok(bytes) = std::fs::read(&path) else { continue };
        if bytes.iter().take(8000).any(|&b| b == 0) {
            continue; // binary
        }
        let text = String::from_utf8_lossy(&bytes);
        for (i, line) in text.lines().enumerate() {
            if re.is_match(line) {
                out.push(format!("{}:{}:{}", path.display(), i + 1, truncate(line.trim_end(), 300)));
                if out.len() >= 200 {
                    out.push("... (match limit 200 reached)".into());
                    break 'files;
                }
            }
        }
    }
    if out.is_empty() {
        return Ok("No matches.".into());
    }
    Ok(out.join("\n"))
}

fn find_files(args: &Value) -> Result<String> {
    let m = globset::Glob::new(str_arg(args, "pattern")?)?.compile_matcher();
    let root = resolve(args["path"].as_str().unwrap_or("."));
    let mut out: Vec<String> = walk(&root)
        .map(|e| rel(&root, e.path()))
        .filter(|r| m.is_match(r))
        .take(500)
        .collect();
    out.sort();
    if out.is_empty() {
        return Ok("No files found.".into());
    }
    Ok(out.join("\n"))
}
