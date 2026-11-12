use crate::config::Profile;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn to_json(&self) -> Value {
        let args = if self.arguments.trim().is_empty() { "{}" } else { &self.arguments };
        json!({"id": self.id, "type": "function", "function": {"name": self.name, "arguments": args}})
    }
}

#[derive(Debug, Default)]
pub struct AssistantTurn {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub cancelled: bool,
}

pub enum Event<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

pub struct Client {
    http: reqwest::blocking::Client,
    pub profile: Profile,
    key: Option<String>,
}

impl Client {
    pub fn new(profile: Profile) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(profile.timeout_secs))
            .connect_timeout(Duration::from_secs(15))
            .build()?;
        let key = profile.resolved_key();
        Ok(Self { http, profile, key })
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.profile.base_url.trim_end_matches('/'))
    }

    pub fn chat(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(Event),
    ) -> Result<AssistantTurn> {
        let mut body = json!({
            "model": self.profile.model,
            "messages": messages,
            "stream": self.profile.stream,
        });
        if let Some(t) = tools {
            body["tools"] = t.clone();
        }
        if let Some(t) = self.profile.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(m) = self.profile.max_tokens {
            body["max_tokens"] = json!(m);
        }

        let mut req = self.http.post(self.endpoint()).json(&body);
        if let Some(k) = &self.key {
            req = req.bearer_auth(k);
        }
        let resp = req
            .send()
            .with_context(|| format!("request to {} failed", self.endpoint()))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().unwrap_or_default();
            bail!("HTTP {status}: {}", text.trim());
        }
        if self.profile.stream {
            read_stream(resp, cancel, on_event)
        } else {
            let v: Value = resp.json().context("invalid JSON response")?;
            parse_full(&v, on_event)
        }
    }
}

fn parse_full(v: &Value, on_event: &mut dyn FnMut(Event)) -> Result<AssistantTurn> {
    if let Some(err) = v.get("error") {
        bail!("API error: {err}");
    }
    let msg = &v["choices"][0]["message"];
    let mut turn = AssistantTurn::default();
    for key in ["reasoning_content", "reasoning"] {
        if let Some(r) = msg[key].as_str() {
            on_event(Event::Reasoning(r));
        }
    }
    if let Some(c) = msg["content"].as_str() {
        on_event(Event::Text(c));
        turn.content = c.to_string();
    }
    if let Some(tcs) = msg["tool_calls"].as_array() {
        for (i, tc) in tcs.iter().enumerate() {
            let f = &tc["function"];
            turn.tool_calls.push(ToolCall {
                id: tc["id"].as_str().map(String::from).unwrap_or_else(|| format!("call_{i}")),
                name: f["name"].as_str().unwrap_or_default().to_string(),
                arguments: match &f["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                },
            });
        }
    }
    Ok(turn)
}

fn read_stream(
    resp: reqwest::blocking::Response,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(Event),
) -> Result<AssistantTurn> {
    let reader = BufReader::new(resp);
    let mut turn = AssistantTurn::default();
    let mut calls: Vec<ToolCall> = Vec::new();

    for line in reader.lines() {
        if cancel.load(Ordering::SeqCst) {
            turn.cancelled = true;
            break;
        }
        let line = line.context("stream read error")?;
        let Some(data) = line.trim().strip_prefix("data:") else { continue };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
        if let Some(err) = v.get("error") {
            bail!("API error: {err}");
        }
        let Some(choice) = v["choices"].get(0) else { continue };
        let delta = &choice["delta"];

        for key in ["reasoning_content", "reasoning"] {
            if let Some(r) = delta[key].as_str() {
                if !r.is_empty() {
                    on_event(Event::Reasoning(r));
                }
            }
        }
        if let Some(t) = delta["content"].as_str() {
            if !t.is_empty() {
                on_event(Event::Text(t));
                turn.content.push_str(t);
            }
        }
        if let Some(tcs) = delta["tool_calls"].as_array() {
            for tc in tcs {
                let has_new_id = tc["id"].as_str().is_some_and(|s| !s.is_empty());
                let idx = match tc["index"].as_u64() {
                    Some(i) => i as usize,
                    None if has_new_id || calls.is_empty() => calls.len(),
                    None => calls.len() - 1,
                };
                while calls.len() <= idx {
                    calls.push(ToolCall::default());
                }
                let c = &mut calls[idx];
                if has_new_id {
                    c.id = tc["id"].as_str().unwrap().to_string();
                }
                let f = &tc["function"];
                if let Some(n) = f["name"].as_str() {
                    if c.name.is_empty() {
                        c.name = n.to_string();
                    }
                }
                match &f["arguments"] {
                    Value::String(s) => c.arguments.push_str(s),
                    Value::Null => {}
                    other => c.arguments.push_str(&other.to_string()),
                }
            }
        }
    }

    turn.tool_calls = calls
        .into_iter()
        .filter(|c| !c.name.is_empty())
        .enumerate()
        .map(|(i, mut c)| {
            if c.id.is_empty() {
                c.id = format!("call_{i}");
            }
            c
        })
        .collect();
    Ok(turn)
}
