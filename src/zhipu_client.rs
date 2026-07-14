use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use serde_json::json;

// =============================================================================
// KONSTANTA KONFIGURASI API Z.AI
// =============================================================================

const BASE_URL: &str = "https://open.bigmodel.cn/api/paas/v4";
const EMBEDDINGS_URL: &str = "https://open.bigmodel.cn/api/paas/v4/embeddings";
const CHAT_URL: &str = "https://open.bigmodel.cn/api/paas/v4/chat/completions";
const DEFAULT_EMBEDDING_MODEL: &str = "embedding-3";
const DEFAULT_CHAT_MODEL: &str = "glm-4.7-flash";
const REQUEST_TIMEOUT_SECS: u64 = 120;

// =============================================================================
// STRUCT RESPONSE DARI API
// =============================================================================

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: String,
}

// =============================================================================
// KLIEN Z.AI
// =============================================================================

pub struct ZaiClient {
    api_key: String,
    http: reqwest::Client,
    pub embedding_model: String,
    pub chat_model: String,
}

impl ZaiClient {
    pub fn new(api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            api_key,
            http,
            embedding_model: DEFAULT_EMBEDDING_MODEL.to_string(),
            chat_model: DEFAULT_CHAT_MODEL.to_string(),
        }
    }

    /// Embedding untuk satu teks.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let body = json!({
            "model": self.embedding_model,
            "input": text,
        });

        let resp = self
            .http
            .post(EMBEDDINGS_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let raw = resp.text().await?;

        if !status.is_success() {
            return Err(anyhow!(
                "Z.AI embeddings API error ({}): {}\n\
                 TIPS: Jika error 'Unknown Model', embeddings mungkin belum tersedia \
                 di endpoint global api.z.ai. Pertimbangkan gunakan endpoint China \
                 (open.bigmodel.cn) untuk embeddings, atau provider embedding lain.",
                status, raw
            ));
        }

        let parsed: EmbeddingResponse = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("Gagal parse response embedding: {} | raw: {}", e, raw))?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow!("Response embedding kosong: tidak ada data dalam response."))
    }

    /// Embedding batch dengan progress indicator.
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let mut result = Vec::with_capacity(total);

        for (i, t) in texts.iter().enumerate() {
            print!("      🔄 Chunk {}/{} ... ", i + 1, total);
            let _ = std::io::Write::flush(&mut std::io::stdout());

            let emb = self.embed(t).await?;
            result.push(emb);
            println!("✅");
        }

        Ok(result)
    }

    /// Chat completion dengan system + user prompt.
    pub async fn chat(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: "system",
                content: system_prompt,
            },
            ChatMessage {
                role: "user",
                content: user_prompt,
            },
        ];

        let body = json!({
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.3,
        });

        let resp = self
            .http
            .post(CHAT_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let raw = resp.text().await?;

        if !status.is_success() {
            return Err(anyhow!("Z.AI chat API error ({}): {}", status, raw));
        }

        let parsed: ChatResponse = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("Gagal parse response chat: {} | raw: {}", e, raw))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("Response chat kosong: tidak ada choices dalam response."))
    }

    /// OCR / Vision: ekstrak teks dari gambar (PNG/JPEG bytes) menggunakan glm-4v.
    /// Berguna untuk PDF berbasis gambar/scan.
    pub async fn vision_ocr(&self, image_bytes: &[u8]) -> Result<String> {
        let base64_image = general_purpose::STANDARD.encode(image_bytes);

        let messages = vec![json!({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all readable text from this image. Preserve table structures using markdown format if present. Return only the extracted text, no explanations."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/png;base64,{}", base64_image)
                    }
                }
            ]
        })];

        let body = json!({
            "model": "glm-4v",
            "messages": messages,
        });

        let resp = self
            .http
            .post(CHAT_URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let raw = resp.text().await?;

        if !status.is_success() {
            return Err(anyhow!("Z.AI vision API error ({}): {}", status, raw));
        }

        let parsed: ChatResponse = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("Gagal parse response vision: {} | raw: {}", e, raw))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow!("Response vision kosong."))
    }
}