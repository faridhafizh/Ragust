use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

const LLAMA_RELEASE_URL: &str = "https://github.com/ggerganov/llama.cpp/releases/download/b3800/llama-b3800-bin-win-avx2-x64.zip";
const MODEL_URL: &str = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
const LLAMA_DIR: &str = "models/llama";
const LLAMA_CLI: &str = "llama-cli.exe";

pub struct LocalLlm {
    llama_cli_path: PathBuf,
    model_path: PathBuf,
}

impl LocalLlm {
    pub async fn new() -> Result<Self> {
        let llama_cli_path = ensure_llama_cli().await?;
        let model_path = ensure_model_file().await?;

        println!("🤖 Local LLM ready:");
        println!("   Binary: {}", llama_cli_path.display());
        println!("   Model:  {}", model_path.display());

        Ok(Self {
            llama_cli_path,
            model_path,
        })
    }

    /// Chat dengan local LLM via llama-cli subprocess
    pub async fn chat(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let full_prompt = format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            system_prompt, user_prompt
        );

        let output = tokio::process::Command::new(&self.llama_cli_path)
            .args([
                "-m",
                self.model_path.to_str().unwrap(),
                "-p",
                &full_prompt,
                "-n",
                "512", // max tokens
                "-c",
                "2048", // context size
                "--temp",
                "0.3",                 // temperature
                "--no-display-prompt", // don't echo prompt
                "-ngl",
                "0", // no GPU layers (CPU only)
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .with_context(|| "Gagal menjalankan llama-cli.exe")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "llama-cli error: {}\nStderr: {}",
                output.status,
                stderr
            ));
        }

        let response = String::from_utf8_lossy(&output.stdout);
        // Clean up response - remove prompt echo if any
        let response = response.trim();
        Ok(response.to_string())
    }
}

async fn ensure_llama_cli() -> Result<PathBuf> {
    let dir = PathBuf::from(LLAMA_DIR);
    std::fs::create_dir_all(&dir)?;

    let exe_path = dir.join(LLAMA_CLI);
    if exe_path.exists() {
        return Ok(exe_path);
    }

    println!("⬇️  Downloading llama.cpp Windows binary...");
    let zip_path = dir.join("llama.zip");

    let response = reqwest::get(LLAMA_RELEASE_URL).await?;
    let bytes = response.bytes().await?;

    let mut file = std::fs::File::create(&zip_path)?;
    file.write_all(&bytes)?;
    println!("   ✅ Downloaded llama.zip ({} bytes)", bytes.len());

    // Extract zip
    println!("   📦 Extracting...");
    let archive = zip::ZipArchive::new(std::fs::File::open(&zip_path)?)?;
    extract_zip(archive, &dir)?;

    // Clean up zip
    let _ = std::fs::remove_file(&zip_path);

    if !exe_path.exists() {
        return Err(anyhow!(
            "llama-cli.exe tidak ditemukan setelah extract. Cek struktur ZIP."
        ));
    }

    Ok(exe_path)
}

async fn ensure_model_file() -> Result<PathBuf> {
    let dir = PathBuf::from(LLAMA_DIR);
    std::fs::create_dir_all(&dir)?;

    let model_path = dir.join("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    if model_path.exists() {
        println!("   📦 Model ditemukan: {}", model_path.display());
        return Ok(model_path);
    }

    println!("⬇️  Downloading TinyLlama-1.1B model (~600MB)...");
    println!("   Ini hanya sekali saja, mohon tunggu...");

    let response = reqwest::get(MODEL_URL).await?;
    let total_size = response.content_length().unwrap_or(0);
    println!("   Total size: {} MB", total_size / 1024 / 1024);

    let bytes = response.bytes().await?;
    println!("   ✅ Downloaded ({} bytes)", bytes.len());

    let mut file = std::fs::File::create(&model_path)?;
    file.write_all(&bytes)?;

    Ok(model_path)
}

fn extract_zip(mut archive: zip::ZipArchive<std::fs::File>, dest: &PathBuf) -> Result<()> {
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest.join(file.name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outfile = std::fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}
