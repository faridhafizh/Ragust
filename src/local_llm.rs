use anyhow::{anyhow, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;

const LLAMA_RELEASE_BASE: &str = "https://github.com/ggerganov/llama.cpp/releases/download/b3800/";
const MODEL_URL: &str = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
const LLAMA_DIR: &str = "models/llama";

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

    pub async fn chat(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        let full_prompt = format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            system_prompt, user_prompt
        );

        // Failsafe: Ganti null byte (\0) dengan spasi agar tidak ditolak oleh OS Windows
        let safe_prompt = full_prompt.replace('\0', " ");

        let output = tokio::process::Command::new(&self.llama_cli_path)
            .args([
                "-m",
                self.model_path.to_str().unwrap(),
                "-p",
                &safe_prompt, // Gunakan safe_prompt di sini
                "-n",
                "512",
                "-c",
                "2048",
                "--temp",
                "0.3",
                "--no-display-prompt",
                "-ngl",
                "0",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .with_context(|| "Gagal menjalankan llama-cli")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "llama-cli error: {}\nStderr: {}",
                output.status,
                stderr
            ));
        }

        let response = String::from_utf8_lossy(&output.stdout);
        Ok(response.trim().to_string())
    }
}

fn get_llama_asset() -> (String, String) {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    match os {
        "windows" => ("llama-b3800-bin-win-avx2-x64.zip".to_string(), "llama-cli.exe".to_string()),
        "linux" => ("llama-b3800-bin-linux-x64.zip".to_string(), "llama-cli".to_string()),
        "macos" => {
            if arch == "aarch64" {
                ("llama-b3800-bin-macos-arm64.zip".to_string(), "llama-cli".to_string())
            } else {
                ("llama-b3800-bin-macos-x64.zip".to_string(), "llama-cli".to_string())
            }
        },
        _ => ("llama-b3800-bin-linux-x64.zip".to_string(), "llama-cli".to_string()),
    }
}

async fn ensure_llama_cli() -> Result<PathBuf> {
    let dir = PathBuf::from(LLAMA_DIR);
    std::fs::create_dir_all(&dir)?;
    
    let (zip_name, cli_name) = get_llama_asset();
    
    // 1. Cek apakah sudah ada di folder utama atau di dalam subfolder
    if let Some(exe_path) = find_executable(&dir, &cli_name) {
        set_executable_permission(&exe_path);
        return Ok(exe_path);
    }

    println!("⬇️  Downloading llama.cpp binary untuk {}...", std::env::consts::OS);
    let zip_path = dir.join("llama.zip");
    let url = format!("{}{}", LLAMA_RELEASE_BASE, zip_name);
    
    let response = reqwest::get(&url).await?;
    let bytes = response.bytes().await?;
    
    let mut file = std::fs::File::create(&zip_path)?;
    file.write_all(&bytes)?;

    let archive = zip::ZipArchive::new(std::fs::File::open(&zip_path)?)?;
    extract_zip(archive, &dir)?;
    let _ = std::fs::remove_file(&zip_path);

    // 2. Cek lagi setelah diekstrak (otomatis mencari ke dalam subfolder)
    if let Some(exe_path) = find_executable(&dir, &cli_name) {
        set_executable_permission(&exe_path);
        return Ok(exe_path);
    }

    Err(anyhow!("{} tidak ditemukan setelah ekstrak di dalam {}.", cli_name, dir.display()))
}

/// Mencari file executable di folder utama atau di dalam subfolder (1 level)
fn find_executable(dir: &PathBuf, cli_name: &str) -> Option<PathBuf> {
    // Cek di root (langsung di models/llama/)
    let direct_path = dir.join(cli_name);
    if direct_path.exists() {
        return Some(direct_path);
    }
    
    // Cek di dalam subfolder (karena release llama.cpp sering membungkus file di dalam folder)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let candidate = path.join(cli_name);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Mengatur permission agar bisa dieksekusi (khusus Linux/Mac)
#[cfg(unix)]
fn set_executable_permission(path: &PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = std::fs::metadata(path) {
        let mut perms = metadata.permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(path, perms);
    }
}

#[cfg(not(unix))]
fn set_executable_permission(_path: &PathBuf) {
    // Tidak perlu melakukan apa-apa di Windows
}

async fn ensure_model_file() -> Result<PathBuf> {
    let dir = PathBuf::from(LLAMA_DIR);
    std::fs::create_dir_all(&dir)?;
    let model_path = dir.join("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    if model_path.exists() {
        return Ok(model_path);
    }

    println!("⬇️  Downloading TinyLlama-1.1B model (~600MB)...");
    let response = reqwest::get(MODEL_URL).await?;
    let bytes = response.bytes().await?;
    
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