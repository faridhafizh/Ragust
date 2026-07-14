mod document_reader;
mod local_embedder;
mod local_llm;
mod vector_store;

use anyhow::{anyhow, Result};
use clap::Parser;
use std::time::Instant;
use vector_store::VectorStore;

/// RAG CLI: tanya-jawab dokumen multi-format pakai AI LOCAL (100% offline).
#[derive(Parser, Debug)]
#[command(
    name = "rag-local",
    about = "RAG (Retrieval-Augmented Generation) untuk dokumen PDF/DOCX/TXT/MD/HTML, 100% LOCAL AI"
)]
struct Args {
    /// Path ke file dokumen (pdf, docx, txt, md, html)
    #[arg(short, long)]
    pdf: String,

    /// Pertanyaan yang ingin diajukan
    #[arg(short, long)]
    question: String,

    /// Ukuran tiap chunk (karakter)
    #[arg(long, default_value_t = 800)]
    chunk_size: usize,

    /// Overlap antar chunk (karakter)
    #[arg(long, default_value_t = 150)]
    overlap: usize,

    /// Jumlah chunk relevan untuk konteks
    #[arg(long, default_value_t = 4)]
    top_k: usize,

    /// Jumlah fitur TF-IDF maksimum untuk menghemat RAM/CPU
    #[arg(long, default_value_t = 256)]
    max_features: usize,

    /// Jalankan mode ringan: lewati download model lokal besar dan jawab dari konteks hasil retrieval
    #[arg(long)]
    lite: bool,

    /// Gunakan OCR (vision API) untuk PDF berbasis gambar/scan
    #[arg(long)]
    ocr: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let total_start = Instant::now();

    let args = Args::parse();

    // ============================================================
    // INISIALISASI LOCAL LLM (download model pertama kali)
    // ============================================================
    let llm = if args.lite {
        println!("🪶 Mode ringan aktif: melewati model lokal besar untuk menghemat RAM/CPU.");
        None
    } else {
        println!("🚀 Memuat Local AI (ini pertama kali akan download model chat)...");
        println!("   Chat: TinyLlama-1.1B (~600MB)");
        println!();
        Some(local_llm::LocalLlm::new().await?)
    };

    // ============================================================
    // LANGKAH 1: EKSTRAKSI DOKUMEN
    // ============================================================
    println!("📄 Membaca dokumen: {}", args.pdf);
    let read_start = Instant::now();

    let doc_content = if args.ocr {
        println!("🔍 Mode OCR aktif. Mengkonversi PDF ke gambar...");
        let images = document_reader::pdf_pages_to_images(&args.pdf)?;
        let mut full_text = String::new();

        for (i, _img) in images.iter().enumerate() {
            println!(
                "   ⚠️ OCR halaman {} - butuh model vision terpisah. Lewati.",
                i + 1
            );
            full_text.push_str(&format!("[Halaman {} - OCR tidak tersedia]\n", i + 1));
        }

        document_reader::DocumentContent {
            full_text: full_text.clone(),
            pages: vec![document_reader::PageContent {
                page_number: 1,
                text: full_text.clone(),
                hash: document_reader::compute_hash(&full_text),
            }],
            tables: vec![],
            source_hash: document_reader::compute_hash(&full_text),
        }
    } else {
        document_reader::extract_document(&args.pdf)?
    };

    let read_duration = read_start.elapsed();
    let text_len = if doc_content.pages.is_empty() {
        doc_content.full_text.len()
    } else {
        doc_content.pages.iter().map(|page| page.text.len()).sum()
    };

    println!(
        "   ✅ Selesai membaca dalam {:.2}s | Karakter: {}",
        read_duration.as_secs_f64(),
        text_len
    );

    if text_len == 0 {
        return Err(anyhow!("Dokumen kosong."));
    }

    // ============================================================
    // LANGKAH 2: CHUNKING
    // ============================================================
    println!(
        "✂️  Memecah teks menjadi chunk (size={}, overlap={})...",
        args.chunk_size, args.overlap
    );
    let chunk_start = Instant::now();

    let mut all_chunks: Vec<(String, Option<usize>, usize)> = Vec::new();
    let mut global_idx = 0;

    if doc_content.pages.is_empty() {
        let page_chunks =
            document_reader::chunk_text(&doc_content.full_text, args.chunk_size, args.overlap);
        for text in page_chunks {
            all_chunks.push((text, None, global_idx));
            global_idx += 1;
        }
    } else {
        for page in &doc_content.pages {
            let page_chunks =
                document_reader::chunk_text(&page.text, args.chunk_size, args.overlap);
            for text in page_chunks {
                all_chunks.push((text, Some(page.page_number), global_idx));
                global_idx += 1;
            }
        }
    }

    for table in &doc_content.tables {
        let table_text = format!("TABEL (Markdown):\n{}", table.markdown);
        all_chunks.push((table_text, table.page_number, global_idx));
        global_idx += 1;
    }

    let chunk_duration = chunk_start.elapsed();
    println!(
        "   ✅ {} chunk dihasilkan dalam {:.2}s",
        all_chunks.len(),
        chunk_duration.as_secs_f64()
    );

    // ============================================================
    // LANGKAH 3: TF-IDF EMBEDDING (PURE RUST)
    // ============================================================
    println!(
        "🔢 Membuat TF-IDF embedding untuk {} chunk (PURE RUST)...",
        all_chunks.len()
    );
    let embed_start = Instant::now();

    let chunk_texts: Vec<&str> = all_chunks.iter().map(|(t, _, _)| t.as_str()).collect();

    // Fit TF-IDF pada seluruh corpus
    let mut embedder = local_embedder::TfIdfEmbedder::with_max_features(args.max_features);
    embedder.fit(chunk_texts.iter().copied());

    // Embed semua chunk
    let embeddings = embedder.embed_batch(&chunk_texts)?;
    let embed_duration = embed_start.elapsed();

    println!(
        "   ✅ {} embedding selesai dalam {:.2}s",
        embeddings.len(),
        embed_duration.as_secs_f64()
    );

    // ============================================================
    // LANGKAH 4: VECTOR STORE
    // ============================================================
    let mut store = VectorStore::new();
    for ((text, page_num, idx), emb) in all_chunks.into_iter().zip(embeddings.into_iter()) {
        store.add(
            text,
            emb,
            &args.pdf,
            &doc_content.source_hash,
            page_num,
            idx,
        );
    }

    // ============================================================
    // LANGKAH 5: SEARCH
    // ============================================================
    println!("🔍 Mencari chunk paling relevan untuk pertanyaan...");
    let query_start = Instant::now();
    let query_embedding = embedder.embed(&args.question);
    let top_matches = store.search(&query_embedding, args.top_k);
    let query_duration = query_start.elapsed();

    println!(
        "   ✅ {} chunk relevan ditemukan dalam {:.2}s",
        top_matches.len(),
        query_duration.as_secs_f64()
    );

    let context = top_matches
        .iter()
        .enumerate()
        .map(|(i, (chunk, score))| {
            let page_info = chunk
                .page_number
                .map(|p| format!(" | halaman {}", p))
                .unwrap_or_default();
            format!(
                "[Konteks {} | skor={:.3}{}]\n{}",
                i + 1,
                score,
                page_info,
                chunk.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // ============================================================
    // LANGKAH 6: CHAT (LOCAL LLM)
    // ============================================================
    let system_prompt = "Kamu adalah asisten yang menjawab pertanyaan HANYA berdasarkan \
        konteks dokumen yang diberikan. Jika jawaban tidak ada di konteks, katakan dengan \
        jujur bahwa informasinya tidak ditemukan dalam dokumen. Jawab dalam Bahasa Indonesia \
        yang jelas dan ringkas.";

    let user_prompt = format!(
        "Konteks dari dokumen:\n\n{}\n\nPertanyaan: {}\n\nJawablah berdasarkan konteks di atas.",
        context, args.question
    );

    let chat_start = Instant::now();
    let answer = if args.lite {
        build_lite_answer(&context, &args.question)
    } else {
        println!("🤖 Meminta jawaban dari Local LLM (TinyLlama-1.1B)...");
        println!("   ⏳ Ini bisa memakan waktu 30-120 detik di CPU...");
        llm.as_ref()
            .unwrap()
            .chat(system_prompt, &user_prompt)
            .await?
    };
    let chat_duration = chat_start.elapsed();

    let total_duration = total_start.elapsed();

    // ============================================================
    // OUTPUT
    // ============================================================
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("                        JAWABAN");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{}", answer);
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "⏱️  Baca {:.2}s | Chunk {:.2}s | Embed {:.2}s | Cari {:.2}s | Chat {:.2}s | Total {:.2}s",
        read_duration.as_secs_f64(),
        chunk_duration.as_secs_f64(),
        embed_duration.as_secs_f64(),
        query_duration.as_secs_f64(),
        chat_duration.as_secs_f64(),
        total_duration.as_secs_f64()
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!("💡 100% LOCAL - Tidak ada data yang dikirim ke cloud!");

    Ok(())
}

fn build_lite_answer(context: &str, question: &str) -> String {
    let mut lines: Vec<&str> = context
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    lines.truncate(6);

    let excerpt = lines.join("\n");
    if excerpt.is_empty() {
        return format!("Saya tidak menemukan informasi yang cukup untuk menjawab pertanyaan: {}", question);
    }

    format!(
        "Ringkasan ringan dari konteks yang paling relevan:\n\n{}\n\nPertanyaan: {}",
        excerpt, question
    )
}
