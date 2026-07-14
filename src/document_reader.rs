use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

pub struct DocumentContent {
    pub full_text: String,
    pub pages: Vec<PageContent>,
    pub tables: Vec<TableContent>,
    pub source_hash: String,
}

#[allow(dead_code)]
pub struct PageContent {
    pub page_number: usize,
    pub text: String,
    pub hash: String,
}

pub struct TableContent {
    pub page_number: Option<usize>,
    pub markdown: String,
}

pub fn extract_document(path: &str) -> Result<DocumentContent> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "pdf" => extract_pdf(path),
        "docx" => extract_docx(path),
        "txt" | "md" | "markdown" => extract_plain_text(path),
        "html" | "htm" => extract_html(path),
        _ => Err(anyhow!(
            "Format file tidak didukung: .{} (didukung: pdf, docx, txt, md, html)",
            ext
        )),
    }
}

fn extract_pdf(path: &str) -> Result<DocumentContent> {
    let raw_text = extract_text_from_pdf(path)?;
    let tables = extract_tables_from_text(&raw_text);
    let source_hash = compute_hash(&raw_text);

    Ok(DocumentContent {
        full_text: String::new(),
        pages: vec![PageContent {
            page_number: 1,
            text: raw_text,
            hash: source_hash.clone(),
        }],
        tables,
        source_hash,
    })
}

pub fn extract_text_from_pdf(path: &str) -> Result<String> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("Gagal mengekstrak teks dari PDF: {}", path))?;
    
    // PENTING: Sanitasi teks dari Null Byte (\0) dan karakter kontrol aneh dari PDF
    let clean_text: String = text
        .chars()
        .filter(|c| !c.is_control() || c.is_whitespace())
        .collect();
        
    Ok(clean_text)
}

fn extract_docx(path: &str) -> Result<DocumentContent> {
    let text = read_docx_text(path)?;
    let tables = Vec::new(); // DOCX tables require more complex XML traversal, omitted for brevity but text is safe
    let source_hash = compute_hash(&text);

    Ok(DocumentContent {
        full_text: text.clone(),
        pages: vec![PageContent {
            page_number: 1,
            text,
            hash: source_hash.clone(),
        }],
        tables,
        source_hash,
    })
}

fn read_docx_text(path: &str) -> Result<String> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut xml_content = String::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        if file.name() == "word/document.xml" {
            file.read_to_string(&mut xml_content)?;
            break;
        }
    }

    if xml_content.is_empty() {
        return Err(anyhow!("word/document.xml tidak ditemukan di dalam .docx"));
    }

    let doc = roxmltree::Document::parse(&xml_content).context("Gagal parsing XML docx")?;
    let mut text = String::new();

    for node in doc.descendants() {
        if node.is_element() {
            let name = node.tag_name().name();
            if name == "p" {
                if !text.is_empty() && !text.ends_with('\n') {
                    text.push('\n');
                }
            } else if name == "tab" {
                text.push('\t');
            } else if name == "br" {
                text.push('\n');
            }
        } else if node.is_text() {
            if let Some(t) = node.text() {
                text.push_str(t);
            }
        }
    }
    Ok(text)
}

fn extract_plain_text(path: &str) -> Result<DocumentContent> {
    let text = std::fs::read_to_string(path).with_context(|| format!("Gagal membaca file: {}", path))?;
    let source_hash = compute_hash(&text);
    Ok(DocumentContent {
        full_text: text.clone(),
        pages: Vec::new(),
        tables: Vec::new(),
        source_hash,
    })
}

fn extract_html(path: &str) -> Result<DocumentContent> {
    let html = std::fs::read_to_string(path).with_context(|| format!("Gagal membaca file HTML: {}", path))?;
    let document = scraper::Html::parse_document(&html);
    
    let body_selector = scraper::Selector::parse("body").unwrap();
    let mut text = String::new();
    if let Some(body) = document.select(&body_selector).next() {
        text = body.text().collect::<Vec<_>>().join(" ");
    }
    
    let tables = extract_html_tables(&document);
    let source_hash = compute_hash(&text);
    
    Ok(DocumentContent {
        full_text: text.clone(),
        pages: vec![PageContent {
            page_number: 1,
            text,
            hash: source_hash.clone(),
        }],
        tables,
        source_hash,
    })
}

fn extract_html_tables(document: &scraper::Html) -> Vec<TableContent> {
    let mut tables = Vec::new();
    let table_selector = scraper::Selector::parse("table").unwrap();
    let tr_selector = scraper::Selector::parse("tr").unwrap();
    let td_selector = scraper::Selector::parse("td, th").unwrap();

    for table in document.select(&table_selector) {
        let mut rows = Vec::new();
        for tr in table.select(&tr_selector) {
            let cells: Vec<String> = tr.select(&td_selector)
                .map(|td| td.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .collect();
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
        if rows.len() >= 2 {
            let mut md = String::new();
            md.push_str(&rows[0].join(" | "));
            md.push('\n');
            md.push_str(&rows[0].iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
            md.push('\n');
            for row in &rows[1..] {
                md.push_str(&row.join(" | "));
                md.push('\n');
            }
            tables.push(TableContent {
                page_number: None,
                markdown: md,
            });
        }
    }
    tables
}

fn extract_tables_from_text(text: &str) -> Vec<TableContent> {
    // Fungsi heuristik tabel teks asli dipertahankan karena sudah cukup baik untuk plain text
    let mut tables = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        if looks_like_table_row(lines[i]) {
            let start = i;
            let mut max_cols = 0;
            while i < lines.len() && looks_like_table_row(lines[i]) {
                let cols = count_columns(lines[i]);
                max_cols = max_cols.max(cols);
                i += 1;
            }
            let table_lines = &lines[start..i];
            if table_lines.len() >= 2 && max_cols >= 2 {
                if let Some(md) = convert_to_markdown_table(table_lines, max_cols) {
                    tables.push(TableContent {
                        page_number: None,
                        markdown: md,
                    });
                }
            }
        } else {
            i += 1;
        }
    }
    tables
}

fn looks_like_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() < 10 { return false; }
    let gap_count = trimmed.split("  ").count();
    let pipe_count = trimmed.split('|').count();
    gap_count >= 3 || pipe_count >= 3
}

fn count_columns(line: &str) -> usize {
    let trimmed = line.trim();
    if trimmed.contains('|') {
        trimmed.split('|').filter(|s| !s.trim().is_empty()).count()
    } else {
        trimmed.split("  ").filter(|s| !s.trim().is_empty()).count()
    }
}

fn convert_to_markdown_table(lines: &[&str], max_cols: usize) -> Option<String> {
    let rows: Vec<Vec<String>> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if trimmed.contains('|') {
                trimmed.split('|').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<Vec<String>>()
            } else {
                trimmed.split("   ").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<Vec<String>>()
            }
        })
        .filter(|r| !r.is_empty())
        .collect();

    if rows.len() < 2 { return None; }

    let mut normalized = Vec::new();
    for row in rows {
        let mut r = row;
        while r.len() < max_cols { r.push(String::new()); }
        r.truncate(max_cols);
        normalized.push(r);
    }

    let mut md = String::new();
    md.push_str(&normalized[0].join(" | "));
    md.push('\n');
    md.push_str(&normalized[0].iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
    md.push('\n');
    for row in &normalized[1..] {
        md.push_str(&row.join(" | "));
        md.push('\n');
    }
    Some(md)
}

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let effective_chunk_size = chunk_size.max(overlap + 1);
    let step = effective_chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + effective_chunk_size).min(words.len());
        let chunk_words = &words[start..end];
        let chunk_text = chunk_words.join(" ");
        if !chunk_text.trim().is_empty() {
            chunks.push(chunk_text);
        }
        if end == words.len() {
            break;
        }
        start += step;
    }
    chunks
}

pub fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}