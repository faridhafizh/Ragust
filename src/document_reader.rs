use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

const WHITESPACE_SEPARATOR: &str = " ";

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Content extracted from any supported document.
pub struct DocumentContent {
    /// Full concatenated text of the document.
    pub full_text: String,
    /// Per-page content (for PDFs; single entry for other formats).
    pub pages: Vec<PageContent>,
    /// Tables extracted as markdown.
    pub tables: Vec<TableContent>,
    /// SHA256 hash of the raw document content.
    pub source_hash: String,
}

pub struct PageContent {
    pub page_number: usize,
    pub text: String,
    pub hash: String,
}

pub struct TableContent {
    pub page_number: Option<usize>,
    pub markdown: String,
}

// ============================================================================
// UNIFIED DOCUMENT EXTRACTION
// ============================================================================

/// Extract content from any supported file based on extension.
/// Supported: .pdf, .docx, .txt, .md, .html, .htm
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

// ============================================================================
// PDF EXTRACTION (text + OCR-ready helpers)
// ============================================================================

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

/// Extract text from PDF using pdf_extract crate.
pub fn extract_text_from_pdf(path: &str) -> Result<String> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("Gagal mengekstrak teks dari PDF: {}", path))?;
    Ok(text)
}

/// Convert PDF pages to PNG images using `pdftoppm` (poppler-utils).
/// Returns a Vec of raw PNG bytes per page.
/// Requires `pdftoppm` installed on the system.
pub fn pdf_pages_to_images(path: &str) -> Result<Vec<Vec<u8>>> {
    let temp_dir = std::env::temp_dir();
    let prefix = format!("rag_ocr_{}", std::process::id());
    let out_prefix = temp_dir.join(&prefix).to_string_lossy().to_string();

    let status = std::process::Command::new("pdftoppm")
        .args(["-png", "-r", "150", path, &out_prefix])
        .status()
        .map_err(|e| {
            anyhow!(
                "Gagal menjalankan pdftoppm: {}. \
             Install poppler-utils (Linux: apt install poppler-utils, \
             Mac: brew install poppler, Windows: download poppler binaries).",
                e
            )
        })?;

    if !status.success() {
        return Err(anyhow!("pdftoppm exited with non-zero status"));
    }

    // Collect generated PNG files
    let mut entries: Vec<_> = std::fs::read_dir(&temp_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with(&prefix) && name.ends_with(".png")
        })
        .collect();

    entries.sort_by_key(|e| e.file_name());

    let mut images = Vec::new();
    for entry in entries {
        images.push(std::fs::read(entry.path())?);
        let _ = std::fs::remove_file(entry.path());
    }

    if images.is_empty() {
        return Err(anyhow!(
            "Tidak ada gambar yang dihasilkan dari PDF. Periksa apakah PDF valid."
        ));
    }

    Ok(images)
}

// ============================================================================
// DOCX EXTRACTION (pure Rust via ZIP + XML parsing)
// ============================================================================

fn extract_docx(path: &str) -> Result<DocumentContent> {
    let text = read_docx_text(path)?;
    let tables = Vec::new(); // TODO: extend to extract docx tables from word/document.xml
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

/// Read .docx by extracting word/document.xml from the ZIP archive
/// and pulling text from <w:t> tags.
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
        return Err(anyhow!(
            "word/document.xml tidak ditemukan dalam file .docx"
        ));
    }

    // Simple extraction of <w:t> tags
    let mut text = String::new();
    let mut pos = 0;
    while let Some(start) = xml_content[pos..].find("<w:t") {
        let tag_start = pos + start;
        let close_bracket = xml_content[tag_start..]
            .find('>')
            .ok_or_else(|| anyhow!("XML .docx tidak valid: tidak ada >"))?
            + tag_start;
        let end_tag = xml_content[close_bracket..]
            .find("</w:t>")
            .ok_or_else(|| anyhow!("XML .docx tidak valid: tidak ada </w:t>"))?
            + close_bracket;
        let content = &xml_content[close_bracket + 1..end_tag];
        text.push_str(content);

        // Check if next tag is a break/tab
        if let Some(next) = xml_content[end_tag + 6..].find('<') {
            let snippet = &xml_content[end_tag + 6 + next..end_tag + 6 + next + 15];
            if snippet.contains("<w:br") || snippet.contains("<w:tab") {
                text.push('\n');
            }
        }

        pos = end_tag + 6;
    }

    Ok(text)
}

// ============================================================================
// PLAIN TEXT & HTML EXTRACTION
// ============================================================================

fn extract_plain_text(path: &str) -> Result<DocumentContent> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("Gagal membaca file: {}", path))?;
    let source_hash = compute_hash(&text);

    Ok(DocumentContent {
        full_text: text.clone(),
        pages: Vec::new(),
        tables: Vec::new(),
        source_hash,
    })
}

fn extract_html(path: &str) -> Result<DocumentContent> {
    let html = std::fs::read_to_string(path)
        .with_context(|| format!("Gagal membaca file HTML: {}", path))?;
    let text = strip_html_tags(&html);
    let tables = extract_html_tables(&html);
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

/// Strip HTML tags and script/style blocks. No external crate needed.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let lower = html.to_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();

    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            // Detect script/style start
            if i + 7 < lower_chars.len()
                && lower_chars[i + 1..].starts_with(&['s', 'c', 'r', 'i', 'p', 't'])
            {
                in_script = true;
            } else if i + 6 < lower_chars.len()
                && lower_chars[i + 1..].starts_with(&['s', 't', 'y', 'l', 'e'])
            {
                in_script = true;
            } else if i + 8 < lower_chars.len()
                && lower_chars[i + 1..].starts_with(&['/', 's', 'c', 'r', 'i', 'p', 't'])
            {
                in_script = false;
            } else if i + 7 < lower_chars.len()
                && lower_chars[i + 1..].starts_with(&['/', 's', 't', 'y', 'l', 'e'])
            {
                in_script = false;
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if chars[i] == '>' {
            in_tag = false;
            i += 1;
            continue;
        }
        if !in_tag && !in_script {
            result.push(chars[i]);
        }
        i += 1;
    }

    // Normalize whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract HTML tables and convert to markdown.
fn extract_html_tables(html: &str) -> Vec<TableContent> {
    let mut tables = Vec::new();
    let lower = html.to_lowercase();
    let mut start = 0;

    while let Some(table_start) = lower[start..].find("<table") {
        let abs_start = start + table_start;
        if let Some(table_end) = lower[abs_start..].find("</table>") {
            let abs_end = abs_start + table_end + 8;
            let table_html = &html[abs_start..abs_end];
            if let Some(md) = html_table_to_markdown(table_html) {
                tables.push(TableContent {
                    page_number: None,
                    markdown: md,
                });
            }
            start = abs_end;
        } else {
            break;
        }
    }

    tables
}

fn html_table_to_markdown(table_html: &str) -> Option<String> {
    let lower = table_html.to_lowercase();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut pos = 0;

    while let Some(tr_start) = lower[pos..].find("<tr") {
        let tr_abs = pos + tr_start;
        let tr_end = lower[tr_abs..].find("</tr>")? + tr_abs + 5;
        let tr_slice = &table_html[tr_abs..tr_end];

        let mut cells = Vec::new();
        for tag in ["<td", "<th"] {
            let mut cell_pos = 0;
            let tr_lower = tr_slice.to_lowercase();
            while let Some(td_start) = tr_lower[cell_pos..].find(tag) {
                let td_abs = cell_pos + td_start;
                let td_close = tr_slice[td_abs..].find('>')? + td_abs + 1;
                let td_end =
                    tr_lower[td_abs..].find(if tag == "<td" { "</td>" } else { "</th>" })? + td_abs;
                let cell_text = strip_html_tags(&tr_slice[td_close..td_end]);
                cells.push(cell_text.trim().to_string());
                cell_pos = td_end + 5;
            }
        }

        if !cells.is_empty() {
            rows.push(cells);
        }
        pos = tr_end;
    }

    if rows.len() < 2 {
        return None;
    }

    let mut md = String::new();
    md.push_str(&rows[0].join(" | "));
    md.push('\n');
    md.push_str(
        &rows[0]
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | "),
    );
    md.push('\n');
    for row in &rows[1..] {
        md.push_str(&row.join(" | "));
        md.push('\n');
    }
    Some(md)
}

// ============================================================================
// TABLE EXTRACTION FROM PLAIN TEXT (HEURISTIC)
// ============================================================================

/// Detect table-like structures in plain text using multi-space column gaps
/// or pipe separators, then convert to markdown tables.
fn extract_tables_from_text(text: &str) -> Vec<TableContent> {
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
    if trimmed.is_empty() || trimmed.len() < 10 {
        return false;
    }
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
                trimmed
                    .split('|')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<String>>()
            } else {
                trimmed
                    .split("  ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<String>>()
            }
        })
        .filter(|r| !r.is_empty())
        .collect();

    if rows.len() < 2 {
        return None;
    }

    // Normalize column count
    let mut normalized = Vec::new();
    for row in rows {
        let mut r = row;
        while r.len() < max_cols {
            r.push(String::new());
        }
        r.truncate(max_cols);
        normalized.push(r);
    }

    let mut md = String::new();
    md.push_str(&normalized[0].join(" | "));
    md.push('\n');
    md.push_str(
        &normalized[0]
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | "),
    );
    md.push('\n');
    for row in &normalized[1..] {
        md.push_str(&row.join(" | "));
        md.push('\n');
    }
    Some(md)
}

// ============================================================================
// TEXT CHUNKING (existing, unchanged)
// ============================================================================

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let effective_chunk_size = chunk_size.max(overlap + 1);
    let cleaned: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(WHITESPACE_SEPARATOR);
    let chars: Vec<char> = cleaned.chars().collect();
    let mut chunks = Vec::new();

    if chars.is_empty() {
        return chunks;
    }

    let step = effective_chunk_size.saturating_sub(overlap).max(1);
    let mut start = 0;

    while start < chars.len() {
        let end = (start + effective_chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        let trimmed = chunk.trim().to_string();

        if !trimmed.is_empty() {
            chunks.push(trimmed);
        }

        if end == chars.len() {
            break;
        }
        start += step;
    }

    chunks
}

// ============================================================================
// UTILITIES
// ============================================================================

/// Compute SHA256 hash of content for incremental indexing.
pub fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}
