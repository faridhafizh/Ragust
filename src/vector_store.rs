use serde::{Deserialize, Serialize};

/// Satu potongan dokumen beserta embedding dan metadata sumber.
#[derive(Clone, Serialize, Deserialize)]
pub struct DocChunk {
    pub text: String,
    pub embedding: Vec<f32>,
    /// File sumber chunk ini (mis: "document.pdf").
    pub source_file: String,
    /// Hash konten sumber saat chunk ini dibuat.
    pub source_hash: String,
    /// Nomor halaman (jika tersedia).
    pub page_number: Option<usize>,
    /// Index chunk global dalam dokumen.
    pub chunk_index: usize,
}

#[derive(Serialize, Deserialize)]
pub struct VectorStore {
    pub chunks: Vec<DocChunk>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Tambahkan chunk dengan metadata lengkap.
    pub fn add(
        &mut self,
        text: String,
        embedding: Vec<f32>,
        source_file: &str,
        source_hash: &str,
        page_number: Option<usize>,
        chunk_index: usize,
    ) {
        self.chunks.push(DocChunk {
            text,
            embedding,
            source_file: source_file.to_string(),
            source_hash: source_hash.to_string(),
            page_number,
            chunk_index,
        });
    }

    pub fn save_to_file(&self, path: &str) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let store = serde_json::from_reader(file)?;
        Ok(store)
    }

    /// Cari top-k chunk paling relevan berdasarkan cosine similarity.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&DocChunk, f32)> {
        if top_k == 0 || self.chunks.is_empty() {
            return Vec::new();
        }

        let mut best: Vec<(&DocChunk, f32)> = Vec::with_capacity(top_k.min(self.chunks.len()));

        for chunk in &self.chunks {
            let score = cosine_similarity(&chunk.embedding, query_embedding);
            if best.len() < top_k {
                best.push((chunk, score));
                best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else if let Some(last) = best.last() {
                if score > last.1 {
                    best.pop();
                    best.push((chunk, score));
                    best.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
        }

        best
    }

    /// Hapus semua chunk dari file sumber tertentu.
    pub fn remove_by_source_file(&mut self, source_file: &str) {
        self.chunks.retain(|c| c.source_file != source_file);
    }

    /// Cek apakah cache sudah berisi chunk untuk file+hash ini.
    pub fn has_cached_source(&self, source_file: &str, source_hash: &str) -> bool {
        self.chunks
            .iter()
            .any(|c| c.source_file == source_file && c.source_hash == source_hash)
    }

    /// Hitung jumlah chunk untuk file sumber tertentu.
    pub fn count_by_source(&self, source_file: &str) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.source_file == source_file)
            .count()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
