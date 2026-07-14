use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct DocChunk {
    pub text: String,
    pub embedding: Vec<f32>,
    pub source_file: String,
    pub source_hash: String,
    pub page_number: Option<usize>,
    pub chunk_index: usize,
}

#[derive(Serialize, Deserialize)]
pub struct VectorStore {
    pub chunks: Vec<DocChunk>,
}

#[allow(dead_code)]
impl VectorStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

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

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(&DocChunk, f32)> {
        if top_k == 0 || self.chunks.is_empty() {
            return Vec::new();
        }

        let mut scored_chunks: Vec<(&DocChunk, f32)> = self.chunks.iter()
            .map(|chunk| (chunk, cosine_similarity(&chunk.embedding, query_embedding)))
            .collect();

        // Urutkan dari skor tertinggi ke terendah
        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        scored_chunks.into_iter().take(top_k).collect()
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