use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub struct TfIdfEmbedder {
    idf: HashMap<String, f32>,
    vocab: HashMap<String, usize>,
    dim: usize,
    max_features: usize,
}

impl TfIdfEmbedder {
    pub fn with_max_features(max_features: usize) -> Self {
        Self {
            idf: HashMap::new(),
            vocab: HashMap::new(),
            dim: 0,
            max_features: max_features.max(1),
        }
    }

    pub fn fit<'a, I>(&mut self, documents: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        let docs: Vec<&'a str> = documents.into_iter().collect();
        let n_docs = docs.len() as f32;
        if n_docs == 0.0 {
            self.dim = 0;
            self.idf.clear();
            self.vocab.clear();
            return;
        }

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut all_terms: HashSet<String> = HashSet::new();

        for doc in &docs {
            let terms = tokenize(doc);
            let unique_terms: HashSet<String> = terms.into_iter().collect();
            for term in unique_terms {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
                all_terms.insert(term);
            }
        }

        let mut term_stats: Vec<(String, f32)> = all_terms
            .iter()
            .map(|term| {
                let df = *doc_freq.get(term).unwrap_or(&1) as f32;
                let idf_val = (n_docs / df).ln() + 1.0;
                (term.clone(), idf_val)
            })
            .collect();

        term_stats.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut vocab_list: Vec<(String, f32)> =
            term_stats.into_iter().take(self.max_features).collect();
        
        vocab_list.sort_by(|a, b| a.0.cmp(&b.0));

        self.vocab.clear();
        self.idf.clear();
        for (idx, (term, idf_val)) in vocab_list.iter().enumerate() {
            self.idf.insert(term.clone(), *idf_val);
            self.vocab.insert(term.clone(), idx);
        }
        self.dim = vocab_list.len();
        
        println!(
            "   📚 Vocabulary size: {} terms (capped at {})",
            self.dim, self.max_features
        );
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let terms = tokenize(text);
        let mut tf: HashMap<String, f32> = HashMap::new();

        for term in terms {
            *tf.entry(term).or_insert(0.0) += 1.0;
        }

        for (term, count) in tf {
            if let Some(&idx) = self.vocab.get(&term) {
                let idf_val = self.idf.get(&term).copied().unwrap_or(1.0);
                vec[idx] = count * idf_val;
            }
        }
        l2_normalize(&mut vec);
        vec
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let mut result = Vec::with_capacity(total);
        for (i, text) in texts.iter().enumerate() {
            print!("      🔄 Chunk {}/{} ... ", i + 1, total);
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let emb = self.embed(text);
            result.push(emb);
            println!("✅");
        }
        Ok(result)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|s| !s.is_empty()) // Filter panjang karakter dihapus agar kata "di", "ke", "dan" tidak terbuang
        .collect()
}

fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}