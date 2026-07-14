# rag-zhipu

RAG Zhipu adalah aplikasi Rust CLI untuk melakukan tanya-jawab atas dokumen dengan pendekatan Retrieval-Augmented Generation (RAG). Proyek ini dirancang untuk bekerja secara lokal dan ringan, sehingga cocok untuk mesin dengan sumber daya terbatas.

## Fitur Utama

- Mendukung dokumen berformat PDF, DOCX, TXT, MD, dan HTML
- Memecah teks menjadi chunk dengan overlap untuk retrieval yang lebih baik
- Menggunakan embedding TF-IDF berbasis Rust yang ringan dan tidak bergantung pada model besar
- Menyediakan mode ringan `--lite` untuk menghindari download model lokal besar
- Mendukung pencarian chunk relevan menggunakan cosine similarity
- Dapat dipakai sebagai dasar untuk pengembangan RAG yang lebih luas ke database vektor di masa depan

## Arsitektur Singkat

Alur kerja proyek ini adalah:

1. Ekstraksi teks dari dokumen
2. Pembagian teks menjadi chunk
3. Pembuatan embedding untuk setiap chunk
4. Penyimpanan chunk dan metadata ke vector store in-memory
5. Pencarian chunk paling relevan berdasarkan pertanyaan pengguna
6. Penyusunan konteks dan jawaban dari hasil retrieval

## Struktur Proyek

```text
rag-zhipu/
├── Cargo.toml
├── README.md
└── src/
    ├── document_reader.rs   # ekstraksi teks dari dokumen
    ├── local_embedder.rs    # embedding TF-IDF berbasis Rust
    ├── local_llm.rs         # dukungan model lokal (opsional)
    ├── main.rs              # entry point CLI
    ├── vector_store.rs      # vector store in-memory
    └── zhipu_client.rs      # client API Z.AI (opsional/ekperimental)
```

## Prasyarat

Pastikan Anda telah menginstal:

- Rust dan Cargo (disarankan melalui rustup)
- Windows/Linux/macOS dengan dukungan toolchain Rust yang sehat

## Instalasi

Clone repository ini lalu build proyek:

```bash
git clone https://github.com/your-username/rag-zhipu.git
cd rag-zhipu
cargo build --release
```

## Penggunaan

### Mode default

```bash
cargo run --release -- --pdf "path/ke/dokumen.pdf" --question "Apa isi dokumen ini secara garis besar?"
```

### Mode ringan untuk mesin terbatas

```bash
cargo run --release -- --pdf "path/ke/dokumen.pdf" --question "Apa poin penting dari dokumen ini?" --lite
```

### Contoh dengan pengaturan manual

```bash
cargo run --release -- \
  --pdf dokumen.pdf \
  --question "Sebutkan poin-poin penting bab 2" \
  --chunk-size 1000 \
  --overlap 200 \
  --top-k 5 \
  --max-features 128
```

## Opsi CLI

| Flag | Default | Keterangan |
|------|---------|------------|
| `--pdf` | wajib | Jalur file dokumen yang akan diproses |
| `--question` | wajib | Pertanyaan yang ingin dijawab |
| `--chunk-size` | 800 | Ukuran tiap chunk teks (karakter) |
| `--overlap` | 150 | Overlap antar chunk |
| `--top-k` | 4 | Jumlah chunk relevan yang diambil |
| `--max-features` | 256 | Jumlah fitur TF-IDF maksimum untuk menghemat memori |
| `--lite` | false | Menjalankan mode ringan tanpa model lokal besar |
| `--ocr` | false | Mode OCR untuk PDF scan (masih memerlukan integrasi tambahan) |

## Catatan Penting

- Mode `--lite` sangat disarankan jika Anda menggunakan komputer dengan RAM atau CPU terbatas.
- Model lokal yang dipakai oleh pipeline LLM bersifat opsional dan dapat memerlukan unduhan pertama kali yang cukup besar.
- Untuk dokumen scan/PDF gambar, dukungan OCR belum sepenuhnya dioptimalkan dan memerlukan pengembangan lebih lanjut.

## Roadmap

Beberapa pengembangan yang sedang atau dapat dipertimbangkan:

- Integrasi dengan database vektor seperti Qdrant atau Milvus
- Dukungan chunking yang lebih cerdas berdasarkan paragraf dan struktur dokumen
- Optimasi performa untuk dokumen besar
- Integrasi OCR yang lebih matang untuk PDF hasil scan

## Kontribusi

Kontribusi sangat diterima. Jika Anda ingin memperbaiki fitur, menambah dukungan format dokumen, atau mengoptimalkan performa, silakan buat pull request atau buka issue.
