"""
SISTEM CERDAS PENENTU KEBIJAKAN PENDIDIKAN
Metode: TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pymysql

# ==============================================================
# KONFIGURASI HALAMAN
# ==============================================================
st.set_page_config(
    page_title="SPK Penentuan Prioritas MBG & Pembangunan Fasilitas Pendidikan",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# DEFINISI KRITERIA & BOBOT
# ==============================================================
KRITERIA = ["C1", "C2", "C3", "C4"]
BOBOT    = [0.15, 0.30, 0.40, 0.15]          # Total = 1.0
SIFAT    = ["benefit", "cost", "benefit", "cost"]  # benefit=max, cost=min

LABEL_KRITERIA = {
    "C1": "Jml. Anak Usia Sekolah",
    "C2": "Rata-rata Pendapatan (Rp)",
    "C3": "Jml. Anak Tidak Sekolah",
    "C4": "Jml. Sekolah Ada",
}

# ==============================================================
# KONEKSI DATABASE (di-cache agar tidak reconnect tiap refresh)
# ==============================================================
@st.cache_resource
def get_connection():
    try:
        conn = pymysql.connect(
            host     = st.secrets["mysql"]["host"],
            port     = int(st.secrets.get("mysql", {}).get("port", 3306)),
            user     = st.secrets["mysql"]["user"],
            password = st.secrets["mysql"]["password"],
            database = st.secrets["mysql"]["database"],
            cursorclass = pymysql.cursors.DictCursor,
            autocommit  = True,
        )
        return conn
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke database: {e}")
        st.info(
            "Pastikan file `.streamlit/secrets.toml` sudah diisi dengan benar.\n\n"
            "Contoh isi secrets.toml:\n"
            "```toml\n"
            "[mysql]\n"
            'host     = "localhost"\n'
            'port     = 3306\n'
            'user     = "root"\n'
            'password = "password_anda"\n'
            'database = "spk_pendidikan"\n'
            "```"
        )
        return None


def get_conn():
    """Helper: ambil koneksi & pastikan masih hidup (ping reconnect)."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        conn.ping(reconnect=True)
    except Exception:
        # Hapus cache dan coba ulang sekali
        get_connection.clear()
        conn = get_connection()
    return conn


def init_table():
    """Buat tabel jika belum ada."""
    conn = get_conn()
    if conn is None:
        return
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS data_kecamatan (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                nama_kecamatan  VARCHAR(100) NOT NULL,
                c1              FLOAT NOT NULL,
                c2              FLOAT NOT NULL,
                c3              FLOAT NOT NULL,
                c4              FLOAT NOT NULL
            )
        """)


# ==============================================================
# FUNGSI TOPSIS
# ==============================================================
def hitung_topsis(df: pd.DataFrame) -> dict:

    hasil = {}

    # --- Matriks Keputusan ---
    X = df[["c1", "c2", "c3", "c4"]].values.astype(float)
    hasil["matriks_keputusan"] = pd.DataFrame(
        X,
        columns=KRITERIA,
        index=df["nama_kecamatan"].values,
    )

    # --- Normalisasi Vector ---
    # Rumus: r_ij = x_ij / sqrt(sum(x_ij^2))
    norm_pembagi = np.sqrt((X ** 2).sum(axis=0))   # akar jumlah kuadrat per kolom
    R = X / norm_pembagi
    hasil["matriks_normalisasi"] = pd.DataFrame(
        R,
        columns=KRITERIA,
        index=df["nama_kecamatan"].values,
    )

    # --- LANGKAH 3: Matriks Ternormalisasi Terbobot ---
    # Rumus: y_ij = w_j * r_ij
    bobot_arr = np.array(BOBOT)
    Y = R * bobot_arr
    hasil["matriks_terbobot"] = pd.DataFrame(
        Y,
        columns=KRITERIA,
        index=df["nama_kecamatan"].values,
    )

    # --- LANGKAH 4: Solusi Ideal Positif (A+) dan Negatif (A-) ---
    # Benefit → A+ = max, A- = min
    # Cost    → A+ = min, A- = max
    A_plus  = []
    A_minus = []
    for j, sifat in enumerate(SIFAT):
        if sifat == "benefit":
            A_plus.append(Y[:, j].max())
            A_minus.append(Y[:, j].min())
        else:  # cost
            A_plus.append(Y[:, j].min())
            A_minus.append(Y[:, j].max())

    A_plus  = np.array(A_plus)
    A_minus = np.array(A_minus)
    hasil["A_plus"]  = pd.DataFrame([A_plus],  columns=KRITERIA, index=["A+"])
    hasil["A_minus"] = pd.DataFrame([A_minus], columns=KRITERIA, index=["A-"])

    # --- LANGKAH 5: Jarak ke Solusi Ideal ---
    # D+_i = sqrt( sum( (y_ij - A+_j)^2 ) )
    # D-_i = sqrt( sum( (y_ij - A-_j)^2 ) )
    D_plus  = np.sqrt(((Y - A_plus)  ** 2).sum(axis=1))
    D_minus = np.sqrt(((Y - A_minus) ** 2).sum(axis=1))
    hasil["jarak"] = pd.DataFrame(
        {"D+ (Jarak ke A+)": D_plus, "D- (Jarak ke A-)": D_minus},
        index=df["nama_kecamatan"].values,
    )

    # --- LANGKAH 6: Nilai Preferensi (V) ---
    # Rumus: V_i = D-_i / (D-_i + D+_i)
    # Semakin tinggi V → semakin dekat ke A+ → peringkat lebih baik (butuh perhatian lebih)
    V = D_minus / (D_minus + D_plus)
    hasil["nilai_preferensi"] = V
    hasil["nama_kecamatan"]   = df["nama_kecamatan"].values

    return hasil


def buat_rekomendasi(df_rank: pd.DataFrame) -> pd.DataFrame:
    n = len(df_rank)
    rekomendasi = []

    for peringkat in df_rank["Peringkat"]:
        if peringkat == 1:
            rekomendasi.append(
                "🚨 Krisis Infrastruktur: Prioritas Utama Pembangunan Sekolah Baru."
            )
        elif peringkat == n:
            rekomendasi.append(
                "✅ Infrastruktur Memadai: Mulai Pemberian Makan Bergizi Gratis (MBG)."
            )
        else:
            rekomendasi.append(
                "🔄 Evaluasi Berkala: Lakukan perbaikan fasilitas yang sudah ada."
            )

    df_rank = df_rank.copy()
    df_rank["Rekomendasi Tindakan"] = rekomendasi
    return df_rank


# ==============================================================
# CUSTOM CSS
# ==============================================================
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a5c 0%, #0d2137 100%);
    }
    [data-testid="stSidebar"] * { color: #e8f4fd !important; }

    /* Judul utama */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a3a5c;
        border-bottom: 3px solid #e8a020;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Badge langkah TOPSIS */
    .step-badge {
        background: #1a3a5c;
        color: white;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
    }

    /* Kartu metrik */
    [data-testid="stMetric"] {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 12px;
        border-left: 4px solid #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# SIDEBAR NAVIGASI
# ==============================================================
with st.sidebar:
    st.markdown("## 🏫 SPK Pendidikan NTT")
    st.markdown("---")
    menu = st.selectbox(
        "Pilih Menu",
        ["📥 Input Data", "📋 Data Alternatif", "📊 Analisis TOPSIS & Rekomendasi"],
    )
    st.markdown("---")
    st.markdown("**Bobot Kriteria:**")
    for k, w, s in zip(KRITERIA, BOBOT, SIFAT):
        tipe = "🔼 Benefit" if s == "benefit" else "🔽 Cost"
        st.markdown(f"- **{k}**: {w} ({tipe})")
    st.markdown("---")
    st.caption("Metode: TOPSIS | Penentuan Skala Prioritas")

# Inisialisasi tabel saat app pertama jalan
init_table()


# ==============================================================
# MENU 1: INPUT DATA
# ==============================================================
if menu == "📥 Input Data":
    st.markdown('<div class="main-title">📥 Input Data Kecamatan</div>', unsafe_allow_html=True)

    with st.form("form_input", clear_on_submit=True):
        nama = st.text_input("Nama Kecamatan", placeholder="Contoh: Kec. Kupang Tengah")

        col1, col2 = st.columns(2)
        with col1:
            c1 = st.number_input(
                "C1 – Jumlah Anak Usia Sekolah (jiwa)",
                min_value=0, step=1,
                help="Benefit: semakin tinggi → kebutuhan lebih besar",
            )
            c3 = st.number_input(
                "C3 – Jumlah Anak Tidak Sekolah (jiwa)",
                min_value=0, step=1,
                help="Benefit: semakin tinggi → urgensi pembangunan sekolah lebih tinggi",
            )
        with col2:
            c2 = st.number_input(
                "C2 – Rata-rata Pendapatan Keluarga/Bulan (Rp)",
                min_value=0, step=50000,
                help="Cost: pendapatan rendah → kebutuhan bantuan lebih besar",
            )
            c4 = st.number_input(
                "C4 – Jumlah Sekolah yang Sudah Ada",
                min_value=0, step=1,
                help="Cost: semakin sedikit sekolah → kebutuhan lebih besar",
            )

        submitted = st.form_submit_button("💾 Simpan Data", use_container_width=True)

    if submitted:
        if not nama.strip():
            st.warning("⚠️ Nama kecamatan tidak boleh kosong.")
        else:
            conn = get_conn()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO data_kecamatan (nama_kecamatan, c1, c2, c3, c4) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (nama.strip(), c1, c2, c3, c4),
                        )
                    st.success(f"✅ Data **{nama}** berhasil disimpan ke database!")
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan: {e}")


# ==============================================================
# MENU 2: DATA ALTERNATIF
# ==============================================================
elif menu == "📋 Data Alternatif":
    st.markdown('<div class="main-title">📋 Data Alternatif Kecamatan</div>', unsafe_allow_html=True)

    conn = get_conn()
    if conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM data_kecamatan ORDER BY id")
            rows = cur.fetchall()

        if rows:
            df_view = pd.DataFrame(rows)
            df_view = df_view.rename(columns={
                "id": "ID",
                "nama_kecamatan": "Nama Kecamatan",
                "c1": "C1 (Anak Usia Sekolah)",
                "c2": "C2 (Pendapatan Rp)",
                "c3": "C3 (Anak Tidak Sekolah)",
                "c4": "C4 (Jml Sekolah)",
            })

            st.metric("Total Kecamatan Terdaftar", len(df_view))
            st.dataframe(df_view, use_container_width=True, hide_index=True)

            st.markdown("---")
            with st.expander("🗑️ Hapus Semua Data"):
                st.warning("Aksi ini akan menghapus **seluruh** data dari tabel secara permanen!")
                if st.button("⚠️ Kosongkan Tabel", type="secondary"):
                    with conn.cursor() as cur:
                        cur.execute("TRUNCATE TABLE data_kecamatan")
                    st.success("✅ Tabel berhasil dikosongkan.")
                    st.rerun()
        else:
            st.info("Belum ada data. Silakan input data terlebih dahulu melalui menu **Input Data**.")


# ==============================================================
# MENU 3: ANALISIS TOPSIS & REKOMENDASI
# ==============================================================
elif menu == "📊 Analisis TOPSIS & Rekomendasi":
    st.markdown('<div class="main-title">📊 Analisis TOPSIS & Rekomendasi Kebijakan</div>', unsafe_allow_html=True)

    conn = get_conn()
    if conn is None:
        st.stop()

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM data_kecamatan ORDER BY id")
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        st.warning("⚠️ Minimal 2 data kecamatan diperlukan untuk menjalankan TOPSIS.")
        st.stop()

    df = pd.DataFrame(rows)

    # ---- STEP 1: Matriks Keputusan ----
    st.markdown('<span class="step-badge">LANGKAH 1 · Matriks Keputusan (X)</span>', unsafe_allow_html=True)
    st.caption("Data mentah dari database yang menjadi input perhitungan TOPSIS.")
    hasil = hitung_topsis(df)
    st.dataframe(hasil["matriks_keputusan"].style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")

    # ---- STEP 2: Normalisasi ----
    st.markdown('<span class="step-badge">LANGKAH 2 · Matriks Ternormalisasi (R)</span>', unsafe_allow_html=True)
    st.caption("Normalisasi vector: r_ij = x_ij / √Σ(x_ij²) — menyeragamkan skala antar kriteria.")
    st.dataframe(hasil["matriks_normalisasi"].style.format("{:.6f}"), use_container_width=True)

    st.markdown("---")

    # ---- STEP 3: Terbobot ----
    st.markdown('<span class="step-badge">LANGKAH 3 · Matriks Ternormalisasi Terbobot (Y)</span>', unsafe_allow_html=True)
    st.caption(f"y_ij = w_j × r_ij — Bobot: {dict(zip(KRITERIA, BOBOT))}")
    st.dataframe(hasil["matriks_terbobot"].style.format("{:.6f}"), use_container_width=True)

    st.markdown("---")

    # ---- STEP 4: Solusi Ideal ----
    st.markdown('<span class="step-badge">LANGKAH 4 · Solusi Ideal Positif (A+) & Negatif (A-)</span>', unsafe_allow_html=True)
    st.caption("Benefit → A+ = max, A- = min | Cost → A+ = min, A- = max")
    col_ap, col_am = st.columns(2)
    with col_ap:
        st.write("**A+ (Solusi Ideal Positif)**")
        st.dataframe(hasil["A_plus"].style.format("{:.6f}"), use_container_width=True)
    with col_am:
        st.write("**A- (Solusi Ideal Negatif)**")
        st.dataframe(hasil["A_minus"].style.format("{:.6f}"), use_container_width=True)

    st.markdown("---")

    # ---- STEP 5: Jarak ----
    st.markdown('<span class="step-badge">LANGKAH 5 · Jarak ke Solusi Ideal (D+, D-)</span>', unsafe_allow_html=True)
    st.caption("D+_i = √Σ(y_ij − A+_j)²  |  D-_i = √Σ(y_ij − A-_j)²")
    st.dataframe(hasil["jarak"].style.format("{:.6f}"), use_container_width=True)

    st.markdown("---")

    # ---- STEP 6: Nilai Preferensi & Ranking ----
    st.markdown('<span class="step-badge">LANGKAH 6 · Nilai Preferensi (V) & Peringkat Akhir</span>', unsafe_allow_html=True)
    st.caption("V_i = D-_i / (D-_i + D+_i)  — Semakin tinggi V → semakin mendesak kebutuhan intervensi.")

    V     = hasil["nilai_preferensi"]
    nama  = hasil["nama_kecamatan"]

    df_rank = pd.DataFrame({
        "Kecamatan"          : nama,
        "Nilai Preferensi (V)": V,
    })
    df_rank = df_rank.sort_values("Nilai Preferensi (V)", ascending=False).reset_index(drop=True)
    df_rank["Peringkat"] = df_rank.index + 1

    # Tambahkan kolom Rekomendasi
    df_rank = buat_rekomendasi(df_rank)

    # Format tampilan
    df_display = df_rank[["Peringkat", "Kecamatan", "Nilai Preferensi (V)", "Rekomendasi Tindakan"]].copy()
    df_display["Nilai Preferensi (V)"] = df_display["Nilai Preferensi (V)"].map("{:.6f}".format)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Peringkat"           : st.column_config.NumberColumn("🏆 Peringkat", width="small"),
            "Kecamatan"           : st.column_config.TextColumn("Kecamatan", width="medium"),
            "Nilai Preferensi (V)": st.column_config.TextColumn("Nilai V", width="small"),
            "Rekomendasi Tindakan": st.column_config.TextColumn("Rekomendasi Tindakan", width="large"),
        },
    )

    st.markdown("---")

    # ---- VISUALISASI GRAFIK BATANG ----
    st.markdown('<span class="step-badge">VISUALISASI · Grafik Nilai Preferensi (V)</span>', unsafe_allow_html=True)
    st.caption("Kecamatan dengan nilai V lebih tinggi memiliki urgensi intervensi infrastruktur yang lebih besar.")

    df_chart = df_rank.set_index("Kecamatan")[["Nilai Preferensi (V)"]].sort_values(
        "Nilai Preferensi (V)", ascending=False
    )
    st.bar_chart(df_chart, use_container_width=True, color="#1a3a5c")

    # ---- RINGKASAN EKSEKUTIF ----
    st.markdown("---")
    st.subheader("📌 Ringkasan Eksekutif")
    top    = df_rank.iloc[0]
    bottom = df_rank.iloc[-1]

    col_top, col_bot = st.columns(2)
    with col_top:
        st.error(
            f"**Prioritas Utama: {top['Kecamatan']}**\n\n"
            f"Nilai V = {top['Nilai Preferensi (V)']:.4f}\n\n"
            f"{top['Rekomendasi Tindakan']}"
        )
    with col_bot:
        st.success(
            f"**✅ Kondisi Terbaik: {bottom['Kecamatan']}**\n\n"
            f"Nilai V = {bottom['Nilai Preferensi (V)']:.4f}\n\n"
            f"{bottom['Rekomendasi Tindakan']}"
        )