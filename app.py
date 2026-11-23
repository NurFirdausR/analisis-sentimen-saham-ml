import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈")

st.title("📈 Sistem Analisis Sentimen Pasar Saham")
st.write("Klasifikasi sentimen berita & katalis emiten (**Bullish / Positif** vs **Bearish / Negatif**) menggunakan **TF-IDF** dan **Multinomial Naive Bayes**.")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("dataset_saham.csv", sep=';')
 
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["berita"], df["sentimen"], test_size=0.2, random_state=42
    )
    
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    akurasi = accuracy_score(y_test, y_pred) * 100
    return df, vectorizer, model, akurasi

df, vectorizer, model, akurasi = load_and_train()

with st.sidebar:
    st.header("ℹ️ Informasi Model Finansial")
    st.metric(label="Akurasi Data Uji", value=f"{akurasi:.1f}%")
    st.write(f"**Total Dataset:** {len(df)} berita emiten")
    st.write("**Algoritma:** TF-IDF + Multinomial Naive Bayes")
    if st.checkbox("Lihat Sampel Dataset Saham"):
        st.dataframe(df.head(10))

st.subheader("Uji Coba Analisis Berita / Katalis Saham")
input_teks = st.text_area(
    "Masukkan potongan berita atau laporan kinerja emiten:",
    placeholder="Contoh: Emiten mencatatkan lonjakan laba bersih 50 persen dan membagikan dividen tunai."
)

if st.button("Analisis Sentimen Saham"):
    if input_teks.strip() == "":
        st.warning("⚠️ Silakan masukkan teks berita emiten terlebih dahulu.")
    else:
        teks_tfidf = vectorizer.transform([input_teks])
        prediksi = model.predict(teks_tfidf)[0]
        probabilitas = model.predict_proba(teks_tfidf)[0]
        skor_keyakinan = max(probabilitas) * 100
        
        st.divider()
        st.subheader("Hasil Analisis:")
        if prediksi == "Positif":
            st.success(f"**Sentimen: POSITIF / BULLISH** (Tingkat Keyakinan: **{skor_keyakinan:.2f}%**)")
        else:
            st.error(f"**Sentimen: NEGATIF / BEARISH** (Tingkat Keyakinan: **{skor_keyakinan:.2f}%**)")
            
        st.write("**Rincian Probabilitas:**")
        for k, p in zip(model.classes_, probabilitas):
            label_tampil = "Positif (Bullish)" if k == "Positif" else "Negatif (Bearish)"
            st.progress(float(p), text=f"{label_tampil}: {p*100:.2f}%")