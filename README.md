#  İstanbul Konut Piyasası Tahmin ve Analiz Sistemi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

Bu proje, İstanbul emlak piyasasındaki konut fiyatlarını tahmin etmek ve gayrimenkul değerlemesi yapmak amacıyla geliştirilmiş **hibrit yapılı bir makine öğrenmesi uygulamasıdır.**

Uygulama, geçmiş verilerden öğrenen yapay zeka modellerini (Random Forest), güncel ekonomik koşullara uyarlayan dinamik simülasyon araçlarıyla birleştirir.

## Projenin Özellikleri

* **Hibrit Modelleme:** Makine öğrenmesi (Random Forest) çıktıları, alan bilgisi (Domain Knowledge) kuralları ile desteklenir. (Örn: Havuzlu siteler için manuel katsayı düzeltmesi).
* **Dinamik Enflasyon Simülasyonu:** Model 2020 verileriyle eğitilmiştir, ancak kullanıcıya sunulan **"Piyasa Projeksiyon Katsayısı"** ile 2025 ve sonrası için senaryo bazlı tahminler üretilebilir.
* **CBS (Coğrafi Bilgi Sistemi) Desteği:** İlçe bazlı fiyat yoğunlukları interaktif Folium haritası üzerinde görselleştirilmiştir.
* **Gelişmiş Veri Analizi:** IQR yöntemi ile Outlier (Aykırı Değer) temizliği yapılmış, lüks ve ulaşım skorları gibi öznitelik mühendisliği teknikleri uygulanmıştır.
* **Segmentasyon:** Evin sadece fiyatını değil; "Ekonomik", "Standart" veya "Lüks" sınıfında olup olmadığını da tahmin eder.

## Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Gerekli kütüphaneleri yükleyin:**

3.  **Uygulamayı başlatın:**
    ```bash
    streamlit run app.py
    ```

Projede **Linear Regression** ve **Random Forest** algoritmaları kıyaslanmıştır.
---
*Not: Bu proje akademik/eğitim amaçlı geliştirilmiştir. Finansal yatırım tavsiyesi içermez.*
