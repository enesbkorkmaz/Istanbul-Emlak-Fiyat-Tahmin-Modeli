# 🏠 İstanbul Konut Piyasası Tahmin ve Analiz Sistemi

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

Bu proje, İstanbul emlak piyasasındaki konut fiyatlarını tahmin etmek ve gayrimenkul değerlemesi yapmak amacıyla geliştirilmiş **hibrit yapılı bir makine öğrenmesi uygulamasıdır.**

Uygulama, geçmiş verilerden öğrenen yapay zeka modellerini (Random Forest), güncel ekonomik koşullara uyarlayan dinamik simülasyon araçlarıyla birleştirir.

## 🚀 Projenin Özellikleri

* **🤖 Hibrit Modelleme:** Makine öğrenmesi (Random Forest) çıktıları, alan bilgisi (Domain Knowledge) kuralları ile desteklenir. (Örn: Havuzlu siteler için manuel katsayı düzeltmesi).
* **📈 Dinamik Enflasyon Simülasyonu:** Model 2020 verileriyle eğitilmiştir, ancak kullanıcıya sunulan **"Piyasa Projeksiyon Katsayısı"** ile 2025 ve sonrası için senaryo bazlı tahminler üretilebilir.
* **🗺️ CBS (Coğrafi Bilgi Sistemi) Desteği:** İlçe bazlı fiyat yoğunlukları interaktif Folium haritası üzerinde görselleştirilmiştir.
* **🔍 Gelişmiş Veri Analizi:** IQR yöntemi ile Outlier (Aykırı Değer) temizliği yapılmış, lüks ve ulaşım skorları gibi öznitelik mühendisliği teknikleri uygulanmıştır.
* **🏆 Segmentasyon:** Evin sadece fiyatını değil; "Ekonomik", "Standart" veya "Lüks" sınıfında olup olmadığını da tahmin eder.

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1.  **Repoyu klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git)
    cd PROJE_ADINIZ
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı başlatın:**
    ```bash
    streamlit run app.py
    ```

## 📊 Metodoloji ve Performans

Projede **Linear Regression** ve **Random Forest** algoritmaları kıyaslanmıştır.

| Model | R² Skoru (Başarı) | Durum |
| :--- | :---: | :--- |
| **Random Forest** | **%85.4** | ✅ Seçilen Model |
| Linear Regression | %62.1 | ❌ Yetersiz |

* **Veri Temizliği:** IQR yöntemi ile uç değerler temizlenmiştir.
* **Feature Importance:** Modelin fiyatı belirlerken en çok **Net Metrekare** ve **İlçe Konum Değerine** dikkat ettiği tespit edilmiştir.

## 🖥️ Ekran Görüntüleri

*(Buraya uygulamanın ekran görüntülerini ekleyebilirsiniz)*

| Tahmin Ekranı | Harita Analizi |
| :---: | :---: |
| ![Tahmin](screenshots/tahmin.png) | ![Harita](screenshots/map.png) |

## 📂 Dosya Yapısı

* `app.py`: Ana Streamlit uygulama dosyası.
* `Home Sale Data.csv`: Kullanılan veri seti.
* `istanbul-geojson-master/`: Harita görselleştirmesi için gerekli JSON verileri.
* `requirements.txt`: Gerekli Python kütüphaneleri.

## 👨‍💻 Hazırlayan

**[Adınız Soyadınız]**
* [LinkedIn Profiliniz](https://linkedin.com/in/kullaniciadi)
* [GitHub Profiliniz](https://github.com/kullaniciadi)

---
*Not: Bu proje akademik/eğitim amaçlı geliştirilmiştir. Finansal yatırım tavsiyesi içermez.*
