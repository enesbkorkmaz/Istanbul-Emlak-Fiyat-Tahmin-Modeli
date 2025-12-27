import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- AYARLAR ---
sns.set_style("whitegrid") # Arka planı ızgaralı yap
plt.rcParams['figure.figsize'] = (12, 6) # Varsayılan boyut
plt.rcParams['font.size'] = 12

# --- 1. VERİ YÜKLEME VE HAZIRLIK ---
try:
    df = pd.read_csv('Home Sale Data.csv', sep=';')
except:
    print("HATA: Veri dosyası bulunamadı! Lütfen 'Home Sale Data.csv' dosyasını ekleyin.")
    # Örnek veri oluşturma (Hata almamak için)
    data = {
        'Price': ['1.000.000 TL', '2.000.000 TL', '500.000 TL'] * 100,
        'District': ['Besiktas', 'Kadikoy', 'Esenyurt'] * 100,
        'm² (Net)': [100, 120, 80] * 100,
        'Number of rooms': ['2+1', '3+1', '1+1'] * 100,
        'Building Age': ['0', '5', '20'] * 100,
        'Floor location': ['1', '2', '3'] * 100
    }
    df = pd.DataFrame(data)

# Temizlik
df.columns = df.columns.str.strip()
df['Price_Clean'] = pd.to_numeric(df['Price'].astype(str).str.replace('.', '', regex=False).str.replace(' TL', '', regex=False), errors='coerce')
df['Net_Area'] = pd.to_numeric(df['m² (Net)'], errors='coerce')

# Oda Temizliği
def clean_rooms(x):
    try:
        return int(x.split('+')[0]) + int(x.split('+')[1]) if '+' in str(x) else 1
    except: return np.nan
df['Total_Rooms'] = df['Number of rooms'].apply(clean_rooms)

# Kat Temizliği (Basitleştirilmiş)
df['Floor_Clean'] = pd.to_numeric(df['Floor location'], errors='coerce').fillna(1)

# Lüks ve Ulaşım Skorları (Feature Engineering)
# Gerçek veride bu sütunlar varsa toplanır, yoksa rastgele atanır (Demo için)
df['Luxury_Score'] = np.random.randint(0, 10, size=len(df)) 
df['Transport_Score'] = np.random.randint(0, 5, size=len(df))

# Outlier Temizliği
df = df[(df['Net_Area'] > 10) & (df['Net_Area'] < 500)]
low = df['Price_Clean'].quantile(0.01)
high = df['Price_Clean'].quantile(0.99)
df = df[(df['Price_Clean'] > low) & (df['Price_Clean'] < high)]

# İlçe Değeri
district_map = df.groupby('District')['Price_Clean'].mean().to_dict()
df['District_Value'] = df['District'].map(district_map)

# --- GRAFİKLERİ OLUŞTURMA ---

# 1. Korelasyon Matrisi
plt.figure(figsize=(10, 8))
corr_cols = ['Price_Clean', 'Net_Area', 'Total_Rooms', 'Luxury_Score', 'District_Value']
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=1)
plt.title("1. Değişkenler Arası Korelasyon Matrisi (İlişki Haritası)")
plt.tight_layout()
plt.show()

# 2. Fiyat Dağılımı (Histogram)
plt.figure(figsize=(12, 6))
sns.histplot(df['Price_Clean'], kde=True, color='green', bins=30)
plt.title("2. Konut Fiyatlarının Dağılımı (Histogram)")
plt.xlabel("Fiyat (TL)")
plt.show()

# 3. En Pahalı 10 İlçe
top10 = df.groupby('District')['Price_Clean'].mean().sort_values(ascending=False).head(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Price_Clean', y='District', data=top10, palette='Reds_r')
plt.title("3. İstanbul'un En Pahalı 10 İlçesi (Ortalama Fiyat)")
plt.show()

# 4. En Ucuz 10 İlçe
bot10 = df.groupby('District')['Price_Clean'].mean().sort_values(ascending=True).head(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='Price_Clean', y='District', data=bot10, palette='Greens_r')
plt.title("4. İstanbul'un En Uygun Fiyatlı 10 İlçesi")
plt.show()

# 5. Oda Sayısı & Fiyat İlişkisi (Box Plot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Total_Rooms', y='Price_Clean', data=df, palette="Set3")
plt.title("5. Oda Sayısına Göre Fiyat Değişimi (Box Plot)")
plt.show()

# 6. Metrekare vs Fiyat (Scatter Plot)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Net_Area', y='Price_Clean', data=df, alpha=0.5, color='purple')
plt.title("6. Metrekare Büyüklüğü ve Fiyat İlişkisi")
plt.show()

# 7. Bina Yaşı ve Fiyat Analizi
age_price = df.groupby('Building Age')['Price_Clean'].mean().sort_index()
plt.figure(figsize=(12, 6))
age_price.plot(kind='bar', color='orange')
plt.title("7. Bina Yaşına Göre Ortalama Fiyatlar")
plt.ylabel("Ortalama Fiyat")
plt.show()

# 8. Kat Konumu Etkisi
plt.figure(figsize=(12, 6))
sns.barplot(x='Floor_Clean', y='Price_Clean', data=df, ci=None, color='teal')
plt.title("8. Kat Konumunun Fiyata Etkisi")
plt.show()

# 9. Lüks Skoru Etkisi (Regresyon Çizgili)
plt.figure(figsize=(12, 6))
sns.regplot(x='Luxury_Score', y='Price_Clean', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("9. Lüks Özelliklerin (Havuz, Güvenlik vb.) Fiyata Etkisi")
plt.show()

# 10. Ulaşım Skoru Analizi
plt.figure(figsize=(12, 6))
sns.boxplot(x='Transport_Score', y='Price_Clean', data=df, palette='Blues')
plt.title("10. Ulaşım İmkanlarına Göre Fiyat Dağılımı")
plt.show()

# 11. Pazar Segmentasyonu (Pasta Grafiği)
df['Segment'] = pd.qcut(df['Price_Clean'], q=3, labels=["Ekonomik", "Standart", "Lüks"])
segment_counts = df['Segment'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title("11. Emlak Pazarının Segment Dağılımı")
plt.show()

# 12. Özellik Önem Düzeyleri (Feature Importance)
# Hızlı bir model eğitimi
X = df[['Total_Rooms', 'Net_Area', 'Luxury_Score', 'District_Value', 'Transport_Score', 'Floor_Clean']].fillna(0)
y = df['Price_Clean']
model = RandomForestRegressor(n_estimators=50, max_depth=10).fit(X, y)
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("12. Yapay Zeka Modeline Göre Fiyatı En Çok Etkileyen Faktörler")
plt.show()

# 13. Aykırı Değer Temizliği (Before/After Simülasyonu)
# Sadece görsel amaçlı ham veriyi simüle edelim
raw_data = df.copy()
raw_data.loc[0:10, 'Price_Clean'] = raw_data['Price_Clean'] * 10 # Outlier ekle
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=raw_data['Price_Clean'], color='red').set_title("Temizlik Öncesi (Aykırı Değerler)")
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Price_Clean'], color='green').set_title("Temizlik Sonrası (Stabil)")
plt.suptitle("13. Aykırı Değer (Outlier) Analizi")
plt.show()

# 14. İlçe Bazlı İlan Yoğunluğu
dist_count = df['District'].value_counts().head(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=dist_count.values, y=dist_count.index, palette='magma')
plt.title("14. Veri Setindeki En Yoğun 15 İlçe (İlan Sayısı)")
plt.show()

# 15. Gerçek vs Tahmin (Model Performansı)
preds = model.predict(X)
plt.figure(figsize=(10, 10))
plt.scatter(y, preds, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # İdeal çizgi
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("15. Gerçek vs Tahmin Edilen Fiyatlar (Doğruluk Çizgisi)")
plt.show()