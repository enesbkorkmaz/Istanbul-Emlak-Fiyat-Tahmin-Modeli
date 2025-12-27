#KÜTÜPHANELER
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="İstanbul Konut Piyasasında Fiyat Tahmini ve Değer Analizi", #Sayfa başlığı
    page_icon="🏠", #Logo
    layout="wide"
)

# --- BAŞLIK ---
st.title("🏠İstanbul Konut Piyasasında Fiyat Tahmini ve Değer Analizi")
st.markdown("""
Bu proje, İstanbul emlak piyasasındaki konut özellikleri ve fiyat verilerini analiz ederek eğitilmiş kapsamlı bir yapay zeka modelidir. Seçilen özelliklere dayalı olarak detaylı bir fiyat tahmin analizi sunmaktadır.
""")

# --- FONKSİYONLAR ---

@st.cache_data #Veriyi bir kerelik çektik bu sayede her sayfa yenilediğinde tekrar veriyi çekmekle uğraşmayacak
def load_data_raw():
    try:
        df = pd.read_csv('Home Sale Data.csv', sep=';')
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def clean_and_process_data(df): #Verisetini temizliyoruz
    df = df.copy()
    df.columns = df.columns.str.strip()
    
    # Fiyat
    df['Price_Clean'] = df['Price'].astype(str).str.replace('.', '', regex=False).str.replace(' TL', '', regex=False)
    df['Price_Clean'] = pd.to_numeric(df['Price_Clean'], errors='coerce')
    
    # Oda
    def clean_rooms(room_str):
        try:
            if '+' in str(room_str):
                parts = room_str.split('+')
                return int(parts[0]) + int(parts[1])
            elif 'Stüdyo' in str(room_str):
                return 1
            return float('nan')
        except:
            return float('nan')
    df['Total_Rooms'] = df['Number of rooms'].apply(clean_rooms)
    
    # Kat
    floor_mapping = {
        'Giriş Katı': 0, 'Ground floor': 0, 'Bahçe Katı': 0, 'Garden Floor': 0,
        'Yüksek Giriş': 0.5, 'High entrance': 0.5, 'Kot 1': -1, 'Kot 2': -2,
        'Bodrum': -1, 'Basement': -1, 'Çatı Katı': 100, 'Teras Katı': 100 #kontrol et
    }
    def clean_floor(floor_str):
        if str(floor_str).replace('.', '', 1).isdigit():
            return float(floor_str)
        elif floor_str in floor_mapping:
            return floor_mapping[floor_str]
        try:
            import re
            num = re.search(r'-?\d+', str(floor_str))
            return float(num.group()) if num else np.nan
        except:
            return np.nan
    df['Floor_Clean'] = df['Floor location'].apply(clean_floor)
    
    # Diğer
    age_mapping = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5-10 between': 7, 
        '11-15 between': 13, '16-20 between': 18, '21-25 between': 23,
        '26-30 between': 28, '31 and more than': 35
    }
    df['Age_Clean'] = df['Building Age'].map(age_mapping).fillna(15)
    df['Net_Area'] = pd.to_numeric(df['m² (Net)'], errors='coerce')
    df['Neighborhood'] = df['Neighborhood'].str.strip()

    # NaN Temizliği
    df.dropna(subset=['Price_Clean', 'Total_Rooms', 'Floor_Clean', 'Net_Area', 'District', 'Neighborhood'], inplace=True)

    # Outlier Temizliği (IQR Mantığı)
    df = df[(df['Net_Area'] > 10) & (df['Net_Area'] < 500)]
    low_limit = df['Price_Clean'].quantile(0.01)
    high_limit = df['Price_Clean'].quantile(0.99)
    df = df[(df['Price_Clean'] > low_limit) & (df['Price_Clean'] < high_limit)]
    
    # Sınıflandırma Etiketi (Segment)
    df['Price_Segment'] = pd.qcut(df['Price_Clean'], q=3, labels=["Ekonomik", "Standart", "Lüks"])
    
    return df

@st.cache_data
def feature_engineering(df): #Burada lüks skoru, ulaşım skoru, bölge değeri oluşturduk. Modelin tahmin doğruluğunu artıracak.
    luxury_cols = [
        'Smart House', 'Sauna', 'Turkish Hamam', 'Jacuzzi', 
        'With Private Pool', 'Swimming Pool (Open)', 'Swimming Pool (Indoor)',
        'Parking Lot', 'Closed Garage', 'Security', 'Video intercom'
    ]
    available_luxury = [col for col in luxury_cols if col in df.columns]
    df['Luxury_Score'] = df[available_luxury].sum(axis=1)
    
    transport_cols = [
        'Metro', 'Metrobus', 'Marmaray', 'Tram', 'Cable car', 
        'Bus stop', 'E-5', 'TEM', 'Airport', 'Ferry', 'Sea bus'
    ]
    available_transport = [col for col in transport_cols if col in df.columns]
    df['Transport_Score'] = df[available_transport].sum(axis=1)
    
    district_map = df.groupby('District')['Price_Clean'].mean().to_dict()
    df['District_Value'] = df['District'].map(district_map)
    
    neighborhood_map = df.groupby('Neighborhood')['Price_Clean'].mean().to_dict()
    df['Neighborhood_Value'] = df['Neighborhood'].map(neighborhood_map)
    df['Neighborhood_Value'].fillna(df['District_Value'], inplace=True)
    
    return df, district_map, neighborhood_map

@st.cache_resource
def train_full_suite(df): #MAKİNE ÖĞRENMESİ KISMI
    features = ['Total_Rooms', 'Floor_Clean', 'Net_Area', 'Age_Clean', 'Luxury_Score', 'Transport_Score', 'District_Value', 'Neighborhood_Value']
    
    # Veri Hazırlığı
    X = df[features]
    y_reg = df['Price_Clean']
    y_class = df['Price_Segment']
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)   #Eğitim seti %80 test seti %20 olacak şekilde böler random state aynı şekilde bölünmesi için
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # 1. MODEL KARŞILAŞTIRMA (REGRESYON)
    # A) Linear Regression Kısmı
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train_reg)
    y_pred_lin = lin_model.predict(X_test)
    r2_lin = r2_score(y_test_reg, y_pred_lin)
    
    # B) Random Forest Kısmı
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42) #Maks derinlik 15, Karar ağacı sayısı 100, random state de her çalıştırmada aynı sonucu vermesi için
    rf_model.fit(X_train, y_train_reg) #tahmin kısmı
    y_pred_rf = rf_model.predict(X_test) #Eğitilmiş model tahmin yapar
    
    # Regresyon Metrikleri R^2 skoru vb.
    reg_metrics = {
        'R2_RF': r2_score(y_test_reg, y_pred_rf),
        'R2_Lin': r2_lin,
        'MAE': mean_absolute_error(y_test_reg, y_pred_rf),
        'MSE': mean_squared_error(y_test_reg, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
    }
    
    # 2. SINIFLANDIRMA (SEGMENT TAHMİNİ)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train_c, y_train_c)
    y_pred_c = clf_model.predict(X_test_c)
    
    # Confusion Matrix
    labels = ["Ekonomik", "Standart", "Lüks"]
    cm = confusion_matrix(y_test_c, y_pred_c, labels=labels)
    
    # Sınıflandırma Metrikleri
    class_metrics = {
        'Accuracy': accuracy_score(y_test_c, y_pred_c),
        'Precision': precision_score(y_test_c, y_pred_c, average='weighted'),
        'Recall': recall_score(y_test_c, y_pred_c, average='weighted'),
        'F1': f1_score(y_test_c, y_pred_c, average='weighted')
    }
    
    return rf_model, clf_model, reg_metrics, class_metrics, cm, labels, features

def show_advanced_map(df): #Haritalama kısmı. Github'dan bulduğum json dosyası burada işleniyor interaktif harita haline geliyor.
    district_stats = df.groupby('District').agg({
        'Price_Clean': 'mean',
        'Net_Area': 'mean',
        'District': 'count'
    }).rename(columns={'District': 'Ilan_Sayisi', 'Price_Clean': 'Ortalama_Fiyat'}).reset_index()
    
    file_path = "istanbul-geojson-master/ilce_geojson.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            geo_json_data = json.load(f)
        for feature in geo_json_data['features']:
            if 'display_name' in feature['properties']:
                feature['properties']['name'] = feature['properties']['display_name'].split(',')[0].strip()
    except:
        return None

    # Harita yüklendiğinde bu konumdan(İstanbul) gösterilecek
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10, tiles="CartoDB positron")

    # Bölgelerin renklendirilmesi
    cp = folium.Choropleth(
        geo_data=geo_json_data,
        name='Bölgesel Fiyatlar',
        data=district_stats,
        columns=['District', 'Ortalama_Fiyat'],
        key_on='feature.properties.name',
        fill_color='YlOrRd',
        fill_opacity=0.6,
        line_opacity=0.2,
        legend_name='Ortalama Fiyat (TL)',
        highlight=True
    ).add_to(m)

    # Mouse ile bölge üzerine gelindiğinde detay verecek
    district_dict = district_stats.set_index('District').to_dict('index')
    for feature in geo_json_data['features']:
        d_name = feature['properties']['name']
        if d_name in district_dict:
            feature['properties']['fiyat'] = f"{district_dict[d_name]['Ortalama_Fiyat']:,.0f} TL"
            feature['properties']['adet'] = str(district_dict[d_name]['Ilan_Sayisi'])
            feature['properties']['m2'] = f"{district_dict[d_name]['Net_Area']:.0f} m²"
        else:
            feature['properties']['fiyat'] = "Veri Yok"
            feature['properties']['adet'] = "0"
            feature['properties']['m2'] = "-"

    folium.GeoJsonTooltip(
        fields=['name', 'fiyat', 'adet', 'm2'],
        aliases=['İlçe:', 'Ort. Fiyat:', 'İlan Sayısı:', 'Ort. Büyüklük:'],
        localize=True,
        sticky=False,
        labels=True,
        style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
    ).add_to(cp.geojson)

    return m

# --- UYGULAMA AKIŞI ---

with st.spinner('Sistem başlatılıyor...'):
    raw_df = load_data_raw()
    if not raw_df.empty:
        df = clean_and_process_data(raw_df)
        df, district_map, neighborhood_map = feature_engineering(df)
        rf_model, clf_model, reg_metrics, class_metrics, cm, labels, feature_names = train_full_suite(df)
    else:
        st.stop()

# --- SOL MENÜ ---
st.sidebar.header("🔎 Simülasyon Ayarları")
user_district = st.sidebar.selectbox("İlçe", sorted(df['District'].unique()))
valid_neighborhoods = df[df['District'] == user_district]['Neighborhood'].unique()
user_neighborhood = st.sidebar.selectbox("Mahalle", sorted(valid_neighborhoods))
user_m2 = st.sidebar.number_input("m² (Net)", 10, 500, 100)
user_rooms = st.sidebar.selectbox("Oda", ["1+1", "2+1", "3+1", "4+1", "5+1"])
user_age = st.sidebar.slider("Bina Yaşı", 0, 40, 5)
user_floor = st.sidebar.number_input("Kat", -2, 40, 3)
st.sidebar.markdown("---")
has_pool = st.sidebar.checkbox("Havuz var mı?")
has_parking = st.sidebar.checkbox("Otopark var mı?")
is_near_metro = st.sidebar.checkbox("Metroya yakın mı?")
st.sidebar.markdown("---")
inflation_multiplier = st.sidebar.slider("2025 Çarpanı (x)", 15.0, 25.0, 15.0, 0.5)
predict_btn = st.sidebar.button("FİYAT TAHMİN ET", type="primary")

# --- ANA EKRAN SEKMELERİ ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Veri & Outlier", 
    "📈 Görsel Analiz (EDA)", 
    "🗺️ Coğrafi Analiz", 
    "🏆 Model & Metrikler", 
    "🔮 Tahmin"
])

# 1. SEKME: VERİ & OUTLIER
with tab1:
    st.header("Veri Ön İşleme ve Aykırı Değer (Outlier) Analizi")
    st.markdown("**IQR Yöntemi** ile aykırı değer temizliği yapıldı.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Ham Veri Sayısı", f"{len(raw_df):,}")
    col2.metric("Temizlik Sonrası Veri Sayısı", f"{len(df):,}")
    col3.metric("Temizlenen Veri Sayısı (Aykırı Değer,Boş Değer Vb.)", f"{len(raw_df) - len(df):,}", delta_color="inverse")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Ham Veri (İşlenmemiş)")
        st.caption("Fiyatlar string formatında, oda sayıları karmaşık.")
        st.dataframe(raw_df[['District', 'Price', 'Number of rooms', 'Floor location']].head())
        
    with col_b:
        st.subheader("2. İşlenmiş Veri (Temiz)")
        st.caption("Fiyatlar sayısal hale getirildi, özellikler modelin anlayacağı şekle getirildi.")
        st.dataframe(df[['District', 'Price_Clean', 'Total_Rooms', 'Floor_Clean']].head())
    st.markdown("---")
    
    st.subheader("📊 Aykırı Değer Analizi (Öncesi vs Sonrası)")
    raw_plot = raw_df.copy()
    raw_plot['Price'] = pd.to_numeric(raw_plot['Price'].astype(str).str.replace('.','',regex=False).str.replace(' TL','',regex=False), errors='coerce')
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        fig_raw = px.box(raw_plot, y="Price", title="Temizlik Öncesi (Outliers Var)")
        st.plotly_chart(fig_raw, use_container_width=True)
    with col_plot2:
        fig_clean = px.box(df, y="Price_Clean", title="Temizlik Sonrası (Stabil)")
        st.plotly_chart(fig_clean, use_container_width=True)

# 2. SEKME: DETAYLI GRAFİKLER
with tab2:
    st.header("Keşifçi Veri Analizi (EDA)")
    
    st.subheader("Korelasyon Matrisi (Heatmap)")
    corr_cols = ['Price_Clean', 'Net_Area', 'Total_Rooms', 'Age_Clean', 'Luxury_Score', 'Transport_Score', 'District_Value']
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
    fig_corr.update_traces(textfont_size=14)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.divider()
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("Oda Sayısı - Fiyat İlişkisi")
        fig2 = px.box(df, x='Total_Rooms', y='Price_Clean', color='Total_Rooms', title="Oda Sayısı ve Fiyat")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_y:
        st.subheader("En Pahalı 10 İlçe")
        top_districts = df.groupby('District')['Price_Clean'].mean().sort_values(ascending=False).head(10).reset_index()
        fig1 = px.bar(top_districts, x='District', y='Price_Clean', title="İlçe Ortalamaları")
        st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Lüks Skorunun Fiyata Etkisi")
    avg_luxury = df.groupby('Luxury_Score')['Price_Clean'].mean().reset_index()
    fig3 = px.line(avg_luxury, x='Luxury_Score', y='Price_Clean', markers=True, title="Lüks Donanım ve Fiyat karşılaştırması")
    st.plotly_chart(fig3, use_container_width=True)

# 3. SEKME: COĞRAFİ ANALİZ
with tab3:
    st.header("Coğrafi Analiz")
    st.markdown("Harita üzerinde **İlçe Ortalamaları** gösterilmektedir. Detaylar için farenizi ilçelerin üzerine getirin.")
    map_obj = show_advanced_map(df)
    if map_obj:
        st_folium(map_obj, use_container_width=True, height=700)

# 4. SEKME: MODEL & METRİKLER (DETAYLI LİSTE)
with tab4:
    st.header("Model Benchmark ve Başarı Metrikleri")
    
    st.info("Random Forest modeli, Linear Regression modeline göre veriyi çok daha iyi açıklamaktadır.")
    c_bm1, c_bm2 = st.columns(2)
    c_bm1.metric("Linear Reg. (R²)", f"%{reg_metrics['R2_Lin']*100:.2f}")
    c_bm2.metric("Random Forest (R²)", f"%{reg_metrics['R2_RF']*100:.2f}", delta="Daha yüksek başarı oranına sahip olduğu için tahminlerde bu modeli kullandık.")
    
    st.divider()
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📉 Regresyon Başarısı (Fiyat)")
        st.markdown("Fiyat tahmin modelinin hata oranları:")
        st.markdown(f"""
        * **R² Skoru:** %{reg_metrics['R2_RF']*100:.2f}
        * **MAE (Ort. Mutlak Hata):** {reg_metrics['MAE']:,.0f} TL
        * **MSE (Hata Kareleri Ort.):** {reg_metrics['MSE']:,.0f}
        * **RMSE (Kök Ortalama Hata):** {reg_metrics['RMSE']:,.0f}
        """)
        
    with col_r:
        st.subheader("📊 Sınıflandırma Başarısı (Segment)")
        st.markdown("Evi 'Ekonomik/Standart/Lüks' olarak ayırma başarısı:")
        st.markdown(f"""
        * **Accuracy (Doğruluk):** %{class_metrics['Accuracy']*100:.2f}
        * **Precision (Hassasiyet):** %{class_metrics['Precision']*100:.2f}
        * **Recall (Duyarlılık):** %{class_metrics['Recall']*100:.2f}
        * **F1 Skoru:** %{class_metrics['F1']*100:.2f}
        """)

    st.divider()
    st.subheader("Hata Matrisi (Confusion Matrix)")
    fig_cm = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Tahmin Edilen", y="Gerçek Sınıf"),
                       x=labels, y=labels, color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)

# 5. SEKME: TAHMİN
with tab5:
    st.header("Fiyat ve Segment Tahmini")
    if predict_btn:
        u_room = int(user_rooms.split('+')[0]) + int(user_rooms.split('+')[1])
        u_luxury = int(has_pool) + int(has_parking)
        u_transport = int(is_near_metro)
        u_district_val = district_map.get(user_district, df['District_Value'].mean())
        u_neighborhood_val = neighborhood_map.get(user_neighborhood, u_district_val)
        
        input_data = pd.DataFrame([[u_room, user_floor, user_m2, user_age, u_luxury, u_transport, u_district_val, u_neighborhood_val]], columns=feature_names)
        
        # Mantık Düzeltmeli Regresyon Tahmini
        raw_pred = rf_model.predict(input_data)[0]
        if has_pool: raw_pred *= 1.10
        if has_parking: raw_pred *= 1.05
        if is_near_metro: raw_pred *= 1.05
        
        pred_2025 = raw_pred * inflation_multiplier
        segment_pred = clf_model.predict(input_data)[0]
        
        st.success(f"📍 Konum: **{user_district} / {user_neighborhood}**")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Segment", segment_pred)
            st.caption("Evin Sınıfı")
        with c2:
            st.metric("2020 Tahmini", f"{raw_pred:,.0f} TL")
            st.caption("Model Çıktısı")
        with c3:
            st.metric("2025 Tahmini", f"{pred_2025:,.0f} TL", delta=f"{inflation_multiplier}x çarpan ile")
            st.warning("NOT: Uyarlanmış tahmindir. Model bu fiyatı tahmin etmedi.", icon="⚠️")
    else:
        st.info("👈 Tahmin butonuna basınız.")