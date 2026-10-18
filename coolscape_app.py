"""
CoolScape AI – v5 prototip (Siemens Türkiye Climathon)
Optimizasyon tabanlı karar destek: kısıtlı soğutma suyunu saat saat ısı stresine göre dağıtır.

Ne gerçek, ne temsili:
- Hava verisi GERÇEK: Open-Meteo. Kayıtlı günler için geçmiş arşiv (ERA5 tabanlı), "Yarın" için tahmin API'si.
- Isı yüzeyi GERÇEK (varsa): Landsat 8/9 yüzey sıcaklığı (+ varsa Sentinel-2 NDVI), H3 altıgenler
  (veri_hazirla.py ile bir kez üretilen isi_yuzeyi_h3.csv, repoda durur; uygulama kendisi okur).
- Hassas tesisler GERÇEK: OpenStreetMap (repodaki tesisler_osm.json/csv; yoksa Overpass API'den canlı).
- Nüfus GERÇEK (varsa): WorldPop 2025 toplam nüfusu (nufus_hazirla.py ile üretilen nufus_h3.csv): yerleşim maskesi,
  park nüfus skoru, erişim nüfusu. WorldPop 65+ katmanı mahalle düzeyinde değildir: toplam nüfusu ilçe başına tek
  bir yaş oranıyla böler (pilot bölgede yalnızca %6,8 / %9,8 / %11,2). Bu yüzden 65+ sayısı öncelik ölçütü olarak
  "ilçe oranıyla" etiketlenir; kırılganlık göstergesi 65+ payından DEĞİL, OSM okul/sağlık/huzurevi yakınlığından
  (VEKİL) türetilir. Nüfus dosyası yoksa park nüfusu da OSM vekilinden gelir.
- TEMSİLİ: hat kapasitesi, sulanan alan, pompa katsayısı, arıtılmış su erişimi, kişi/saat erişim katsayıları.
- Kıyı: deniz (LST < 36 °C) ve kıyıdaki karışık piksel altıgenleri analiz dışıdır.
- Soğutma eğrisi LİTERATÜR ÖLÇEKLİ: azami etki Paris saha ölçümlerinden (Hendel vd. 2016, ~1 °C UTCI),
  belirsizlik bandı Melbourne/Paris ölçümlerinden (0,5–2 °C), doyma Adelaide çalışmasından (Broadbent vd. 2018).
  Antalya'da yerinde doğrulanmadı; doğrulama 2027 saha pilotunun işi.

Çalıştırma:  pip install -r requirements.txt  &&  python -m streamlit run coolscape_app.py
"""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import pulp
import requests
import streamlit as st
from pythermalcomfort.models import utci

# ---------------------------------------------------------------------------
# SABİTLER VE VARSAYIMLAR
# ---------------------------------------------------------------------------
ANTALYA_LAT, ANTALYA_LON = 36.89, 30.71
SAATLER = list(range(8, 21))  # 08:00–20:00 karar ufku
GUNLER = {
    "25 Temmuz 2025 · kayıtlı sıcak hava dalgası (resmî ölçüm 46,1 °C)": ("arsiv", "2025-07-25"),
    "10 Ağustos 2025 · kayıtlı sıcak gün (MGM: Kepez 45,2 °C)": ("arsiv", "2025-08-10"),
    "Yarın · Open-Meteo tahmini": ("tahmin", None),
}
ISI_YUZEYI_CSV = "isi_yuzeyi_h3.csv"
NUFUS_CSV = "nufus_h3.csv"  # isteğe bağlı: nufus_hazirla.py (WorldPop 2025, 100 m) ile üretilir; varsa kullanılır
SU_ESIGI_C = 36.0  # bu yüzey sıcaklığının altındaki altıgenler deniz/su kabul edilir (Temmuz LST'si iki tepeli:
                   # deniz ~30 °C, kara 44–55 °C); su altıgenleri ısı analizinden çıkarılır
UTCI_ESIK = 32.0          # "güçlü ısı stresi" alt sınırı
OYNATMA_ADIM_S = 1.0      # oynatmada bir saatin ekranda kalma süresi (s)
BBOX = (36.85, 30.62, 36.97, 30.80)  # güney, batı, kuzey, doğu (pilot bölge)

# UTCI ısı stresi sınıfları: eşikler resmî (Bröde vd. 2012); renkler bu uygulamanın seçimidir (YlOrRd tabanlı)
UTCI_SINIFLARI = [
    (-99, 26, "Isı stresi yok", (120, 144, 156)),
    (26, 32, "Orta", (254, 204, 92)),
    (32, 38, "Güçlü", (253, 141, 60)),
    (38, 46, "Çok güçlü", (240, 59, 32)),
    (46, 99, "Aşırı", (189, 0, 38)),
]

# Soğutma eğrisi: saha ölçümlerinden ölçeklenir (kaynaklar uygulamanın "Neden?" bölümünde)
E_REF = 1.0              # °C: sulanan noktada yaya seviyesi UTCI düşüşü, merkezi değer
                         # (Hendel vd. 2016, Paris: −0,93 ve −1,03 °C UTCI)
E_ARALIK = (0.5, 2.0)    # çarpan: 0,5–2,0 °C (Traill vd. 2024: −0,2…−2,0 °C; Parison vd. 2020: −1,9 °C)
VPD_DOYMA = 2.5          # kPa, VARSAYIM: tipik sıcak yaz öğleden sonrası; daha nemli saatlerde buharlaşma
                         # ve dolayısıyla serinlik VPD ile orantılı azalır (Penman tipi buharlaşma ilişkisi)
X0_MM = 0.5              # mm (L/m²) / saat: azalan getiri ölçeği. ~1,5 mm/saatte etkinin %95'i alınır;
                         # Broadbent vd. 2018: günde ~20 L/m² üzeri ek sulama ek serinlik getirmiyor
X0_ARALIK = (0.5, 2.0)   # çarpan: 0,25–1,0 mm duyarlılık
KIRILIM = [0, 2, 4, 6, 8, 12, 16, 24, 32]
MIN_DOZ = 2.0
CEVRE_YARICAP_M = 500   # parkın koruduğu çevre (ısı stresi bu halkadan ölçülür)
YURUME_5DK_M = 400      # kuş uçuşu ~5 dk yürüme
YURUME_10DK_M = 800     # kuş uçuşu ~10 dk yürüme
ETKI_OLCEGI_M = 150     # varsayım: park soğumasının çevreye taşınma ölçeği (e^(−d/150 m))
ETKI_KESME_M = 500      # bu mesafeden sonra etki sıfır kabul edilir

# ---------------------------------------------------------------------------
# YÖNTEMLER: aynı su bütçesi üç farklı soğutma yöntemine dağıtılabilir
# ---------------------------------------------------------------------------
# Park sulama ve yol ıslatma: E = E_REF × min(VPD/2,5; 1) × (1 − e^(−doz/x0)), doz mm cinsinden
# Sisleme: aç/kapa; açıkken E = SISLEME_E × min(VPD/2,5; 1), sabit su tüketimi
SISLEME_E = 4.4          # °C UTCI: Vanos vd. 2022 (Tempe, Arizona, kuru sıcak; güneşte sisli alan ortalaması)
SISLEME_SU_M3_SAAT = 0.12  # VARSAYIM: ~20 nozullu bir istasyon, nozul başına ~0,1 L/dk
SISLEME_ALAN_M2 = 100    # sislenen alan (~10 × 10 m)
ISLATMA_ALAN_M2 = 3000   # VARSAYIM: ~300 m cadde × 10 m kaldırım/yol
ISLATMA_KAPASITE = 5.0   # m³/saat: bir arazözün bir güzergâhta saatte boşaltabileceği (varsayım)
YONTEMLER = {
    # tür: (etiket, harita rengi, yayılım ölçeği m, kesme m)
    "park": ("Park sulama", (30, 136, 229), ETKI_OLCEGI_M, ETKI_KESME_M),
    "sisleme": ("Sisleme noktası", (0, 188, 212), 15, 50),
    "islatma": ("Yol ıslatma", (149, 117, 205), 60, 200),
}
ADAY_SISLEME, ADAY_ISLATMA = 4, 3  # sıcak nokta haritasından otomatik seçilen aday sayısı
ADAY_ARALIK_M = 1000               # adaylar birbirinden ve parklardan en az bu kadar uzak
ADAY_PARK_UZAKLIK_M = (600, 3000)  # yerleşim içinde kalsın: en yakın parka uzaklık aralığı (sezgisel)
ADAY_DOGU_SINIRI = 30.785          # havalimanı ve çevresindeki açık alanları dışarıda bırakır

# Erişim (o saatte yöntemin dokunduğu kişi sayısı) – TEMSİLİ varsayımlar, gelişmiş ayarlardan değişir
ERISIM_VARSAYILAN = {"park_carpan": 50, "sisleme": 40, "islatma": 150}
SEHIR_KIRILGAN_YUZDE = 15          # aday noktalar için temsili kırılgan nüfus oranı

# Öncelik ön ayarları: operatör yalnızca birini seçer, ayrıntı "Gelişmiş ayarlar"da
ON_AYARLAR = {
    "Dengeli": {"kirilgan_pay": 40, "w_k": 0.5},
    "Kırılgan nüfusa öncelik": {"kirilgan_pay": 60, "w_k": 1.5},
    "En çok toplam serinlik": {"kirilgan_pay": 0, "w_k": 0.0},
}

# ---------------------------------------------------------------------------
# PARK VERİSİ – pilot bölge Muratpaşa–Kepez
# Koordinatlar: Google Haritalar. Nüfus, kırılgan grup, hat kapasitesi ve arıtılmış su erişimi TEMSİLİ.
# ---------------------------------------------------------------------------
PARKLAR = pd.DataFrame({
    "park": ["Karaalioğlu Parkı", "Atatürk Parkı", "Düden Park", "Dokuma Park", "Tek Sarnıç Kent Parkı"],
    "ilce": ["Muratpaşa", "Muratpaşa", "Muratpaşa", "Kepez", "Kepez"],
    "lat": [36.87964, 36.88385, 36.85378, 36.91285, 36.90318],
    "lon": [30.70488, 30.68648, 30.78207, 30.67538, 30.63917],
    "yuzey_farki_c": [0.7, -0.3, -2.3, 2.7, 1.5],
    "yesil_skoru": [7, 6, 8, 4, 6],
    "nufus_skoru": [8, 7, 9, 5, 7],
    "hassas_yuzde": [20, 15, 25, 10, 18],
    "pompa_katsayisi": [1.2, 1.1, 1.5, 1.8, 1.3],
    "hat_kapasitesi_m3_saat": [15, 12, 20, 10, 15],
    "sulanan_alan_ha": [1.5, 1.2, 2.0, 1.0, 1.5],
    "aritilmis_su_erisimi": [False, False, True, True, False],
})
PARKLAR["tur"] = "park"


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
HAVA_DEGISKENLERI = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation,direct_normal_irradiance"


def _hava_tablosu(h):
    df = pd.DataFrame({
        "zaman": pd.to_datetime(h["time"]),
        "sicaklik_c": h["temperature_2m"], "bagil_nem": h["relative_humidity_2m"],
        "ruzgar_ms": h["wind_speed_10m"], "gunes_wm2": h["shortwave_radiation"],
    })
    if "direct_normal_irradiance" in h:
        df["dni_wm2"] = h["direct_normal_irradiance"]
    df["saat"] = df["zaman"].dt.hour
    return df


@st.cache_data(show_spinner=False)
def hava_verisi_cek(tarih: str) -> pd.DataFrame:
    params = {"latitude": ANTALYA_LAT, "longitude": ANTALYA_LON, "start_date": tarih, "end_date": tarih,
              "hourly": HAVA_DEGISKENLERI, "wind_speed_unit": "ms", "timezone": "Europe/Istanbul"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
    r.raise_for_status()
    return _hava_tablosu(r.json()["hourly"])


@st.cache_data(show_spinner=False, ttl=3600)
def hava_tahmini_cek(tarih: str) -> pd.DataFrame:
    params = {"latitude": ANTALYA_LAT, "longitude": ANTALYA_LON, "start_date": tarih, "end_date": tarih,
              "hourly": HAVA_DEGISKENLERI, "wind_speed_unit": "ms", "timezone": "Europe/Istanbul"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    return _hava_tablosu(r.json()["hourly"])


def csv_hava_verisi(dosya) -> pd.DataFrame:
    df = pd.read_csv(dosya)
    eksik = {"saat", "sicaklik_c", "bagil_nem", "ruzgar_ms", "gunes_wm2"} - set(df.columns)
    if eksik:
        raise ValueError(f"CSV'de eksik sütunlar: {', '.join(sorted(eksik))}")
    return df


def kiyi_siniflari(df):
    """Su ve kıyı karışık piksel maskesi. Kıyıdaki altıgenlerde 30 m Landsat pikselleri deniz ve karayı karıştırır;
    bu altıgenler yapay olarak serin görünür (gerçek veride kıyı altıgenleri iç kesimden 5–12 °C serin).
    Kural: ≥2 su komşusu olan ya da 1 su komşusu olup iç kesim kara altıgenlerinin 5. yüzdeliğinden serin olan
    altıgenler karışık sayılır ve analiz dışı bırakılır."""
    su = set(df.loc[df["lst_c"] < SU_ESIGI_C, "hex"])
    kara = df[df["lst_c"] >= SU_ESIGI_C]
    komsu = np.array([sum(n in su for n in h3.grid_ring(h, 1)) for h in kara["hex"]])
    ic = kara["lst_c"].values[komsu == 0]
    p5 = float(np.percentile(ic, 5)) if len(ic) else -np.inf
    karisik = (komsu >= 2) | ((komsu == 1) & (kara["lst_c"].values < p5))
    return su, set(kara["hex"].values[karisik])


@st.cache_data(show_spinner=False)
def isi_yuzeyi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "lst_c", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lst_c"])
    su, karisik = kiyi_siniflari(df)
    df = df[~df["hex"].isin(su | karisik)].reset_index(drop=True)
    df["lst_fark_c"] = df["lst_c"] - df["lst_c"].median()
    df.attrs["su_altigen"] = len(su)
    df.attrs["kiyi_altigen"] = len(karisik)
    return df


OSM_KATEGORILER = {
    "Okul / kreş": (r'["amenity"~"^(school|kindergarten)$"]', (245, 245, 245)),
    "Sağlık": (r'["amenity"~"^(hospital|clinic)$"]', (240, 98, 146)),
    "Huzurevi": (r'["social_facility"="nursing_home"]', (174, 213, 129)),
    "İçme suyu": (r'["amenity"="drinking_water"]', (178, 235, 242)),
}
OVERPASS_ADRESLERI = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                      "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
TESIS_CSV = "tesisler_osm.csv"  # osm_hazirla.py ile bir kez üretilip repoya konur; varsa öncelikli kullanılır
TESIS_JSON = "tesisler_osm.json"  # overpass-turbo "ham veri" (JSON) dışa aktarımı da kabul edilir


@st.cache_data(show_spinner=False)
def tesis_dosyasi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    if yol.lower().endswith(".json"):  # Overpass JSON: elements[].tags + lat/lon ya da center
        import json
        satirlar = []
        for e in json.load(open(yol, encoding="utf-8")).get("elements", []):
            t = e.get("tags", {})
            lat = e.get("lat", e.get("center", {}).get("lat"))
            lon = e.get("lon", e.get("center", {}).get("lon"))
            if lat is not None and lon is not None:
                satirlar.append({"@lat": lat, "@lon": lon, "amenity": t.get("amenity"),
                                 "social_facility": t.get("social_facility"), "name": t.get("name")})
        if not satirlar:
            return None
        df = pd.DataFrame(satirlar)
    else:
        df = pd.read_csv(yol, sep=None, engine="python")
    if {"@lat", "@lon"} <= set(df.columns):  # tarayıcıdan indirilen ham Overpass CSV'si
        t = df.rename(columns={"@lat": "lat", "@lon": "lon"})
        for c in ("amenity", "social_facility", "name"):
            if c not in t:
                t[c] = None
        am, sf = t["amenity"].fillna(""), t["social_facility"].fillna("")
        t["kategori"] = np.select(
            [am.isin(["school", "kindergarten"]), am.isin(["hospital", "clinic"]),
             (sf == "nursing_home") | (am == "nursing_home"), am == "drinking_water"],
            ["Okul / kreş", "Sağlık", "Huzurevi", "İçme suyu"], default="")
        t["ad"] = t["name"].fillna(t["kategori"])
        df = t
    if not {"kategori", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["kategori"].isin(list(OSM_KATEGORILER))].copy()
    if "ad" not in df:
        df["ad"] = df["kategori"]
    return df[["kategori", "ad", "lat", "lon"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def osm_tesisleri_cek():
    """OpenStreetMap'ten pilot bölgedeki hassas tesisler. Erişilemezse (None, hata) döner; uydurma veri yok."""
    g, b, k, d = BBOX
    parcalar = []
    for etiket, _ in OSM_KATEGORILER.values():
        parcalar.append(f"nwr{etiket}({g},{b},{k},{d});")
    parcalar.append(f'nwr["amenity"="nursing_home"]({g},{b},{k},{d});')
    sorgu = f"[out:json][timeout:25];({''.join(parcalar)});out center tags;"
    son_hata = None
    for adres in OVERPASS_ADRESLERI:
        try:
            r = requests.post(adres, data={"data": sorgu}, timeout=60,
                              headers={"User-Agent": "CoolScape-AI-prototip/0.4 (Climathon)", "Accept": "application/json"})
            r.raise_for_status()
            satirlar = []
            for e in r.json().get("elements", []):
                t = e.get("tags", {})
                lat = e.get("lat", e.get("center", {}).get("lat"))
                lon = e.get("lon", e.get("center", {}).get("lon"))
                if lat is None or lon is None:
                    continue
                if t.get("amenity") in ("school", "kindergarten"):
                    kat = "Okul / kreş"
                elif t.get("amenity") in ("hospital", "clinic"):
                    kat = "Sağlık"
                elif t.get("social_facility") == "nursing_home" or t.get("amenity") == "nursing_home":
                    kat = "Huzurevi"
                elif t.get("amenity") == "drinking_water":
                    kat = "İçme suyu"
                else:
                    continue
                satirlar.append({"kategori": kat, "ad": t.get("name", kat), "lat": lat, "lon": lon})
            return pd.DataFrame(satirlar, columns=["kategori", "ad", "lat", "lon"]).drop_duplicates(), None
        except Exception as hata:
            son_hata = f"{adres.split('/')[2]}: {type(hata).__name__}"
    return None, str(son_hata)


def parklari_guncelle(parklar, yuzey):
    """Isı yüzeyi varsa:
    - yüzey sıcaklığı farkı parkın ÇEVRESİNDEN (~500 m halka) alınır: korunacak olan çevredeki mahalledir;
    - yeşil skoru parkın KENDİSİNDEN (en yakın altıgenler) alınır: suyun buharlaşma etkisini bitki örtüsü belirler."""
    p = parklar.copy()
    p[["yuzey_farki_c", "yesil_skoru"]] = p[["yuzey_farki_c", "yesil_skoru"]].astype(float)
    p["yuzey_kaynak"] = "temsili"
    p["yesil_kaynak"] = "temsili"
    if yuzey is None:
        return p
    cozunurluk = h3.get_resolution(yuzey["hex"].iloc[0])
    aralik_m = h3.average_hexagon_edge_length(cozunurluk, unit="m") * 3 ** 0.5
    k_cevre = max(1, int(round(CEVRE_YARICAP_M / aralik_m)))
    idx = yuzey.set_index("hex")
    for i, r in p.iterrows():
        merkez = h3.latlng_to_cell(r["lat"], r["lon"], cozunurluk)
        cevre = [c for c in h3.grid_disk(merkez, k_cevre) if c in idx.index]
        yakin = [c for c in h3.grid_disk(merkez, 1) if c in idx.index]
        if cevre:
            p.at[i, "yuzey_farki_c"] = round(float(idx.loc[cevre, "lst_fark_c"].mean()), 2)
            p.at[i, "yuzey_kaynak"] = f"Landsat (~{CEVRE_YARICAP_M} m çevre)"
        if yakin and "ndvi" in idx.columns and idx.loc[yakin, "ndvi"].notna().any():
            ndvi = float(idx.loc[yakin, "ndvi"].mean())
            p.at[i, "yesil_skoru"] = round(float(np.clip(1 + ndvi / 0.8 * 9, 1, 10)), 1)
            p.at[i, "yesil_kaynak"] = "Sentinel-2 NDVI"
    return p


# ---------------------------------------------------------------------------
# MEKÂNSAL ÖNCELİK (CBS analiziyle aynı yöntem): ağırlıklı çakıştırma + yerleşim maskesi
# ---------------------------------------------------------------------------
ONCELIK_AGIRLIK = {"c_isi": 0.40, "c_yasli": 0.25, "c_park": 0.20, "c_tesis": 0.15}
ONCELIK_OLCUT_AD = {"c_isi": "yüzey ısısı", "c_yasli": "65+ sayısı (ilçe oranıyla)", "c_park": "parka uzaklık",
                    "c_tesis": "okul ve sağlık tesisi"}
NUFUS_YERLESIM_ESIK = 50   # kişi/altıgen (~0,1 km² → ~500 kişi/km²): bunun altı seyrek yerleşim
PARK_NUFUS_YARICAP_M = 800  # parkın hizmet ettiği nüfus: ~10 dk yürüme


@st.cache_data(show_spinner=False)
def nufus_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "nufus", "yasli_65"} <= set(df.columns):
        return None
    return df[["hex", "nufus", "yasli_65"]]
SINIF_ADLARI = ["Çok düşük", "Düşük", "Orta", "Yüksek", "Çok yüksek"]
SINIF_RENK = {0: (205, 205, 205), 1: (255, 247, 188), 2: (254, 227, 145), 3: (254, 196, 79),
              4: (236, 112, 20), 5: (153, 0, 13)}
YERLESIM_YARICAP_M = 1000  # bu mesafede en az bir okul/sağlık tesisi olan altıgen "yerleşim" sayılır (nüfus vekili)


def _yakin_say(lat, lon, t_lat, t_lon, yaricap):
    """Her altıgen için yarıçap içindeki tesis sayısı (eşdikdörtgen yaklaşım, kent ölçeğinde yeterli)."""
    kx = 111_320 * np.cos(np.radians(36.89))
    dx = (lon[:, None] - t_lon[None, :]) * kx
    dy = (lat[:, None] - t_lat[None, :]) * 110_540
    return (np.hypot(dx, dy) <= yaricap).sum(axis=1)


@st.cache_data(show_spinner=False)
def oncelik_hesapla(yuzey, parklar, tesisler, nufus=None):
    """Her kara altıgeni için soğutma önceliği. Ölçütler 0–1: yüzey ısısı (Landsat, 5.–95. yüzdelik),
    parka uzaklık (400 m'de 0, 1500 m'de 1), 400 m içindeki okul/sağlık tesisi sayısı (OSM).
    1 km içinde okul/sağlık tesisi olmayan altıgenler seyrek yerleşim sayılır ve sıralamaya girmez.
    Sınıflar yerleşim alanının beşte birlik dilimleridir."""
    y = yuzey.copy()
    if nufus is not None:
        y = y.drop(columns=[c for c in ("nufus", "yasli_65") if c in y]).merge(nufus, on="hex", how="left")
        y[["nufus", "yasli_65"]] = y[["nufus", "yasli_65"]].fillna(0.0)
    lat, lon = y["lat"].values, y["lon"].values
    y["d_park_m"] = np.min([mesafe_m(r["lat"], r["lon"], lat, lon) for _, r in parklar.iterrows()], axis=0)
    p5, p95 = np.nanpercentile(y["lst_fark_c"], [5, 95])
    y["c_isi"] = np.clip((y["lst_fark_c"] - p5) / max(p95 - p5, 1e-6), 0, 1)
    y["c_park"] = np.clip((y["d_park_m"] - 400) / 1100, 0, 1)
    agirlik = dict(ONCELIK_AGIRLIK)
    hassas = None if tesisler is None else tesisler[tesisler["kategori"].isin(["Okul / kreş", "Sağlık", "Huzurevi"])]
    if hassas is not None and len(hassas):
        y["n_tesis_400"] = _yakin_say(lat, lon, hassas["lat"].values, hassas["lon"].values, 400)
        y["n_tesis_1km"] = _yakin_say(lat, lon, hassas["lat"].values, hassas["lon"].values, YERLESIM_YARICAP_M)
        y["c_tesis"] = np.clip(y["n_tesis_400"] / max(np.percentile(y["n_tesis_400"], 95), 1), 0, 1)
        y["yerlesim"] = y["n_tesis_1km"] >= 1
        kural = f"{YERLESIM_YARICAP_M} m içinde en az bir okul veya sağlık tesisi (nüfus verisi yerine vekil)"
    else:
        y["n_tesis_400"], y["n_tesis_1km"], y["c_tesis"] = 0, 0, 0.0
        agirlik.pop("c_tesis")
        y["yerlesim"] = True
        kural = "tesis verisi yok; tüm kara alanı"
    if "nufus" in y:   # gerçek nüfus varsa yerleşim maskesi ve 65+ ölçütü nüfustan
        y["c_yasli"] = np.clip(y["yasli_65"] / max(np.percentile(y["yasli_65"], 95), 1e-6), 0, 1)
        y["yerlesim"] = y["nufus"] >= NUFUS_YERLESIM_ESIK
        kural = f"WorldPop 2025: altıgende en az {NUFUS_YERLESIM_ESIK} kişi"
    else:
        y["c_yasli"] = 0.0
        agirlik.pop("c_yasli")
    y["oncelik"] = sum(y[k] * w for k, w in agirlik.items()) / sum(agirlik.values())
    y["sinif_no"] = 0
    yer = y["yerlesim"]
    if yer.sum() >= 5:
        y.loc[yer, "sinif_no"] = pd.qcut(y.loc[yer, "oncelik"].rank(method="first"), 5,
                                         labels=[1, 2, 3, 4, 5]).astype(int)
    y["sinif"] = y["sinif_no"].map(lambda i: SINIF_ADLARI[i - 1] if i > 0 else "Seyrek yerleşim")
    alan = float(np.mean([h3.cell_area(h, unit="km^2") for h in y["hex"].iloc[:50]]))  # gerçek hücre alanı
    yuksek = y["sinif_no"] >= 4
    bilgi = {
        "agirlik": " · ".join(f"{ONCELIK_OLCUT_AD[k]} {v:.2f}".replace(".", ",") for k, v in agirlik.items()),
        "kural": kural, "yerlesim_km2": float(yer.sum() * alan),
        "park400_pct": float(100 * (y.loc[yer, "d_park_m"] <= 400).mean()) if yer.any() else 0.0,
        "yuksek_km2": float(yuksek.sum() * alan),
        "yuksek_park_disi_pct": float(100 * (y.loc[yuksek, "d_park_m"] > 400).mean()) if yuksek.any() else 0.0,
        "nufus_var": "nufus" in y,
        "yerlesim_nufus": float(y.loc[yer, "nufus"].sum()) if "nufus" in y else None,
        "park400_nufus_pct": (float(100 * y.loc[yer & (y["d_park_m"] <= 400), "nufus"].sum()
                                    / max(y.loc[yer, "nufus"].sum(), 1)) if "nufus" in y else None),
    }
    return y, bilgi


def park_nufus_guncelle(parklar, yuzey):
    """Parkın hizmet ettiği nüfusu veriden türetir (temsili skor yerine):
    - WorldPop varsa: 800 m içindeki nüfus → nüfus skoru (1–10, parklar arasında göreli). 65+ payı ilçe sabiti
      olduğu için kırılganlığa aktarılmaz.
    - Yoksa ve OSM tesisleri varsa: 800 m içindeki okul/sağlık tesisi sayısı → nüfus skoru (vekil); kırılgan % temsili."""
    p = parklar.copy()
    p["nufus_kaynak"] = "temsili"
    p["hassas_kaynak"] = "temsili"
    p[["nufus_skoru", "hassas_yuzde"]] = p[["nufus_skoru", "hassas_yuzde"]].astype(float)
    if yuzey is None:
        return p
    lat, lon = yuzey["lat"].values, yuzey["lon"].values
    olcu, kaynak = [], None
    for i, r in p.iterrows():
        icinde = mesafe_m(r["lat"], r["lon"], lat, lon) <= PARK_NUFUS_YARICAP_M
        if "nufus" in yuzey:
            n = float(yuzey.loc[icinde, "nufus"].sum())
            olcu.append(n)
            kaynak = f"WorldPop 2025 ({PARK_NUFUS_YARICAP_M} m içi nüfus)"
        elif "n_tesis_1km" in yuzey and yuzey["n_tesis_1km"].sum() > 0:
            olcu.append(float(yuzey.loc[icinde, "n_tesis_400"].sum()))
            kaynak = f"OSM okul/sağlık yoğunluğu ({PARK_NUFUS_YARICAP_M} m, vekil)"
    if kaynak and olcu and max(olcu) > 0:
        p["nufus_skoru"] = [round(1 + 9 * o / max(olcu), 1) for o in olcu]
        p["nufus_kaynak"] = kaynak
    return p


def kirilganlik_ata(noktalar, tesisler, nufus_var):
    """Tüm müdahale noktalarına (park + aday) AYNI veri ve kuralla kırılganlık göstergesi (0–1) atar:
    - OSM varsa: 400 m içindeki okul/kreş, sağlık ve huzurevi sayısı / noktalar arasındaki en yüksek değer
      (vekil: çocukların ve hastaların yoğun olduğu yerler);
    - hiçbiri yoksa: temsili kırılgan % / 30.
    Kırılgan noktalar, göstergesi tüm noktaların medyanında ya da üstünde olanlardır."""
    p = noktalar.copy()
    hassas = None if tesisler is None else tesisler[tesisler["kategori"].isin(["Okul / kreş", "Sağlık", "Huzurevi"])]
    if hassas is not None and len(hassas):
        n = np.array([int((mesafe_m(r["lat"], r["lon"], hassas["lat"].values, hassas["lon"].values) <= 400).sum())
                      for _, r in p.iterrows()])
        p["hassas_tesis_400"] = n
        p["kirilgan_endeks"] = n / max(int(n.max()), 1)
        kaynak = "OSM: 400 m içinde okul/kreş, sağlık, huzurevi sayısı (vekil)"
    else:
        p["kirilgan_endeks"] = np.clip(p["hassas_yuzde"].astype(float) / 30.0, 0, 1)
        kaynak = "temsili kırılgan %"
    p["kirilgan_endeks"] = p["kirilgan_endeks"].round(3)
    p["kirilgan_kaynak"] = kaynak
    esik = float(p["kirilgan_endeks"].median())
    kirilgan = tuple(sorted(p.loc[(p["kirilgan_endeks"] >= esik) & (p["kirilgan_endeks"] > 0), "park"]))
    return p, kirilgan, kaynak


def _aday_hassas(yuzey, r):
    """Aday noktanın 400 m çevresindeki 65+ payı (WorldPop); yoksa temsili kent oranı."""
    if True:   # 65+ payı ilçe sabiti; kırılganlık OSM vekilinden gelir, bu alan yalnızca temsili yedektir
        return SEHIR_KIRILGAN_YUZDE
    icinde = mesafe_m(r["lat"], r["lon"], yuzey["lat"].values, yuzey["lon"].values) <= 400
    n = float(yuzey.loc[icinde, "nufus"].sum())
    return round(100 * float(yuzey.loc[icinde, "yasli_65"].sum()) / n, 1) if n > 0 else SEHIR_KIRILGAN_YUZDE


def aday_noktalar(yuzey, parklar):
    """Sisleme ve ıslatma için ADAY konumlar. Öncelik analizi varsa en öncelikli YERLEŞİM altıgenlerinden,
    yoksa en sıcak altıgenlerden; birbirinden ve parklardan en az 1 km uzak. Kesin yer sahada belediyeyle seçilir."""
    if yuzey is None:
        return parklar.iloc[0:0].copy()
    y = yuzey[yuzey["lon"] <= ADAY_DOGU_SINIRI].copy()
    if "oncelik" in y.columns:
        y = y[y["yerlesim"]]
    d_park = np.min([mesafe_m(r["lat"], r["lon"], y["lat"].values, y["lon"].values)
                     for _, r in parklar.iterrows()], axis=0)
    y["d_park"] = d_park
    y = y[(y["d_park"] >= ADAY_PARK_UZAKLIK_M[0]) & (y["d_park"] <= ADAY_PARK_UZAKLIK_M[1])]
    y = y.sort_values("oncelik" if "oncelik" in y.columns else "lst_c", ascending=False)
    secilen = []
    for _, r in y.iterrows():
        if all(mesafe_m(r["lat"], r["lon"], a["lat"], a["lon"]) >= ADAY_ARALIK_M for a in secilen):
            secilen.append(r)
        if len(secilen) >= ADAY_SISLEME + ADAY_ISLATMA:
            break
    satirlar = []
    for i, r in enumerate(secilen):
        tur = "sisleme" if i < ADAY_SISLEME else "islatma"
        no = i + 1 if tur == "sisleme" else i - ADAY_SISLEME + 1
        en_yakin = parklar.iloc[int(np.argmin(mesafe_m(r["lat"], r["lon"], parklar["lat"].values,
                                                       parklar["lon"].values)))]
        satirlar.append({
            "park": f"{'Sisleme' if tur == 'sisleme' else 'Islatma'} {no} (aday)", "ilce": en_yakin["ilce"],
            "lat": r["lat"], "lon": r["lon"], "yuzey_farki_c": round(float(r["lst_fark_c"]), 2),
            "yesil_skoru": 0.0, "nufus_skoru": 0, "hassas_yuzde": _aday_hassas(yuzey, r), "pompa_katsayisi": 1.0,
            "hat_kapasitesi_m3_saat": SISLEME_SU_M3_SAAT if tur == "sisleme" else ISLATMA_KAPASITE,
            "sulanan_alan_ha": (SISLEME_ALAN_M2 if tur == "sisleme" else ISLATMA_ALAN_M2) / 10_000,
            "aritilmis_su_erisimi": tur == "islatma", "tur": tur,
            "yuzey_kaynak": "Landsat (altıgenin kendisi)", "yesil_kaynak": "–"})
    return pd.DataFrame(satirlar)


# ---------------------------------------------------------------------------
# FİZİK
# ---------------------------------------------------------------------------
def vpd_kpa(sicaklik_c, bagil_nem):
    t = np.asarray(sicaklik_c, dtype=float)
    rh = np.asarray(bagil_nem, dtype=float)
    try:
        from metpy.calc import saturation_vapor_pressure
        from metpy.units import units
        es = saturation_vapor_pressure(t * units.degC).to("kPa").magnitude
    except Exception:
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100.0)


# Ortalama ışınım sıcaklığı (MRT) = hava sıcaklığı + güneş katkısı (SolarCal) + zemin uzun dalga katkısı
# - Güneş katkısı: ASHRAE 55-2020 Ek C SolarCal modeli (pythermalcomfort.solar_gain), doğrudan normal ışınım (DNI,
#   Open-Meteo) ve hesaplanan güneş yüksekliğiyle. Varsayımlar: ayakta duran kişi, güneşte (f_bes = 1),
#   gökyüzü görüş oranı 0,5 (kent sokağı), güneş yandan (göreli açı 90°), giysi yutuculuğu 0,7.
# - Zemin katkısı: yüzey sıcaklığı farkı × 0,5 (yaya yarım küresinin yaklaşık yarısı zemindir; doğrusal ışınım yaklaşımı).
# DNI yoksa (ör. elle yüklenen CSV) eski kaba yaklaşım kullanılır: 0,02 × küresel ışınım.
ZEMIN_GORUS = 0.5
SOLARCAL = {"sharp": 90, "f_svv": 0.5, "f_bes": 1.0, "asw": 0.7, "posture": "standing"}


def gunes_yuksekligi(zaman, lat=ANTALYA_LAT, lon=ANTALYA_LON):
    """Güneş yüksekliği (°), NOAA yaklaşımı. Open-Meteo saatlik ışınımı önceki saatin ortalamasıdır; saat ortası alınır.
    zaman: Europe/Istanbul yerel saati (UTC+3)."""
    t = pd.to_datetime(pd.Series(zaman)) - pd.Timedelta(hours=3, minutes=30)
    doy = t.dt.dayofyear.values
    saat = t.dt.hour.values + t.dt.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (saat - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    dek = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((saat * 60 + eqt + 4 * lon) / 4 - 180)
    f = np.radians(lat)
    return np.degrees(np.arcsin(np.clip(np.sin(f) * np.sin(dek) + np.cos(f) * np.cos(dek) * np.cos(ha), -1, 1)))


def gunes_mrt_ekle(hava):
    """Her saat için güneşin MRT'ye katkısı (dmrt_gunes, °C) ve hesap yöntemi."""
    h = hava.copy()
    if "dni_wm2" in h and h["dni_wm2"].notna().all() and "zaman" in h:
        from pythermalcomfort.models import solar_gain
        alt = gunes_yuksekligi(h["zaman"])
        d = []
        for a, dni in zip(alt, h["dni_wm2"]):
            if a <= 0 or dni <= 0:
                d.append(0.0)
                continue
            r = solar_gain(sol_altitude=float(min(a, 90)), sharp=SOLARCAL["sharp"], sol_radiation_dir=float(dni),
                           sol_transmittance=1.0, f_svv=SOLARCAL["f_svv"], f_bes=SOLARCAL["f_bes"],
                           asw=SOLARCAL["asw"], posture=SOLARCAL["posture"])
            d.append(float(r["delta_mrt"] if isinstance(r, dict) else r.delta_mrt))
        h["gunes_yuksekligi"] = np.round(alt, 1)
        h["dmrt_gunes"] = d
        h.attrs["mrt_yontem"] = "SolarCal (ASHRAE 55)"
    else:
        h["dmrt_gunes"] = 0.02 * h["gunes_wm2"]
        h.attrs["mrt_yontem"] = "kaba yaklaşım (0,02 × küresel ışınım)"
    return h


def utci_hesapla(tdb, mrt, ruzgar, nem):
    """UTCI (Bröde vd. 2012, pythermalcomfort). MRT ve rüzgâr modelin geçerlilik sınırına kırpılır."""
    tdb = np.asarray(tdb, dtype=float)
    mrt = np.clip(np.asarray(mrt, dtype=float), tdb - 30, tdb + 70)
    v = np.clip(np.asarray(ruzgar, dtype=float), 0.5, 17.0)
    sonuc = utci(tdb=tdb, tr=mrt, v=v, rh=np.asarray(nem, dtype=float))
    return np.asarray(getattr(sonuc, "utci", sonuc), dtype=float)


def park_saat_tablosu(hava, parklar):
    h = hava[hava["saat"].isin(SAATLER)]
    parcalar = []
    for _, p in parklar.iterrows():
        d = h.copy()
        d["park"] = p["park"]
        d["mrt_c"] = d["sicaklik_c"] + d["dmrt_gunes"] + ZEMIN_GORUS * p["yuzey_farki_c"]
        parcalar.append(d)
    df = pd.concat(parcalar, ignore_index=True)
    df["utci_c"] = utci_hesapla(df["sicaklik_c"], df["mrt_c"], df["ruzgar_ms"], df["bagil_nem"])
    df["vpd_kpa"] = vpd_kpa(df["sicaklik_c"], df["bagil_nem"])
    df["tehlike"] = (df["utci_c"] - UTCI_ESIK).clip(lower=0).fillna(0)
    return df


def yuzey_utci(yuzey, hava_satiri):
    t = hava_satiri["sicaklik_c"]
    n = len(yuzey)
    return utci_hesapla(np.full(n, t), t + hava_satiri["dmrt_gunes"] + ZEMIN_GORUS * yuzey["lst_fark_c"].values,
                        np.full(n, hava_satiri["ruzgar_ms"]), np.full(n, hava_satiri["bagil_nem"]))


def kent_utci(satir):
    return float(utci_hesapla([satir["sicaklik_c"]], [satir["sicaklik_c"] + satir["dmrt_gunes"]],
                              [satir["ruzgar_ms"]], [satir["bagil_nem"]])[0])


def utci_sinif(deger):
    for alt_s, ust_s, ad, renk in UTCI_SINIFLARI:
        if alt_s <= deger < ust_s:
            return ad, renk
    return UTCI_SINIFLARI[-1][2], UTCI_SINIFLARI[-1][3]


# ---------------------------------------------------------------------------
# OPTİMİZASYON (MILP)
# ---------------------------------------------------------------------------
def sogutma_egrisi(x_m3, emax, x0):
    """x0: m³ cinsinden azalan getiri ölçeği = X0_MM × sulanan alan."""
    return emax * (1 - np.exp(-np.asarray(x_m3, dtype=float) / np.asarray(x0, dtype=float)))


def girdileri_hazirla(tablo, noktalar, w_kirilganlik, erisim):
    """Her nokta × saat için öncelik, azami etki, azalan getiri ölçeği, kapasite ve erişim (kişi)."""
    p = noktalar.set_index("park")
    t = tablo.copy()
    t["tur"] = t["park"].map(p["tur"])
    t["kisi"] = np.select([t["tur"] == "park", t["tur"] == "sisleme"],
                          [t["park"].map(p["nufus_skoru"]) * erisim["park_carpan"], erisim["sisleme"]],
                          default=erisim["islatma"]).astype(float)
    endeks = p["kirilgan_endeks"] if "kirilgan_endeks" in p else p["hassas_yuzde"] / 30.0
    t["oncelik"] = t["tehlike"] * t["kisi"] / 100.0 * (1 + w_kirilganlik * t["park"].map(endeks))
    vpd_f = np.minimum(t["vpd_kpa"] / VPD_DOYMA, 1.0)
    t["emax"] = np.where(t["tur"] == "sisleme", SISLEME_E, E_REF) * vpd_f
    t["alan_m2"] = t["park"].map(p["sulanan_alan_ha"]) * 10_000
    t["x0"] = np.where(t["tur"] == "sisleme", SISLEME_SU_M3_SAAT / 10.0, X0_MM * t["alan_m2"] / 1000.0)
    t["kapasite"] = t["park"].map(p["hat_kapasitesi_m3_saat"]).astype(float)
    t["pompa"] = t["park"].map(p["pompa_katsayisi"])
    return t


def degisken(model, ad, alt_s=None, ust_s=None, tur="Continuous"):
    if hasattr(model, "add_variable"):  # PuLP 4.x
        return model.add_variable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)
    return pulp.LpVariable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)


def cozucu_sec():
    for ad in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "COIN_CMD"):
        sinif = getattr(pulp, ad, None)
        if sinif is None:
            continue
        try:
            c = sinif(msg=False)
            if c.available():
                return c
        except Exception:
            continue
    raise RuntimeError("MILP çözücüsü bulunamadı (pulp==2.9.0 kurulu olmalı).")


TUM_TURLER = ("park", "sisleme", "islatma")


def optimize_et(t, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=1.0,
                turler=TUM_TURLER, park_kesme=18, park_gunluk_mm=6.0):
    """Öncelik ağırlıklı UTCI düşüşünü en çoklayan MILP.
    Park/ıslatma: parçalı doğrusal azalan getiri eğrisi. Sisleme: aç/kapa, sabit su.
    Çim güvenliği: park sulaması akşam kesme saatinden sonra yok; park başına günlük üst sınır (mm)."""
    if float(t["oncelik"].sum()) <= 0:  # hiçbir saatte güçlü ısı stresi yok: sulama önerilmez
        sonuc = t.copy()
        sonuc["su_m3"] = 0.0
        return sonuc, "Optimal"
    model = pulp.LpProblem("CoolScape", pulp.LpMaximize)
    su, amac, toplam, kirilgan, park_gunluk = {}, [], [], [], {}
    for idx, r in t.iterrows():
        y = degisken(model, f"y_{idx}", tur="Binary")
        kapali = (r["tur"] not in turler or r["vpd_kpa"] < vpd_min
                  or (r["tur"] == "park" and r["saat"] >= park_kesme))
        if kapali:
            model += y == 0
        if r["tur"] == "sisleme":
            x = SISLEME_SU_M3_SAAT * y
            amac.append(r["oncelik"] * float(sogutma_egrisi(SISLEME_SU_M3_SAAT, r["emax"], r["x0"])) * y)
            su[idx] = ("sabit", y)
        else:
            segs = []
            for k in range(len(KIRILIM) - 1):
                a, b = KIRILIM[k], min(KIRILIM[k + 1], r["kapasite"])
                if b <= a:
                    break
                v = degisken(model, f"s_{idx}_{k}", alt_s=0, ust_s=float(b - a))
                x0 = r["x0"] * x0_carpan
                egim = (sogutma_egrisi(b, r["emax"], x0) - sogutma_egrisi(a, r["emax"], x0)) / (b - a)
                amac.append(r["oncelik"] * egim * v)
                if enerji_opt:
                    amac.append(-0.01 * r["pompa"] * v)
                segs.append(v)
            x = pulp.lpSum(segs)
            model += x <= r["kapasite"] * y
            model += x >= min(MIN_DOZ, r["kapasite"]) * y
            su[idx] = ("parca", segs)
            if r["tur"] == "park":
                park_gunluk.setdefault(r["park"], ([], r["alan_m2"]))[0].append(x)
        toplam.append(x)
        if r["park"] in kirilgan_parklar and r["tur"] in turler:   # yalnızca bu planda kullanılabilen kırılgan noktalar
            kirilgan.append(x)
    for ad, (xs, alan) in park_gunluk.items():
        model += pulp.lpSum(xs) <= park_gunluk_mm * alan / 1000.0
    model += pulp.lpSum(amac)
    model += pulp.lpSum(toplam) <= butce
    if kirilgan and kirilgan_pay > 0:
        model += pulp.lpSum(kirilgan) >= kirilgan_pay * pulp.lpSum(toplam)
    model.solve(cozucu_sec())
    durum = pulp.LpStatus[model.status] if hasattr(pulp, "LpStatus") else str(model.status)
    sonuc = t.copy()
    degerler = []
    for i in t.index:
        tip, v = su[i]
        degerler.append(round(SISLEME_SU_M3_SAAT * (v.value() or 0), 3) if tip == "sabit"
                        else round(sum((s_.value() or 0) for s_ in v), 2))
    sonuc["su_m3"] = degerler
    return sonuc, durum


def esit_dagitim(t, butce, park_kesme=18, park_gunluk_mm=6.0):
    """Karşılaştırma: bütçe yalnızca parklara, izin verilen gündüz saatlerine eşit bölünür; nem, ısı stresi ve
    kırılganlık gözetilmez. Adil olsun diye CoolScape'le AYNI hat kapasitesi ve çim güvenliği kurallarına uyar."""
    sonuc = t.copy()
    sonuc["su_m3"] = 0.0
    izin = (sonuc["tur"] == "park") & (sonuc["saat"] < park_kesme)
    if izin.sum() == 0:
        return sonuc
    pay_ = np.minimum(butce / int(izin.sum()), sonuc.loc[izin, "kapasite"])
    tavan = (park_gunluk_mm * sonuc.loc[izin, "alan_m2"] / 1000.0) / sonuc.loc[izin].groupby("park")["saat"].transform("count")
    sonuc.loc[izin, "su_m3"] = np.minimum(pay_, tavan)
    return sonuc


def etki(sonuc, x0_carpan=1.0, emax_carpan=1.0):
    s = sonuc.copy()
    x0 = np.where(s["tur"] == "sisleme", s["x0"], s["x0"] * x0_carpan)
    s["utci_dusus_c"] = sogutma_egrisi(s["su_m3"], s["emax"] * emax_carpan, x0)
    s["su_mm"] = s["su_m3"] / (s["alan_m2"] / 1000.0)
    s["agirlikli_fayda"] = s["oncelik"] * s["utci_dusus_c"]
    s["kisi_derece_saat"] = s["kisi"] * s["utci_dusus_c"] * (s["tehlike"] > 0)
    s["utci_sonra_c"] = s["utci_c"] - s["utci_dusus_c"]
    return s


@st.cache_data(show_spinner="Su planı hesaplanıyor...")
def plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, turler, park_kesme, park_mm):
    """Ayarlar her değiştiğinde otomatik çalışır (önbellekli): nominal plan, yalnız-park planı, duyarlılık."""
    kirilgan_parklar = set(kirilgan_parklar)
    ayar = dict(turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
    plan, durum = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, **ayar)
    if durum != "Optimal":
        return None, None, None, durum
    esit = esit_dagitim(girdi, butce, park_kesme, park_mm)
    plan_e, esit_e = etki(plan), etki(esit)
    fayda_esit = esit_e["agirlikli_fayda"].sum()
    artislar = []
    for c in X0_ARALIK:
        p2, d2 = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=c, **ayar)
        fz = etki(esit, c)["agirlikli_fayda"].sum()
        if d2 == "Optimal" and fz > 0:
            artislar.append(etki(p2, c)["agirlikli_fayda"].sum() / fz - 1)
    yalniz_park = None
    if set(turler) != {"park"}:
        pp, dp = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                             turler=("park",), park_kesme=park_kesme, park_gunluk_mm=park_mm)
        if dp == "Optimal" and fayda_esit > 1e-9:
            yalniz_park = etki(pp)["agirlikli_fayda"].sum() / fayda_esit - 1
    if artislar and fayda_esit > 1e-9:
        nominal = plan_e["agirlikli_fayda"].sum() / fayda_esit - 1
        artislar.append(nominal)
    sulanan = plan_e[plan_e["su_m3"] > 0.001]
    ozet = {
        "isi_stresi_var": fayda_esit > 1e-9,
        "artis": plan_e["agirlikli_fayda"].sum() / fayda_esit - 1 if fayda_esit > 1e-9 else 0.0,
        "artis_aralik": (min(artislar), max(artislar)) if artislar else None,
        "yalniz_park": yalniz_park,
        "kds": float(plan_e["kisi_derece_saat"].sum()), "kds_esit": float(esit_e["kisi_derece_saat"].sum()),
        "dusus_aralik": (float(sulanan["utci_dusus_c"].min()), float(sulanan["utci_dusus_c"].max()))
        if len(sulanan) else None,
        "su_tur": plan_e.groupby("tur")["su_m3"].sum().to_dict(),
    }
    return plan_e, esit_e, ozet, durum


def operator_komutlari(sonuc, parklar, aritilmis_senaryo):
    p = parklar.set_index("park")
    satirlar = []
    for park, g in sonuc[sonuc["su_m3"] > 0.001].sort_values("saat").groupby("park"):
        miktar = g.set_index("saat")["su_m3"]
        saatler = list(miktar.index)
        tur = p.loc[park, "tur"]
        if tur == "sisleme":
            kaynak = "İçme suyu (zorunlu, lejyonella riski)"
        elif aritilmis_senaryo and p.loc[park, "aritilmis_su_erisimi"]:
            kaynak = "Arıtılmış su (senaryo)" + (", arazöz" if tur == "islatma" else "")
        else:
            kaynak = "Arazöz, mevcut kaynak" if tur == "islatma" else "Mevcut kaynak"
        bas = onceki = saatler[0]
        for sa in saatler[1:] + [None]:
            if sa is None or sa != onceki + 1:
                satirlar.append({"Nokta": park, "Yöntem": YONTEMLER[tur][0], "Başlangıç": f"{bas:02d}:00",
                                 "Bitiş": f"{onceki + 1:02d}:00",
                                 "Su (m³)": round(float(miktar.loc[bas:onceki].sum()), 2), "Kaynak": kaynak})
                bas = sa
            onceki = sa if sa is not None else onceki
    return pd.DataFrame(satirlar).sort_values(["Başlangıç", "Nokta"]) if satirlar else pd.DataFrame()


# ---------------------------------------------------------------------------
# HARİTA: tıklanabilir 3B altıgen katmanı
# Altıgen başına ortalama değer Python'da hesaplanır, sabit renk tablosuyla boyanır; yükseklik değeri gösterir.
# Renkler yakınlaştırmayla değişmez; lejant aynı tablodan üretilir. Her altıgenin üzerine gelince değerleri görünür.
# ---------------------------------------------------------------------------
SICAK_DURAKLAR = [(0.00, (255, 255, 178, 0)), (0.15, (254, 204, 92, 120)), (0.40, (253, 141, 60, 170)),
                  (0.70, (240, 59, 32, 205)), (1.00, (189, 0, 38, 230))]
SOGUK_DURAKLAR = [(0.00, (198, 219, 239, 0)), (0.08, (158, 202, 225, 150)), (0.30, (107, 174, 214, 190)),
                  (0.60, (49, 130, 189, 215)), (1.00, (8, 81, 156, 235))]
SOGUMA_UST_C = E_REF   # soğuma ölçeği sabit: 0 → 1 °C UTCI düşüşü
YUKSEKLIK_M = 900      # 3B görünümde en yüksek değerin sütun yüksekliği (m), yalnızca görsel


def renk_tablosu(t, duraklar, alfa_carpan=1.0, taban_alfa=0):
    konum = np.array([d[0] for d in duraklar])
    renk = np.array([d[1] for d in duraklar], dtype=float)
    rgba = np.stack([np.interp(t, konum, renk[:, i]) for i in range(4)], axis=-1)
    rgba[:, 3] = np.where(t < 0.02, taban_alfa, rgba[:, 3] * alfa_carpan)
    return rgba.astype(int).tolist()


def mesafe_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    f1, f2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def mahalle_sogumasi(yuzey, parklar, plan, saat):
    """Her altıgen için o saatteki tahmini UTCI düşüşü: noktadaki düşüş × e^(−mesafe/ölçek), kesme mesafesinden sonra 0.
    Ölçek yönteme göre: park 150/500 m, ıslatma 60/200 m, sisleme 15/50 m (varsayım)."""
    sog = np.zeros(len(yuzey))
    if plan is None:
        return sog
    h = plan[plan["saat"] == saat].set_index("park")
    for _, p in parklar.iterrows():
        dusus = float(h["utci_dusus_c"].get(p["park"], 0.0))
        if dusus <= 0:
            continue
        _, _, olcek, kesme = YONTEMLER[p["tur"]]
        d = mesafe_m(p["lat"], p["lon"], yuzey["lat"].values, yuzey["lon"].values)
        sog += np.where(d <= kesme, dusus * np.exp(-d / olcek), 0.0)
    return sog


def harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod, secenekler, tesisler, boyut3=True):
    """mod: 'Önce' (mevcut ısı), 'Sonra' (plan uygulandıktan sonra), 'Soğuma' (planın yarattığı serinleme),
    'Öncelik' (mekânsal soğutma önceliği)."""
    katmanlar, ist = [], None
    satir = hava[hava["saat"] == saat].iloc[0]
    if yuzey is not None:
        utci_once = yuzey_utci(yuzey, satir)
        sog = mahalle_sogumasi(yuzey, parklar, plan, saat)
        utci_sonra = utci_once - sog
        esik = float(np.nanquantile(utci_once, 1 - sicak_pay))   # sıcak nokta eşiği "önce" durumundan sabit
        ust = float(max(np.nanquantile(utci_once, 0.99), esik + 0.5))
        alan = float(np.mean([h3.cell_area(h, unit="km^2") for h in yuzey["hex"].iloc[:50]]))
        ist = {"once": int(np.sum(utci_once > esik)), "sonra": int(np.sum(utci_sonra > esik)),
               "alan_km2": float(np.sum(sog >= 0.1) * alan), "azami": float(sog.max()) if len(sog) else 0.0,
               "esik": esik, "ust": ust}
        hx = pd.DataFrame({"hex": yuzey["hex"].values})
        sinif = yuzey["sinif"].values if "sinif" in yuzey else np.array(["–"] * len(yuzey))
        if mod == "Öncelik" and "sinif_no" in yuzey:
            no = yuzey["sinif_no"].values
            hx["renk"] = [list(SINIF_RENK[int(i)]) + ([215] if i > 0 else [70]) for i in no]
            hx["yukseklik"] = np.where(no > 0, yuzey["oncelik"].values, 0.0) * YUKSEKLIK_M
            hx["ad"] = [f"{c} öncelik" if c != "Seyrek yerleşim" else c for c in sinif]
        elif mod == "Soğuma":
            t = np.clip(np.nan_to_num(sog / SOGUMA_UST_C), 0, 1)
            hx["renk"] = renk_tablosu(t, SOGUK_DURAKLAR, 1.0, taban_alfa=0)
            hx["yukseklik"] = t * YUKSEKLIK_M
            hx["ad"] = [f"Soğuma {v:.2f} °C" for v in sog]
        else:
            deger = utci_sonra if mod == "Sonra" else utci_once
            t = np.nan_to_num(np.clip((deger - esik) / (ust - esik), 0, 1) * (deger > esik))
            alfa = float(np.clip((kent_utci(satir) - 26) / 20, 0.35, 1.0))  # sabah/akşam sıcak noktalar soluklaşır
            hx["renk"] = renk_tablosu(t, SICAK_DURAKLAR, alfa, taban_alfa=28)
            hx["yukseklik"] = t * YUKSEKLIK_M
            hx["ad"] = [f"UTCI {v:.1f} °C" for v in deger]
        d_park = yuzey["d_park_m"].values if "d_park_m" in yuzey else np.full(len(yuzey), np.nan)
        n_t = yuzey["n_tesis_400"].values if "n_tesis_400" in yuzey else np.zeros(len(yuzey))
        hx["bilgi"] = [f"UTCI önce {a:.1f} → sonra {b:.1f} °C · yüzey farkı {f:+.1f} °C\n"
                       f"Öncelik: {c} · en yakın park {d:,.0f} m · 400 m'de {int(n)} okul/sağlık"
                       for a, b, f, c, d, n in zip(utci_once, utci_sonra, yuzey["lst_fark_c"].values, sinif, d_park, n_t)]
        if mod == "Soğuma":
            hx = hx[np.array([r[3] for r in hx["renk"]]) > 0]
        if len(hx):
            katmanlar.append(pdk.Layer(
                "H3HexagonLayer", hx, id="altigen", get_hexagon="hex", get_fill_color="renk",
                get_elevation="yukseklik", elevation_scale=1, extruded=boyut3, coverage=0.92, stroked=False,
                filled=True, pickable=True, auto_highlight=True, highlight_color=[255, 255, 255, 120]))
    p = parklar[["park", "ilce", "lat", "lon", "tur"]].copy()
    if plan is not None:
        h = plan[plan["saat"] == saat].set_index("park")
        p["su"] = p["park"].map(h["su_m3"]).fillna(0)
        p["once"] = p["park"].map(h["utci_c"])
        p["sonra"] = p["park"].map(h["utci_sonra_c"])
        p["yaricap"] = np.where(p["tur"] == "park", np.sqrt(p["su"]) * 20 + 45, 45)
        p["renk"] = [list(YONTEMLER[t][1]) + [235] if s_ > 0 else [140, 140, 140, 200]
                     for t, s_ in zip(p["tur"], p["su"])]
        p["bilgi"] = [f"{YONTEMLER[t][0]} · {s_:.2f} m³ · UTCI {a:.1f} → {b:.1f} °C"
                      for t, s_, a, b in zip(p["tur"], p["su"], p["once"], p["sonra"])]
    else:
        p["yaricap"], p["renk"], p["bilgi"] = 90, [[140, 140, 140, 200]] * len(p), ""
    p["sekil"] = [[255, 255, 255] if t == "park" else list(YONTEMLER[t][1]) for t in p["tur"]]
    p["ad"] = p["park"]
    p["kisa"] = [a if t == "park" else ("S" if t == "sisleme" else "I") + a.split()[1]
                 for a, t in zip(p["park"], p["tur"])]
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        pk = p[p["tur"] == "park"]
        halka = pd.concat([pk.assign(r=YURUME_5DK_M), pk.assign(r=YURUME_10DK_M)])
        katmanlar.append(pdk.Layer("ScatterplotLayer", halka, id="yurume", get_position="[lon, lat]",
                                   get_radius="r", filled=False, stroked=True, get_line_color=[150, 150, 150, 200],
                                   line_width_min_pixels=1.5))
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and not tesisler.empty:
        t_df = tesisler.copy()
        t_df["renk"] = t_df["kategori"].map(lambda k: list(OSM_KATEGORILER[k][1]) + [230])
        t_df["bilgi"] = t_df["kategori"]
        katmanlar.append(pdk.Layer("ScatterplotLayer", t_df, id="tesis", get_position="[lon, lat]",
                                   get_radius=30, radius_min_pixels=3, get_fill_color="renk", pickable=True))
    if boyut3:  # 3B'de müdahale noktaları altıgenlerin üzerinden görünen sütunlar
        p["yukseklik_nokta"] = YUKSEKLIK_M * 1.2
        katmanlar.append(pdk.Layer("ColumnLayer", p, id="parklar", get_position="[lon, lat]",
                                   get_elevation="yukseklik_nokta", radius=70, extruded=True, disk_resolution=16,
                                   get_fill_color="renk", pickable=True))
    else:
        katmanlar.append(pdk.Layer("ScatterplotLayer", p, id="parklar", get_position="[lon, lat]",
                                   get_radius="yaricap", get_fill_color="renk", pickable=True, radius_min_pixels=7,
                                   stroked=True, get_line_color="sekil", line_width_min_pixels=2))
    if "Nokta adları" in secenekler:
        p["z"] = YUKSEKLIK_M * 1.3 if boyut3 else 0
        katmanlar.append(pdk.Layer("TextLayer", p, id="etiket", get_position="[lon, lat, z]", get_text="kisa",
                                   get_size=13, get_color=[255, 255, 255], get_pixel_offset=[0, -14],
                                   character_set=pdk.types.String("auto"), font_weight=600, background=True,
                                   font_family=pdk.types.String("Helvetica, Arial, sans-serif"),
                                   get_background_color=[20, 20, 20, 190], background_padding=[4, 2]))
    merkez = yuzey if yuzey is not None else parklar
    gorunum = pdk.ViewState(latitude=float(merkez["lat"].mean()) - (0.035 if boyut3 else 0),
                            longitude=float(merkez["lon"].mean()), zoom=11.25 if boyut3 else 11.6,
                            pitch=50 if boyut3 else 0, bearing=-12 if boyut3 else 0)
    deck = pdk.Deck(layers=katmanlar, initial_view_state=gorunum, tooltip={"text": "{ad}\n{bilgi}"})
    return deck, ist


def gradyan(duraklar, genislik=150):
    renkler = ", ".join(f"rgba({r},{g},{b},{max(a, 70) / 255:.2f}) {k * 100:.0f}%" for k, (r, g, b, a) in duraklar)
    return (f"<span style='display:inline-block;width:{genislik}px;height:10px;vertical-align:middle;"
            f"border-radius:2px;background:linear-gradient(to right, {renkler})'></span>")


def lejant_html(mod, ist, sicak_pay, tesis_var, boyut3=False):
    parca = []
    if ist is not None and mod == "Öncelik":
        kutu = "".join(f"<span style='display:inline-block;width:14px;height:10px;margin:0 2px 0 6px;"
                       f"background:rgb{SINIF_RENK[i]}'></span>{SINIF_ADLARI[i - 1].lower()}" for i in range(1, 6))
        parca.append(f"<b>Soğutma önceliği</b>{kutu}<span style='display:inline-block;width:14px;height:10px;"
                     f"margin:0 2px 0 6px;background:rgb{SINIF_RENK[0]}'></span>seyrek yerleşim")
    elif ist is not None and mod == "Soğuma":
        parca.append(f"<b>Soğuma etkisi</b> (tahmini UTCI düşüşü) &nbsp;0 °C {gradyan(SOGUK_DURAKLAR)} "
                     f"{SOGUMA_UST_C:.1f} °C")
    elif ist is not None:
        baslik = "Plan uygulanmış" if mod == "Sonra" else "Mevcut durum"
        parca.append(f"<b>{baslik}: sıcak noktalar</b> (kentin en sıcak %{sicak_pay * 100:.0f}'lik alanı) &nbsp;"
                     f"{ist['esik']:.1f} °C {gradyan(SICAK_DURAKLAR)} {ist['ust']:.1f} °C UTCI")
    else:
        parca.append("<b>Isı yüzeyi dosyası yok:</b> yalnızca parklar gösteriliyor")
    parca.append(" &nbsp;".join(f"<span style='color:rgb{YONTEMLER[k][1]}'>●</span> {YONTEMLER[k][0].lower()}"
                                for k in YONTEMLER) + " &nbsp;<span style='color:#8c8c8c'>●</span> bu saatte kapalı")
    if tesis_var:
        parca.append(" &nbsp;".join(f"<span style='color:rgb{OSM_KATEGORILER[k][1]}'>●</span> {k.lower()}"
                                    for k in OSM_KATEGORILER))
    if boyut3 and ist is not None:
        parca.append("<span style='color:gray'>sütun yüksekliği renkteki değerle orantılı · sağ tıkla sürükleyerek döndürün</span>")
    return "<div style='font-size:0.82rem;line-height:1.7'>" + "<br>".join(parca) + "</div>"


def isi_seridi_html(satir):
    u = kent_utci(satir)
    ad, (r, g, b) = utci_sinif(u)
    yazi = "#000" if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else "#fff"
    return (f"<div style='background:rgb({r},{g},{b});color:{yazi};padding:6px 10px;border-radius:4px;"
            f"font-weight:600'>Kent geneli UTCI {u:.0f} °C · {ad.lower()} ısı stresi</div>")


def saat_karti(saat, hava, plan, vpd_min):
    satir = hava[hava["saat"] == saat].iloc[0]
    st.markdown(f"#### {saat:02d}:00")
    st.markdown(isi_seridi_html(satir), unsafe_allow_html=True)
    st.caption(f"Hava {satir['sicaklik_c']:.1f} °C · nem %{satir['bagil_nem']:.0f} · "
               f"rüzgâr {satir['ruzgar_ms']:.1f} m/s · güneş {satir['gunes_wm2']:.0f} W/m² · "
               f"güneşte MRT +{satir['dmrt_gunes']:.0f} °C".replace(".", ","))
    if plan is None:
        return
    h = plan[plan["saat"] == saat].sort_values("su_m3", ascending=False)
    sulanan = h[h["su_m3"] > 0.001]
    if sulanan.empty:
        if h["tehlike"].sum() == 0:
            st.write("Bu saatte güçlü ısı stresi yok; sulama gerekmiyor.")
        elif (h["vpd_kpa"] < vpd_min).all():
            st.write("Bu saatte sulama yok: hava nemli, su az serinletir (nem kapısı).")
        else:
            st.write("Bu saatte sulama yok: su daha verimli olacağı saatlere ayrıldı.")
    sulanan = sulanan.sort_values("agirlikli_fayda", ascending=False)
    for _, r in sulanan.head(6).iterrows():
        sayi = lambda x, f: format(x, f).replace(".", ",")
        st.markdown(f"**{r['park']}** · {YONTEMLER[r['tur']][0].lower()} · {sayi(r['su_m3'], '.2g')} m³  \n"
                    f"UTCI {sayi(r['utci_c'], '.1f')} → {sayi(r['utci_sonra_c'], '.1f')} °C · "
                    f"<span style='color:gray'>{utci_sinif(r['utci_c'])[0].lower()} stres, "
                    f"VPD {sayi(r['vpd_kpa'], '.1f')} kPa</span>", unsafe_allow_html=True)
    if len(sulanan) > 6:
        st.caption(f"ve {len(sulanan) - 6} nokta daha (tam liste aşağıdaki onay listesinde)")


def plan_matrisi(plan):
    d = plan[["park", "saat", "su_m3", "utci_dusus_c"]].copy()
    d["saat_et"] = d["saat"].map(lambda s: f"{s:02d}:00")
    d["etiket"] = d["su_m3"].map(lambda v: f"{v:.0f}" if v >= 0.5 else ("•" if v > 0.001 else ""))
    taban = alt.Chart(d).encode(x=alt.X("saat_et:O", title="Saat"), y=alt.Y("park:N", title=None))
    hucre = taban.mark_rect().encode(
        color=alt.Color("su_m3:Q", title="Su (m³)", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("park:N", title="Nokta"), alt.Tooltip("saat_et:O", title="Saat"),
                 alt.Tooltip("su_m3:Q", title="Su (m³)", format=".2f"),
                 alt.Tooltip("utci_dusus_c:Q", title="UTCI düşüşü (°C)", format=".2f")])
    yazi = taban.mark_text(fontSize=11).encode(
        text="etiket:N", color=alt.condition("datum.su_m3 > 8", alt.value("white"), alt.value("black")))
    return (hucre + yazi).properties(height=40 * d["park"].nunique() + 30)


def park_ozet_tablosu(parklar, plan, tesisler):  # noqa: tüm noktalar
    satirlar = []
    for _, p in parklar.iterrows():
        g = plan[plan["park"] == p["park"]]
        saatler = g.loc[g["su_m3"] > 0.001, "saat"]
        r = {"Nokta": p["park"], "Yöntem": YONTEMLER[p["tur"]][0], "İlçe": p["ilce"],
             "Günlük su (m³)": round(float(g["su_m3"].sum()), 2),
             "Çalışma saatleri": f"{saatler.min():02d}:00–{saatler.max() + 1:02d}:00" if len(saatler) else "–",
             "En yüksek UTCI (°C)": round(float(g["utci_c"].max()), 1),
             "Erişim (kişi/saat, temsili)": int(g["kisi"].iloc[0]),
             "Kişi·°C·saat": round(float(g["kisi_derece_saat"].sum()), 0)}
        if tesisler is not None:
            d = mesafe_m(p["lat"], p["lon"], tesisler["lat"].values, tesisler["lon"].values) <= YURUME_5DK_M
            for k in ("Okul / kreş", "Sağlık", "Huzurevi"):
                r[f"{k} (400 m)"] = int(((tesisler["kategori"] == k).values & d).sum())
        satirlar.append(r)
    return pd.DataFrame(satirlar).sort_values("Günlük su (m³)", ascending=False)


# ---------------------------------------------------------------------------
# TEMATİK HARİTA (klasik baskı düzeni)
# Altıgen değerleri ~60 m'lik ızgaraya aktarılır ve Gauss süzgeciyle yumuşatılır (σ ≈ 180 m); sınıf alanları
# eş değer eğrileriyle (contourf) çizilir. Yumuşatma yalnızca görseldir: hesaplar altıgen değerleriyle yapılır.
# ---------------------------------------------------------------------------
import io  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

IZGARA_DERECE = 0.0006        # ~55–65 m
YUMUSATMA_PIKSEL = 3.0        # Gauss σ (piksel) ≈ 180 m
ENLEM_ORTA = 36.90
KM_DERECE_BOYLAM = 1 / (111.32 * np.cos(np.radians(ENLEM_ORTA)))
# Yönlendirme etiketleri: yaklaşık konumlar (ilçe sınırı verisi değildir)
YER_ADLARI = [("KEPEZ", 36.935, 30.700, "ilce"), ("MURATPAŞA", 36.872, 30.735, "ilce"),
              ("KONYAALTI", 36.885, 30.632, "ilce"), ("DÖŞEMEALTI", 36.968, 30.660, "komsu"),
              ("AKSU", 36.930, 30.798, "komsu"), ("ANTALYA KÖRFEZİ", 36.855, 30.690, "deniz")]
TESIS_SEMBOL = {"Okul / kreş": ("D", "#111111", 14), "Sağlık": ("P", "#d62728", 22),
                "Huzurevi": ("h", "#2e7d32", 22), "İçme suyu": ("o", "#1565c0", 9)}
NOKTA_SEMBOL = {"park": ("o", "#2e7d32", 95), "sisleme": ("^", "#00838f", 85), "islatma": ("s", "#6a1b9a", 70)}
KAPALI_RENK = "#bdbdbd"


@st.cache_data(show_spinner=False)
def su_hexleri_oku(yol: str, degistirilme: float):
    """(su altıgenleri, kıyı karışık altıgenleri)"""
    if not os.path.exists(yol):
        return (), ()
    su, karisik = kiyi_siniflari(pd.read_csv(yol, usecols=["hex", "lst_c"]).dropna())
    return tuple(sorted(su)), tuple(sorted(karisik))


@st.cache_data(show_spinner=False)
def izgara_hazirla(kara_hex: tuple, su_hex: tuple, kiyi_hex: tuple = ()):
    """Izgara noktalarının hangi kara altıgenine düştüğü (−1: yok, −2: kıyı karışık, değer yok) ve deniz maskesi."""
    res = h3.get_resolution(kara_hex[0])
    merkez = np.array([h3.cell_to_latlng(h) for h in kara_hex + su_hex + kiyi_hex])
    pay = 0.006
    lat = np.arange(merkez[:, 0].min() - pay, merkez[:, 0].max() + pay, IZGARA_DERECE)
    lon = np.arange(merkez[:, 1].min() - pay, merkez[:, 1].max() + pay, IZGARA_DERECE)
    LON, LAT = np.meshgrid(lon, lat)
    hucre = [h3.latlng_to_cell(a, o, res) for a, o in zip(LAT.ravel(), LON.ravel())]
    konum = {h: i for i, h in enumerate(kara_hex)}
    su = set(su_hex)
    kiyi = set(kiyi_hex)
    idx = np.array([konum.get(c, -2 if c in kiyi else -1) for c in hucre]).reshape(LAT.shape)
    deniz = np.array([c in su for c in hucre], dtype=float).reshape(LAT.shape)
    return lon, lat, idx, deniz


def _bulanik(a, s=YUMUSATMA_PIKSEL):
    r = int(3 * s)
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * s * s))
    k /= k.sum()
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def _yumusak(degerler, idx, gecerli=None):
    """Altıgen değerini ızgaraya yayar ve NaN'a duyarlı biçimde yumuşatır."""
    degerler = np.asarray(degerler, dtype=float)
    var = idx >= 0
    v = np.where(var, degerler[np.clip(idx, 0, None)], 0.0)
    if gecerli is not None:
        var = var & np.asarray(gecerli)[np.clip(idx, 0, None)]
    v = np.where(var, np.nan_to_num(v), 0.0)
    w = _bulanik(var.astype(float))
    return np.where(w > 1e-3, _bulanik(v) / np.maximum(w, 1e-3), np.nan)


def _dms(deger, eksen):
    d = int(abs(deger))
    m_tam = (abs(deger) - d) * 60
    m = int(m_tam)
    s = int(round((m_tam - m) * 60))
    if s == 60:
        m, s = m + 1, 0
    return f"{d}°{m:02d}′{s:02d}″{'K' if eksen == 'y' else 'D'}"


def _halo(renk="white", w=2.6):
    return [pe.withStroke(linewidth=w, foreground=renk)]


def tematik_harita(saat, hava, yuzey, parklar, plan, mod, secenekler, tesisler, gun_adi):
    """Klasik tematik harita (PNG bayt) ve özet istatistikler."""
    satir = hava[hava["saat"] == saat].iloc[0]
    su_hex, kiyi_hex = su_hexleri_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV) if os.path.exists(ISI_YUZEYI_CSV) else 0)
    lon, lat, idx, deniz = izgara_hazirla(tuple(yuzey["hex"]), su_hex, kiyi_hex)
    LON, LAT = np.meshgrid(lon, lat)
    kara_w = _bulanik((idx != -1).astype(float))   # kıyı karışık altıgenler kara sınırına dahil, değeri komşulardan
    kara = kara_w > 0.5
    deniz_w = _bulanik(deniz)
    deniz_m = (deniz_w > 0.35) & ~kara

    utci_once_hex = yuzey_utci(yuzey, satir)
    # Soğuma ızgarada hesaplanır (noktalara yakın ince desen kaybolmasın)
    sog = np.zeros(LAT.shape)
    if plan is not None:
        nokta = pd.DataFrame({"lat": LAT[kara], "lon": LON[kara]})
        sog[kara] = mahalle_sogumasi(nokta, parklar, plan, saat)
    sog_hex = mahalle_sogumasi(yuzey, parklar, plan, saat)
    hex_alan = float(np.mean([h3.cell_area(h, unit="km^2") for h in yuzey["hex"].iloc[:50]]))
    ist = {"alan_km2": float(np.sum(sog_hex >= 0.1) * hex_alan), "azami": float(sog.max()),
           "utci_min": float(np.nanmin(utci_once_hex)), "utci_max": float(np.nanmax(utci_once_hex))}

    plt.rcParams["hatch.color"] = "#555555"
    plt.rcParams["hatch.linewidth"] = 0.6
    fig = plt.figure(figsize=(10.4, 6.7), dpi=140)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.065, 0.085, 0.62, 0.86])
    ax.set_facecolor("white")
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_aspect(1 / np.cos(np.radians(ENLEM_ORTA)))

    lejant, basliklar = [], []

    def grup(ad):
        basliklar.append(ad)
        lejant.append(Line2D([], [], linestyle="none", label=ad))

    # --- Tema katmanı ---
    if mod == "Öncelik" and "oncelik" in yuzey:
        yer = yuzey["yerlesim"].values.astype(bool)
        alan = _yumusak(yuzey["oncelik"].values, idx, gecerli=yer)
        yer_w = _yumusak(yer.astype(float), idx)
        sinirlar = np.quantile(yuzey.loc[yer, "oncelik"], [0.2, 0.4, 0.6, 0.8]) if yer.sum() >= 5 else [0.2, 0.4, 0.6, 0.8]
        seviyeler = [-1.0, *sinirlar, 2.0]
        renkler = [tuple(c / 255 for c in SINIF_RENK[i]) for i in range(1, 6)]
        z = np.where(kara & (yer_w >= 0.5), alan, np.nan)
        ax.contourf(LON, LAT, z, levels=seviyeler, colors=renkler, zorder=1)
        ax.contour(LON, LAT, z, levels=sinirlar, colors="black", linewidths=0.35, zorder=2)
        seyrek = np.where(kara & (yer_w < 0.5), 1.0, np.nan)
        ax.contourf(LON, LAT, seyrek, levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=1)
        ax.contour(LON, LAT, np.where(kara, yer_w, np.nan), levels=[0.5], colors="black", linewidths=0.5, zorder=2)
        baslik = "SOĞUTMA ÖNCELİĞİ HARİTASI"
        alt_baslik = "Ağırlıklı çakıştırma; sınıflar yerleşim alanının beşte birlik dilimleri"
        grup("SOĞUTMA ÖNCELİĞİ")
        for i in range(5, 0, -1):
            lejant.append(Patch(facecolor=renkler[i - 1], edgecolor="black", linewidth=0.5, label=SINIF_ADLARI[i - 1]))
        lejant.append(Patch(facecolor="white", edgecolor="black", hatch="////", linewidth=0.5,
                            label="Seyrek yerleşim (sıralama dışı)"))
    elif mod == "Soğuma":
        seviyeler = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 99]
        renkler = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
        sog_g = _bulanik(sog, 1.0)
        z = np.where(kara & (sog_g >= 0.05), sog_g, np.nan)
        if np.isfinite(z).any():
            ax.contourf(LON, LAT, z, levels=seviyeler, colors=renkler, zorder=1)
            ax.contour(LON, LAT, z, levels=seviyeler[:-1], colors="black", linewidths=0.35, zorder=2)
        baslik = "SOĞUMA ETKİSİ HARİTASI"
        alt_baslik = f"{saat:02d}:00 · planın yarattığı tahmini UTCI düşüşü"
        grup("UTCI DÜŞÜŞÜ (°C)")
        etiket = ["2'den fazla", "1–2", "0,5–1", "0,25–0,5", "0,1–0,25", "0,05–0,1"]
        for r_, e_ in zip(renkler[::-1], etiket):
            lejant.append(Patch(facecolor=r_, edgecolor="black", linewidth=0.5, label=e_))
    else:
        alan = _yumusak(utci_once_hex, idx)
        if mod == "Sonra":
            alan = alan - sog
        a2, a98 = np.nanpercentile(utci_once_hex, [2, 98])
        sinif_say = lambda a: int(np.ceil(a98 / a) - np.floor(a2 / a))
        adim = next((a for a in (4.0, 2.0, 1.0, 0.5, 0.2, 0.1) if sinif_say(a) >= 5), 0.1)   # en az 5 sınıf
        alt_s, ust_s = np.floor(a2 / adim) * adim, np.ceil(a98 / adim) * adim
        sinirlar = np.arange(alt_s + adim, ust_s - 1e-9, adim)
        seviyeler = [-99, *sinirlar, 99]
        n = len(seviyeler) - 1
        cmap = plt.get_cmap("YlOrRd")
        renkler = [cmap(0.08 + 0.84 * i / max(n - 1, 1)) for i in range(n)]
        z = np.where(kara, alan, np.nan)
        ax.contourf(LON, LAT, z, levels=seviyeler, colors=renkler, zorder=1)
        if len(sinirlar):
            ax.contour(LON, LAT, z, levels=sinirlar, colors="black", linewidths=0.35, zorder=2)
        baslik = "ISI STRESİ HARİTASI (UTCI)" + ("\nPLAN UYGULANDIKTAN SONRA" if mod == "Sonra" else "")
        alt_baslik = f"{saat:02d}:00 · hissedilen sıcaklık, " + ("plan uygulanmış" if mod == "Sonra" else "mevcut durum")
        grup("UTCI (°C)")
        f_ = (lambda x: f"{x:.1f}".replace(".", ",")) if adim < 1 else (lambda x: f"{x:.0f}")
        etiketler = ([f"{f_(sinirlar[-1])} üstü"] +
                     [f"{f_(a)}–{f_(b)}" for a, b in zip(sinirlar[::-1][1:], sinirlar[::-1][:-1])] +
                     [f"{f_(sinirlar[0])} altı"]) if len(sinirlar) else [f"{f_(alt_s)}–{f_(ust_s)}"]
        for r_, e_ in zip(renkler[::-1], etiketler):
            lejant.append(Patch(facecolor=r_, edgecolor="black", linewidth=0.5, label=e_))

    # --- Deniz ve çalışma alanı sınırı ---
    ax.contourf(LON, LAT, np.where(deniz_m, 1.0, np.nan), levels=[0.5, 1.5], colors=["#d6ecf8"], zorder=0.5)
    ax.contourf(LON, LAT, np.where(deniz_m, 1.0, np.nan), levels=[0.5, 1.5], colors="none", hatches=["---"],
                zorder=0.6)
    ax.contour(LON, LAT, kara_w, levels=[0.5], colors="black", linewidths=2.0, zorder=3)

    # --- Ek katmanlar ---
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        aci = np.linspace(0, 2 * np.pi, 120)
        for _, p in parklar[parklar["tur"] == "park"].iterrows():
            for r_m, stil in ((YURUME_5DK_M, (0, (5, 3))), (YURUME_10DK_M, (0, (1.5, 2)))):
                ax.plot(p["lon"] + r_m / 1000 * KM_DERECE_BOYLAM * np.cos(aci),
                        p["lat"] + r_m / 110_540 * np.sin(aci), color="#1b5e20", lw=1.0, ls=stil, zorder=4)
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and len(tesisler):
        grup("TESİSLER (OpenStreetMap)")
        for kat, (m_, r_, s_) in TESIS_SEMBOL.items():
            t_ = tesisler[tesisler["kategori"] == kat]
            if len(t_):
                ax.scatter(t_["lon"], t_["lat"], marker=m_, c=r_, s=s_, linewidths=0.3, edgecolors="white", zorder=5)
                lejant.append(Line2D([], [], marker=m_, color=r_, linestyle="none", markersize=6, label=kat))
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        grup("PARK ERİŞİMİ")
        lejant.append(Line2D([], [], color="#1b5e20", ls=(0, (5, 3)), lw=1.2, label="5 dk yürüme (400 m)"))
        lejant.append(Line2D([], [], color="#1b5e20", ls=(0, (1.5, 2)), lw=1.2, label="10 dk yürüme (800 m)"))

    # --- Müdahale noktaları ---
    su_saat = {}
    if plan is not None:
        su_saat = plan[plan["saat"] == saat].set_index("park")["su_m3"].to_dict()
    grup("MÜDAHALE NOKTALARI")
    for tur_, (m_, r_, s_) in NOKTA_SEMBOL.items():
        p_ = parklar[parklar["tur"] == tur_]
        if not len(p_):
            continue
        acik = np.array([su_saat.get(a, 0) > 0.001 for a in p_["park"]])
        renk = [r_ if a else KAPALI_RENK for a in acik]
        ax.scatter(p_["lon"], p_["lat"], marker=m_, c=renk, s=s_, edgecolors="black", linewidths=0.9, zorder=7)
        lejant.append(Line2D([], [], marker=m_, color=r_, markeredgecolor="black", linestyle="none", markersize=8,
                             label=YONTEMLER[tur_][0] + (" (pilot park)" if tur_ == "park" else " (aday)")))
    lejant.append(Line2D([], [], marker="o", color=KAPALI_RENK, markeredgecolor="black", linestyle="none",
                         markersize=8, label="Bu saatte kapalı"))
    if "Nokta adları" in secenekler:
        for _, p in parklar.iterrows():
            ad = p["park"] if p["tur"] == "park" else ("S" if p["tur"] == "sisleme" else "I") + p["park"].split()[1]
            ax.annotate(ad, (p["lon"], p["lat"]), xytext=(0, 8), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.6 if p["tur"] == "park" else 7, fontweight="bold",
                        color="#1b3d1f" if p["tur"] == "park" else "#222222", path_effects=_halo(), zorder=8)

    # --- Yer adları ---
    for ad, la, lo, tur_ in YER_ADLARI:
        if not (lon.min() < lo < lon.max() and lat.min() < la < lat.max()):
            continue
        if tur_ == "deniz":
            ax.text(lo, la, " ".join(ad), fontsize=10, style="italic", color="#1f5f8b", ha="center", va="center",
                    zorder=6, path_effects=_halo("#d6ecf8", 3))
        else:
            ax.text(lo, la, " ".join(ad), fontsize=10.5 if tur_ == "ilce" else 9, fontweight="bold",
                    color="#333333" if tur_ == "ilce" else "#777777", ha="center", va="center", zorder=6,
                    path_effects=_halo())

    # --- Koordinat çerçevesi (derece-dakika-saniye) ---
    xt = np.arange(np.ceil(lon.min() / 0.03) * 0.03, lon.max(), 0.03)
    yt = np.arange(np.ceil(lat.min() / 0.02) * 0.02, lat.max(), 0.02)
    xt = xt[(xt > lon.min() + 0.012) & (xt < lon.max() - 0.012)]   # köşelerde etiket çakışmasın
    yt = yt[(yt > lat.min() + 0.008) & (yt < lat.max() - 0.008)]
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([_dms(v, "x") for v in xt], fontsize=7)
    ax.set_yticklabels([_dms(v, "y") for v in yt], fontsize=7, rotation=90, va="center")
    ax.tick_params(direction="out", length=4, top=True, right=True, labeltop=True, labelright=True)
    for s_ in ax.spines.values():
        s_.set_linewidth(1.4)

    # --- Kuzey oku ve ölçek ---
    ax.add_patch(Rectangle((0.015, 0.045), 0.05, 0.19, transform=ax.transAxes, facecolor="white", edgecolor="black",
                           lw=0.6, zorder=8.5))
    ax.annotate("", xy=(0.04, 0.17), xytext=(0.04, 0.07), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=1.0", lw=1.6, color="black"), zorder=9)
    ax.text(0.04, 0.18, "K", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold",
            zorder=9, path_effects=_halo())
    x0 = lon.min() + 0.12 * (lon.max() - lon.min())
    y0 = lat.min() + 0.035 * (lat.max() - lat.min())
    h_ = 0.0022
    ax.add_patch(Rectangle((x0 - 0.004, y0 - 0.003), 4 * KM_DERECE_BOYLAM + 0.011, h_ + 0.0095, facecolor="white",
                           edgecolor="black", lw=0.6, zorder=8.5))
    for i in range(4):
        ax.add_patch(Rectangle((x0 + i * KM_DERECE_BOYLAM, y0), KM_DERECE_BOYLAM, h_, facecolor="black" if i % 2 == 0
                               else "white", edgecolor="black", lw=0.8, zorder=9))
        ax.text(x0 + i * KM_DERECE_BOYLAM, y0 + h_ * 1.6, str(i), ha="center", fontsize=7, zorder=9,
                path_effects=_halo())
    ax.text(x0 + 4 * KM_DERECE_BOYLAM, y0 + h_ * 1.6, "4 km", ha="center", fontsize=7, zorder=9, path_effects=_halo())

    # --- Sağ panel: başlık kutusu ve İŞARETLER ---
    pan = fig.add_axes([0.735, 0.085, 0.255, 0.86])
    pan.axis("off")
    pan.add_patch(Rectangle((0, 0.775), 1, 0.225, transform=pan.transAxes, facecolor="white", edgecolor="#b71c1c",
                            lw=2.2))
    pan.text(0.05, 0.965, "COOLSCAPE AI", transform=pan.transAxes, fontsize=15, fontweight="bold", color="#b71c1c",
             va="top")
    pan.text(0.05, 0.915, baslik, transform=pan.transAxes, fontsize=9.6, fontweight="bold", va="top", wrap=True)
    import textwrap
    alt_metin = textwrap.fill(alt_baslik, 46) + "\n" + gun_adi.split(" · ")[0]
    pan.text(0.05, 0.915 - 0.03 * (baslik.count("\n") + 1) - 0.008, alt_metin,
             transform=pan.transAxes, fontsize=7.4,
             va="top", color="#333333", linespacing=1.4)
    grup("DİĞER")
    lejant.append(Patch(facecolor="#d6ecf8", edgecolor="#555555", hatch="---", linewidth=0.5, label="Deniz"))
    lejant.append(Line2D([], [], color="black", lw=2.0, label="Çalışma alanı sınırı"))
    leg = pan.legend(handles=lejant, loc="upper left", bbox_to_anchor=(0, 0.755), frameon=True, fancybox=False,
                     edgecolor="black", fontsize=7.6, title="İŞARETLER", title_fontproperties={"weight": "bold",
                                                                                             "size": 11},
                     alignment="left", borderpad=0.9, labelspacing=0.55, handlelength=2.0)
    leg.get_frame().set_linewidth(1.3)
    for t_ in leg.get_texts():
        if t_.get_text() in basliklar:
            t_.set_fontweight("bold")
            t_.set_fontsize(7.4)

    fig.text(0.065, 0.018,
             f"Kaynak: Landsat 8/9 C2 L2 yüzey sıcaklığı ({yuzey['landsat_tarih'].iloc[0] if 'landsat_tarih' in yuzey else '?'}); "
             "hava Open-Meteo; tesisler © OpenStreetMap katkıcıları (ODbL); "
             + ("nüfus WorldPop 2025 (CC BY 4.0; 65+ ilçe oranıyla). " if "nufus" in yuzey else "") +
             f"UTCI; güneşin MRT katkısı: {hava.attrs.get('mrt_yontem', '?')}. "
             "Sınıf alanları altıgen değerlerinin ~180 m yumuşatılmış hâlidir. İlçe adları yaklaşık konumdadır.",
             fontsize=5.6, color="#444444", wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor="white")
    plt.close(fig)
    return buf.getvalue(), ist


# ---------------------------------------------------------------------------
# CANLI HARİTA: sokak (OpenStreetMap) ve uydu (Esri) altlığı üzerinde sürekli renk geçişli alan + eş değer çizgileri
# Altıgen değerleri ~60 m ızgarada ~180 m Gauss süzgeciyle yumuşatılır ve yarı saydam katman olarak altlığa bindirilir.
# Yumuşatma yalnızca görseldir; hesaplar altıgen değerleriyle yapılır. Üzerine gelince altıgenin gerçek değeri görünür.
# ---------------------------------------------------------------------------
import base64  # noqa: E402

import folium  # noqa: E402
import branca.colormap as bcm  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

CANLI_SAYDAMLIK = 0.58
CANLI_BUYUTME = 3          # yarı saydam katman 3 kat yoğun ızgaraya doğrusal aradeğerlenir: kıyı kenarı basamaklı görünmez
UTCI_ACIKLIK_C = 4.0       # UTCI renk ölçeğinin en dar genişliği (°C)
UTCI_CIZGI_MIN_C = 0.5     # UTCI eş değer çizgilerinin en küçük aralığı (°C)
ALTLIKLAR = {
    "Uydu (Esri World Imagery)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
                                  "{z}/{y}/{x}",
                                  "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, "
                                  "Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"),
    "Sokak (OpenStreetMap)": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                              "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> katkıcıları"),
}
UYDU_ETIKETLER = [   # uydu üzerinde yol ve yer adları (Esri referans katmanları, saydam)
    ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
     "Yollar"),
    ("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/"
     "{z}/{y}/{x}", "Yer adları"),
]
NOKTA_RENK = {"park": "#2e7d32", "sisleme": "#00838f", "islatma": "#6a1b9a"}


def _guzel_adim(aralik, hedef=8):
    for a in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0):
        if aralik / a <= hedef:
            return a
    return 10.0


def canli_alan(mod, saat, hava, yuzey, parklar, plan):
    """Izgarada sürekli alan (NaN: kara dışı ya da değer yok), altıgen değerleri ve gösterim bilgisi."""
    su_hex, kiyi_hex = su_hexleri_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV)
                                      if os.path.exists(ISI_YUZEYI_CSV) else 0)
    lon, lat, idx, _ = izgara_hazirla(tuple(yuzey["hex"]), su_hex, kiyi_hex)
    LON, LAT = np.meshgrid(lon, lat)
    kara = _bulanik((idx != -1).astype(float)) > 0.5
    satir = hava[hava["saat"] == saat].iloc[0]
    if mod == "LST":
        hexd = yuzey["lst_c"].values
        z = _yumusak(hexd, idx)
        bilgi = {"baslik": "Yüzey sıcaklığı (°C)", "cmap": "RdYlBu_r", "birim": "°C",
                 "not": f"Landsat 8/9 ölçümü, {yuzey['landsat_tarih'].iloc[0] if 'landsat_tarih' in yuzey else '?'}, "
                        "yerel saatle ~11:20 geçiş; o anın yüzey sıcaklığı (saat seçiminden bağımsız)"}
    elif mod == "Öncelik" and "oncelik" in yuzey:
        yer = yuzey["yerlesim"].values.astype(bool)
        hexd = np.where(yer, yuzey["oncelik"].values, np.nan)
        z = _yumusak(yuzey["oncelik"].values, idx, gecerli=yer)
        z = np.where(_yumusak(yer.astype(float), idx) >= 0.5, z, np.nan)
        bilgi = {"baslik": "Soğutma önceliği (0–1)", "cmap": "YlOrRd", "birim": "",
                 "not": "Ağırlıklı çakıştırma; seyrek yerleşim boş bırakıldı"}
    else:
        u = yuzey_utci(yuzey, satir)
        sog = np.zeros(LAT.shape)
        if plan is not None and mod in ("Sonra", "Soğuma"):
            sog[kara] = mahalle_sogumasi(pd.DataFrame({"lat": LAT[kara], "lon": LON[kara]}), parklar, plan, saat)
        if mod == "Soğuma":
            hexd = mahalle_sogumasi(yuzey, parklar, plan, saat)
            z = _bulanik(sog, 1.0)
            z = np.where(z >= 0.05, z, np.nan)
            bilgi = {"baslik": f"Soğuma etkisi, {saat:02d}:00 (UTCI düşüşü, °C)", "cmap": "Blues", "birim": "°C",
                     "cmin": 0.35,   # en küçük etki de uydu üzerinde seçilsin: ölçek açık maviden başlar
                     "not": "Planın yarattığı tahmini UTCI düşüşü; 0,05 °C altı gösterilmez. Etki noktanın çevresinde "
                            "yereldir (park ~150 m, ıslatma ~60 m, sisleme ~15 m ölçekle azalır; varsayım); görmek için "
                            "bir noktaya yakınlaştırın"}
        else:
            hexd = u - (mahalle_sogumasi(yuzey, parklar, plan, saat) if mod == "Sonra" else 0)
            z = _yumusak(u, idx) - (sog if mod == "Sonra" else 0)
            bilgi = {"baslik": f"UTCI, {saat:02d}:00 (°C){' · plan uygulanmış' if mod == 'Sonra' else ''}",
                     "cmap": "YlOrRd", "birim": "°C",
                     "not": f"Kent geneli {kent_utci(satir):.1f} °C ({utci_sinif(kent_utci(satir))[0].lower()} ısı stresi); "
                            "mekânsal fark yalnızca yüzey sıcaklığının MRT'ye katkısından gelir"}
    z = np.where(kara, z, np.nan)
    gecerli = hexd[np.isfinite(hexd)]
    if mod == "Soğuma":
        vmin, vmax = 0.0, max(float(np.nanmax(z)) if np.isfinite(z).any() else 0.1, 0.1)
    else:
        vmin, vmax = (float(v) for v in np.nanpercentile(gecerli, [2, 98]))
        if mod in ("Önce", "Sonra"):
            # Kent içi UTCI farkı ~1 °C; ölçek en az UTCI_ACIKLIK_C genişlikte tutulur ki küçük fark büyük görünmesin
            orta = (vmin + vmax) / 2
            yari = max((vmax - vmin) / 2, UTCI_ACIKLIK_C / 2)
            vmin, vmax = orta - yari, orta + yari
        if vmax - vmin < 1e-6:
            vmax = vmin + 0.1
    adim = max(_guzel_adim(vmax - vmin), UTCI_CIZGI_MIN_C if mod in ("Önce", "Sonra") else 0)
    # ölçek uçları çizgi adımına yuvarlanır: lejant okunaklı, tekrar eden etiket yok
    vmin, vmax = np.floor(vmin / adim + 1e-9) * adim, np.ceil(vmax / adim - 1e-9) * adim
    bilgi.update(vmin=float(vmin), vmax=float(vmax), adim=float(adim))
    return lon, lat, z, hexd, bilgi


def _buyut(a, k):
    """2B dizi ya da (satır, sütun, kanal) diziyi k kat yoğun ızgaraya doğrusal aradeğerler."""
    def eksen(x, ax):
        n = x.shape[ax]
        eski, yeni = np.arange(n), np.linspace(0, n - 1, (n - 1) * k + 1)
        return np.apply_along_axis(lambda v: np.interp(yeni, eski, v), ax, x)
    return eksen(eksen(a, 0), 1)


def _renkli_png(z, bilgi):
    cmap = plt.get_cmap(bilgi["cmap"])
    t = np.clip((z - bilgi["vmin"]) / (bilgi["vmax"] - bilgi["vmin"]), 0, 1)
    t = bilgi.get("cmin", 0.0) + (1 - bilgi.get("cmin", 0.0)) * t
    rgba = cmap(np.nan_to_num(t))
    rgba[..., 3] = np.where(np.isfinite(z), CANLI_SAYDAMLIK, 0.0)
    rgba[..., :3] *= rgba[..., 3:4]                    # önceden çarpılmış renk: kenarda gri hâle oluşmaz
    rgba = _buyut(rgba, CANLI_BUYUTME)
    a = rgba[..., 3:4]
    rgba[..., :3] = np.where(a > 1e-6, rgba[..., :3] / np.maximum(a, 1e-6), 0)
    rgba[..., 3] = np.where(a[..., 0] >= CANLI_SAYDAMLIK * 0.5, CANLI_SAYDAMLIK, 0.0)
    rgba = np.clip(rgba, 0, 1)
    buf = io.BytesIO()
    plt.imsave(buf, np.flipud(rgba), format="png")   # satır 0 kuzey olmalı
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _es_deger_cizgileri(lon, lat, z, bilgi):
    """Eş değer (izoterm) çizgileri: [(değer, [(lat, lon), ...]), ...]"""
    if not np.isfinite(z).any():
        return [], None
    adim = bilgi["adim"]
    seviyeler = np.arange(np.ceil(bilgi["vmin"] / adim) * adim, bilgi["vmax"] + 1e-9, adim)
    if bilgi["birim"] == "" or len(seviyeler) == 0:
        return [], adim
    fig, ax = plt.subplots()
    cs = ax.contour(*np.meshgrid(lon, lat), np.ma.masked_invalid(z), levels=seviyeler)
    cizgiler = []
    for sev, parcalar in zip(cs.levels, cs.allsegs):
        for p in parcalar:
            if len(p) >= 10:
                cizgiler.append((float(sev), [(float(y), float(x)) for x, y in p]))
    plt.close(fig)
    return cizgiler, adim


def canli_harita(mod, saat, hava, yuzey, parklar, plan, secenekler, tesisler, altlik):
    lon, lat, z, hexd, bilgi = canli_alan(mod, saat, hava, yuzey, parklar, plan)
    d = IZGARA_DERECE / (2 * CANLI_BUYUTME)
    sinir =[[float(lat.min() - d), float(lon.min() - d)], [float(lat.max() + d), float(lon.max() + d)]]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=12, tiles=None, control_scale=True,
                   prefer_canvas=True)
    for ad, (url, atif) in ALTLIKLAR.items():   # yalnızca seçili altlık açık başlar; diğeri katman kutusundan seçilir
        folium.TileLayer(url, attr=atif, name=ad, max_zoom=19, show=(ad == altlik)).add_to(m)
    folium.raster_layers.ImageOverlay(_renkli_png(z, bilgi), bounds=sinir, name="Isı katmanı", opacity=1.0,
                                      interactive=False, zindex=2).add_to(m)
    etiket_pane = folium.map.CustomPane("etiketler", z_index=650, pointer_events=False)
    etiket_pane.add_to(m)
    for url, ad in UYDU_ETIKETLER:   # ısı katmanının ÜSTÜNDE: yol ve yer adları renklerin altında kaybolmaz
        folium.TileLayer(url, attr="Esri", name=ad, overlay=True, control=True, show=True, max_zoom=19,
                         pane="etiketler").add_to(m)

    cizgiler, adim = _es_deger_cizgileri(lon, lat, z, bilgi)
    if cizgiler:
        fmt = (lambda v: f"{v:.1f}".replace(".", ",")) if adim < 1 else (lambda v: f"{v:.0f}")
        grup = folium.FeatureGroup(name="Eş değer çizgileri", show=False)
        etiket_say = 0
        for sev, nokta in sorted(cizgiler, key=lambda c: -len(c[1])):
            folium.PolyLine(nokta, color="#1a1a1a", weight=1.1, opacity=0.75,
                            tooltip=f"{fmt(sev)} {bilgi['birim']}").add_to(grup)
            if len(nokta) >= 40 and etiket_say < 30:   # uzun çizgilere, en çok 30 etiket
                etiket_say += 1
                orta = nokta[len(nokta) // 2]
                folium.Marker(orta, icon=folium.DivIcon(
                    icon_size=(40, 14), icon_anchor=(20, 7),
                    html=f"<div style='font:600 11px Arial;color:#111;text-align:center;"
                         f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff'>{fmt(sev)}</div>")).add_to(grup)
        grup.add_to(m)

    # Üzerine gelince altıgenin gerçek değeri
    lst_fark = yuzey["lst_fark_c"].values
    oz = []
    for h, v, f in zip(yuzey["hex"].values, hexd, lst_fark):
        if not np.isfinite(v):
            continue
        kenar = [[round(b, 6), round(a, 6)] for a, b in h3.cell_to_boundary(h)]
        deger = (f"{v:.2f}" if bilgi["birim"] == "" or mod == "Soğuma" else f"{v:.1f}").replace(".", ",")
        oz.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [kenar + [kenar[0]]]},
                   "properties": {"b": f"{bilgi['baslik']}: {deger} · yüzey farkı {f:+.1f} °C".replace(".", ",")}})
    folium.GeoJson({"type": "FeatureCollection", "features": oz}, name="Altıgen değerleri", control=False,
                   style_function=lambda _: {"fillOpacity": 0, "weight": 0, "color": "#000"},
                   highlight_function=lambda _: {"weight": 1, "color": "#000", "fillOpacity": 0.05},
                   tooltip=folium.GeoJsonTooltip(fields=["b"], labels=False)).add_to(m)

    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        g = folium.FeatureGroup(name="Yürüme mesafesi (400 / 800 m)")
        for _, p in parklar[parklar["tur"] == "park"].iterrows():
            for r_, cizgi in ((YURUME_5DK_M, "6 4"), (YURUME_10DK_M, "2 4")):
                folium.Circle([p["lat"], p["lon"]], radius=r_, color="#1b5e20", weight=1.5, fill=False,
                              dash_array=cizgi).add_to(g)
        g.add_to(m)
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and len(tesisler):
        g = folium.FeatureGroup(name="Tesisler (OpenStreetMap)")
        renk = {"Okul / kreş": "#111111", "Sağlık": "#d62728", "Huzurevi": "#2e7d32", "İçme suyu": "#1565c0"}
        for _, t in tesisler.iterrows():
            folium.CircleMarker([t["lat"], t["lon"]], radius=3, color="#ffffff", weight=0.8, fill=True,
                                fill_color=renk.get(t["kategori"], "#555"), fill_opacity=1,
                                tooltip=f"{t['kategori']}: {t['ad']}").add_to(g)
        g.add_to(m)

    su = plan[plan["saat"] == saat].set_index("park") if plan is not None else None
    g = folium.FeatureGroup(name="Müdahale noktaları")
    for _, p in parklar.iterrows():
        acik = su is not None and float(su["su_m3"].get(p["park"], 0)) > 0.001
        ipucu = f"<b>{p['park']}</b><br>{YONTEMLER[p['tur']][0]}"
        if su is not None and p["park"] in su.index:
            r = su.loc[p["park"]]
            ipucu += (f"<br>{saat:02d}:00 · {r['su_m3']:.2f} m³ · UTCI {r['utci_c']:.1f} → {r['utci_sonra_c']:.1f} °C"
                      .replace(".", ","))
        folium.CircleMarker([p["lat"], p["lon"]], radius=8 if p["tur"] == "park" else 6, color="#000", weight=1.5,
                            fill=True, fill_color=NOKTA_RENK[p["tur"]] if acik or su is None else "#9e9e9e",
                            fill_opacity=1, tooltip=ipucu).add_to(g)
        if "Nokta adları" in secenekler:
            ad = p["park"] if p["tur"] == "park" else ("S" if p["tur"] == "sisleme" else "I") + p["park"].split()[1]
            folium.Marker([p["lat"], p["lon"]], icon=folium.DivIcon(
                icon_size=(160, 16), icon_anchor=(80, -10) if p["tur"] == "park" else (80, 22),
                html=f"<div style='font:700 11px Arial;color:#0d2b12;white-space:nowrap;"
                     "text-align:center;"
                     f"text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff'>{ad}</div>")).add_to(g)
    g.add_to(m)

    renkler = [matplotlib.colors.to_hex(plt.get_cmap(bilgi["cmap"])(x))
               for x in np.linspace(bilgi.get("cmin", 0.0), 1, 9)]
    adim_l = bilgi["adim"]
    while (bilgi["vmax"] - bilgi["vmin"]) / adim_l > 8:
        adim_l *= 2
    isaret = [round(float(v), 3) for v in np.arange(bilgi["vmin"], bilgi["vmax"] + 1e-9, adim_l)]
    lejant = bcm.LinearColormap(renkler, vmin=round(bilgi["vmin"], 3), vmax=round(bilgi["vmax"], 3),
                                tick_labels=isaret if len(isaret) >= 2 else None)
    lejant.caption = bilgi["baslik"]
    lejant.add_to(m)
    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    m.fit_bounds(sinir)
    return m, bilgi


def canli_harita_goster(alan, m, yukseklik=640):
    html = m.get_root().render()   # kendi ürettiğimiz Folium HTML'i (dış girdi değil)
    with alan:
        if hasattr(st, "iframe"):    # Streamlit ≥1.5x: components.html kullanımdan kalkıyor
            st.iframe(html, height=yukseklik)
        else:
            components.html(html, height=yukseklik)


# ---------------------------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CoolScape AI", layout="wide", initial_sidebar_state="expanded")
for anahtar, deger in {"saat": 14, "oynuyor": False}.items():
    st.session_state.setdefault(anahtar, deger)

st.title("CoolScape AI")
st.markdown("Kentsel serinlik ve su tahsis motoru · Antalya Muratpaşa–Kepez pilotu")

# --- Girdiler: görünürde yalnızca üç karar, gerisi gelişmiş ayarlarda ---
st.sidebar.header("Günün planı")
gun = st.sidebar.selectbox("Gün", list(GUNLER))
butce = st.sidebar.number_input("Günlük su bütçesi (m³)", value=300, step=25, min_value=25,
                                help="08:00–20:00 arasında parklara, sisleme noktalarına ve yol ıslatmaya dağıtılacak toplam su.")
on_ayar = st.sidebar.radio("Öncelik", list(ON_AYARLAR), help="Hazır ayar. Ayrıntıları aşağıdan değiştirebilirsiniz.")
v = ON_AYARLAR[on_ayar]
YONTEM_ETIKET = {v_[0]: k for k, v_ in YONTEMLER.items()}
secili_yontem = st.sidebar.multiselect("Yöntemler", list(YONTEM_ETIKET), default=list(YONTEM_ETIKET),
                                       help="Sisleme ve yol ıslatma noktaları ısı haritasındaki sıcak noktalardan "
                                            "otomatik önerilen adaylardır.")
turler = tuple(sorted({YONTEM_ETIKET[x] for x in secili_yontem} | {"park"}))
with st.sidebar.expander("Gelişmiş ayarlar"):
    kirilgan_pay = st.slider("Kırılgan bölgelere asgari su payı (%)", 0, 80, v["kirilgan_pay"],
                             key=f"pay_{on_ayar}") / 100
    w_kirilganlik = st.slider("Ağırlık: kırılganlık göstergesi", 0.0, 2.0, v["w_k"], 0.1, key=f"wk_{on_ayar}")
    vpd_min = st.slider("Nem kapısı: en düşük VPD (kPa)", 0.0, 2.0, 0.8, 0.1,
                        help="Buhar basıncı açığı bunun altındaysa hava nemlidir, su az serinletir; o saatte sulama yapılmaz.")
    enerji_opt = st.checkbox("Pompalama enerjisini gözet", value=True)
    st.markdown("**Çim güvenliği (park sulama)**")
    park_kesme = st.slider("Akşam kesme saati", 14, 20, 18, format="%d:00",
                           help="Bu saatten sonra park sulanmaz: yaprak gece ıslak kalırsa mantar hastalığı riski artar.")
    park_mm = st.slider("Park başına günlük üst sınır (mm = L/m²)", 2.0, 12.0, 6.0, 0.5,
                        help="Çimin sıcak bir yaz gününde tükettiği suya yakın bir sınır; aşırı sulamayı önler.")
    st.markdown("**Erişim varsayımları (kişi/saat, temsili)**")
    erisim = {"park_carpan": st.slider("Park: nüfus skoru başına kişi", 10, 100, ERISIM_VARSAYILAN["park_carpan"], 5),
              "sisleme": st.slider("Sisleme noktası", 5, 200, ERISIM_VARSAYILAN["sisleme"], 5),
              "islatma": st.slider("Islatılan cadde", 20, 500, ERISIM_VARSAYILAN["islatma"], 10)}
    aritilmis_senaryo = st.checkbox("Arıtılmış su senaryosu (gelecek)", value=False,
                                    help="Altyapı kurulursa hangi parkların arıtılmış suyla beslenebileceğini gösterir.")
    sicak_pay = st.slider("Haritada gösterilecek en sıcak alan (%)", 5, 50, 20, 5) / 100

# --- Veri: kendiliğinden yüklenir ---
tur, tarih = GUNLER[gun]
if tur == "tahmin":
    tarih = (datetime.now(ZoneInfo("Europe/Istanbul")).date() + timedelta(days=1)).isoformat()
try:
    hava = hava_tahmini_cek(tarih) if tur == "tahmin" else hava_verisi_cek(tarih)
    hava_kaynak = f"Open-Meteo tahmini ({tarih})" if tur == "tahmin" else "Open-Meteo arşivi (ERA5 tabanlı)"
except Exception as hata:
    st.error(f"Open-Meteo'ya erişilemedi ({hata}). Uydurma veriyle devam edilmez. Saatlik veriyi CSV olarak "
             "yükleyebilirsiniz (sütunlar: saat, sicaklik_c, bagil_nem, ruzgar_ms, gunes_wm2).")
    yuklenen = st.file_uploader("Saatlik hava verisi (CSV)", type="csv")
    if yuklenen is None:
        st.stop()
    hava = csv_hava_verisi(yuklenen)
    hava_kaynak = "yüklenen CSV"
if not set(SAATLER) <= set(hava["saat"]):
    st.error("Hava verisinde 08:00–20:00 saatlerinin bir kısmı eksik.")
    st.stop()
hava = gunes_mrt_ekle(hava)
mrt_yontem = hava.attrs.get("mrt_yontem", "")

yuzey = isi_yuzeyi_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV) if os.path.exists(ISI_YUZEYI_CSV) else 0)
tesis_yolu = TESIS_CSV if os.path.exists(TESIS_CSV) else TESIS_JSON
tesisler = tesis_dosyasi_oku(tesis_yolu, os.path.getmtime(tesis_yolu) if os.path.exists(tesis_yolu) else 0)
if tesisler is not None:
    osm_hata, osm_kaynak = None, f"OpenStreetMap ({tesis_yolu}), {len(tesisler)} tesis"
else:
    tesisler, osm_hata = osm_tesisleri_cek()
    osm_kaynak = (f"OpenStreetMap (canlı), {len(tesisler)} tesis" if tesisler is not None
                  else f"OpenStreetMap'e erişilemedi; {TESIS_CSV} dosyasını repoya ekleyin")
parklar_g = parklari_guncelle(PARKLAR, yuzey)
nufus = nufus_oku(NUFUS_CSV, os.path.getmtime(NUFUS_CSV) if os.path.exists(NUFUS_CSV) else 0)
obilgi = None
if yuzey is not None:
    yuzey, obilgi = oncelik_hesapla(yuzey, parklar_g, tesisler, nufus)
parklar_g = park_nufus_guncelle(parklar_g, yuzey)
adaylar = aday_noktalar(yuzey, parklar_g)
parklar = pd.concat([parklar_g, adaylar], ignore_index=True)   # tüm müdahale noktaları
parklar, kirilgan_parklar, kirilgan_kaynak = kirilganlik_ata(parklar, tesisler, nufus is not None)
tablo = park_saat_tablosu(hava, parklar)
girdi = girdileri_hazirla(tablo, parklar, w_kirilganlik, erisim)

if yuzey is not None:
    tarih_bilgi = yuzey["landsat_tarih"].iloc[0] if "landsat_tarih" in yuzey else "?"
    yuzey_kaynak = (f"Landsat yüzey sıcaklığı ({tarih_bilgi}, sabit taban harita), {len(yuzey)} kara altıgeni"
                    + (f"; {yuzey.attrs.get('su_altigen')} deniz/su ve {yuzey.attrs.get('kiyi_altigen', 0)} kıyı karışık "
                       "piksel altıgeni analiz dışı" if yuzey.attrs.get("su_altigen") else ""))
else:
    yuzey_kaynak = f"yok ({ISI_YUZEYI_CSV} bulunamadı)"
nufus_k = parklar_g["nufus_kaynak"].iloc[0] if "nufus_kaynak" in parklar_g else "temsili"
temsili = ["hat kapasitesi", "sulanan alan", "kişi/saat erişim katsayıları"]
if kirilgan_kaynak.startswith("temsili"):
    temsili.insert(0, "kırılgan nüfus payı")
st.caption(f"Hava: {hava_kaynak} · Güneşin MRT katkısı: {mrt_yontem} · Isı yüzeyi: {yuzey_kaynak} · Tesisler: {osm_kaynak} · "
           f"Nüfus: {'WorldPop 2025 toplam nüfus (' + NUFUS_CSV + '); 65+ yalnızca ilçe oranıyla' if nufus is not None else 'nüfus dosyası yok; park nüfusu ' + nufus_k} · "
           f"Temsili: {', '.join(temsili)} · Etki eğrileri Paris, Melbourne, Adelaide ve Arizona saha ölçümlerine göre "
           "ölçeklendi; Antalya saha doğrulaması 2027 pilotunda.")

# --- Plan: ayar değişince kendiliğinden hesaplanır ---
plan, esit, ozet, durum = plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                                       turler, park_kesme, park_mm)
if plan is None:
    st.error(f"Plan bulunamadı (çözücü durumu: {durum}). Gelişmiş ayarlarda kırılgan payı veya nem kapısını gevşetin.")
    st.stop()

if not ozet["isi_stresi_var"]:
    st.info("Bu gün hiçbir saatte güçlü ısı stresi (UTCI ≥ 32 °C) beklenmiyor; soğutma sulaması önerilmiyor.")
else:
    pay = lambda d: d.loc[d["park"].isin(kirilgan_parklar), "su_m3"].sum() / max(d["su_m3"].sum(), 1e-9) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Serinlik kazancı", f"{ozet['artis'] * 100:+.0f}%", "gündüz eşit park sulamasına göre", delta_color="off",
              help="Öncelik ağırlıklı UTCI düşüşü, aynı su miktarıyla. Model simülasyonudur, saha ölçümü değildir.")
    alt_satir = []
    if ozet["artis_aralik"]:
        a, b = ozet["artis_aralik"]
        alt_satir.append(f"Belirsizlik aralığı {a * 100:+.0f}% … {b * 100:+.0f}%")
    if ozet["yalniz_park"] is not None:
        alt_satir.append(f"Yeni yöntem eklemeden, yalnız park sulamasının saat ve yer seçimiyle {ozet['yalniz_park'] * 100:+.0f}%")
    if alt_satir:
        m1.caption(" · ".join(alt_satir))
    m2.metric("Serinlik (kişi·°C·saat)", f"{ozet['kds']:,.0f}".replace(",", "."),
              f"eşit dağıtım {ozet['kds_esit']:,.0f}".replace(",", "."), delta_color="off",
              help="Her saat, ulaşılan kişi sayısı × UTCI düşüşü toplamı (yalnızca güçlü ısı stresi saatleri). "
                   "Erişim sayıları temsilidir.")
    m3.metric("Kullanılan su", f"{plan['su_m3'].sum():.0f} m³", f"bütçe {butce} m³", delta_color="off")
    su_tur = ozet["su_tur"]
    m3.caption(" · ".join(f"{YONTEMLER[k][0]} {v_:.1f} m³" for k, v_ in su_tur.items() if v_ > 0.001))
    m4.metric("Kırılgan noktalara giden su", f"%{pay(plan):.0f}", f"eşit dağıtımda %{pay(esit):.0f}", delta_color="off",
              help=f"Kırılganlık göstergesi tüm noktaların medyanında ya da üstünde olan noktalara giden suyun payı. "
                   f"Gösterge kaynağı: {kirilgan_kaynak}.")
    if ozet["dusus_aralik"]:
        a, b = ozet["dusus_aralik"]
        m4.caption(f"Noktada UTCI düşüşü {a:.1f}–{b:.1f} °C")
    kullanilan = float(plan["su_m3"].sum())
    if butce - kullanilan > max(5.0, 0.05 * butce):
        neden = ("kırılgan bölgelere asgari pay kuralı ve çim güvenliği sınırları"
                 if kirilgan_pay > 0 else "çim güvenliği sınırları ve hat kapasiteleri")
        st.info(f"Bütçenin {butce - kullanilan:.0f} m³'ü kullanılmadı: {neden} nedeniyle daha fazla su faydalı biçimde "
                "dağıtılamıyor. Daha fazla su için gelişmiş ayarlarda bu sınırları gevşetebilir ya da yeni nokta "
                "ekleyebilirsiniz.")
    st.caption("Bugünkü yaygın uygulama gece sulamasıdır; gündüz ısı stresine doğrudan katkısı yok kabul edilir. "
               "Bu yüzden karşılaştırma, daha zor bir referans olan gündüz eşit park sulamasıyla yapılır; bu senaryo da "
               "aynı hat kapasitesi ve çim güvenliği kurallarına uyar.")

# --- Harita ve saat paneli ---
st.subheader("Isı haritası ve saatlik plan")
GORUNUMLER = {"Yüzey sıcaklığı": "LST", "UTCI: mevcut": "Önce", "UTCI: planla": "Sonra", "Soğuma etkisi": "Soğuma",
              "Soğutma önceliği": "Öncelik"}
if obilgi is not None:
    with st.container(border=True):
        st.markdown(
            f"**Park erişim boşluğu.** Beş parkın 5 dakikalık (400 m) yürüme alanı, yerleşim alanının "
            f"(~{obilgi['yerlesim_km2']:.0f} km²) yalnızca **%{str(round(obilgi['park400_pct'], 1)).replace('.', ',')}**'ini kapsıyor. Yüksek ve çok "
            f"yüksek öncelikli {obilgi['yuksek_km2']:.0f} km²'nin **%{obilgi['yuksek_park_disi_pct']:.0f}**'i bu erişimin "
            "dışında; aday sisleme ve yol ıslatma noktaları bu alanlardan seçildi."
            + (f" Yerleşim alanındaki ~{obilgi['yerlesim_nufus']:,.0f}".replace(",", ".")
               + f" kişinin yalnızca **%{str(round(obilgi['park400_nufus_pct'], 1)).replace('.', ',')}**'i "
               "bir pilot parka 5 dakika yürüme mesafesinde (WorldPop 2025)."
               if obilgi.get("nufus_var") else ""))
        st.caption(f"Öncelik: {obilgi['agirlik']} (ağırlıklı çakıştırma). Yerleşim alanı: {obilgi['kural']}. "
                   "Haritada “Soğutma önceliği” görünümünden inceleyin.")
KATMAN_SECENEKLERI = ["Nokta adları", "Yürüme mesafesi (5 ve 10 dk)", "Hassas tesisler (OSM)"]


def oynat_degistir():
    if st.session_state.oynuyor:
        st.session_state.oynuyor = False
    else:
        st.session_state.oynuyor = True
        st.session_state.oyun_yeni = True
        st.session_state.saat = SAATLER[0]


# CBS haritaları QGIS 3.34'te üretilir (cbs_hazirla.py → cbs_poligon.py → qgis_harita.py) ve repoda haritalar/
# klasöründe durur. Yüzey sıcaklığı, öncelik ve erişim boşluğu sabit katmanlar olduğu için önceden üretilmeleri yeterli;
# saat saat değişen UTCI ve soğuma etkileşimli haritada gösterilir.
CBS_KLASOR = "haritalar"
CBS_HARITALAR = {"Yüzey sıcaklığı (Landsat)": "harita_1", "Soğutma önceliği": "harita_2",
                 "Erişim boşluğu ve müdahale planı": "harita_3"}
CBS_VAR = all(os.path.exists(os.path.join(CBS_KLASOR, f + ".png")) for f in CBS_HARITALAR.values())
CANLI = "Uydu haritası (etkileşimli)"
HARITA_TURLERI = [CANLI] + (["CBS haritası (QGIS)"] if CBS_VAR else []) + ["Etkileşimli 3B"]
c_btn, c_gor, c_tur, c_kat = st.columns([0.8, 2.6, 1.6, 2.2], vertical_alignment="bottom")
c_btn.button("Durdur" if st.session_state.oynuyor else "Günü oynat", on_click=oynat_degistir,
             type="primary", key="oynat_dugmesi")
cbs_secim = None
if yuzey is not None:
    harita_turu = c_tur.selectbox("Harita", HARITA_TURLERI, index=0,
                                  help="Uydu haritası: uydu görüntüsü üzerinde yarı saydam, sürekli renk geçişli ısı "
                                       "katmanı; yakınlaştırılır, altlık ve katmanlar sağ üstteki kutudan değişir. CBS haritası: QGIS'te üretilmiş baskı "
                                       "haritaları, PDF indirilebilir. 3B: altıgen sütunlar.")
    if harita_turu == "CBS haritası (QGIS)":
        cbs_secim = c_gor.radio("CBS haritası", list(CBS_HARITALAR), horizontal=True)
        mod = "Öncelik" if cbs_secim == "Soğutma önceliği" else "Önce"
    else:
        mod = GORUNUMLER[c_gor.radio("Görünüm", list(GORUNUMLER), index=0, horizontal=True)]
else:
    mod, harita_turu = "Önce", "Etkileşimli 2B"
boyut3 = harita_turu == "Etkileşimli 3B"
secenekler = c_kat.multiselect("Ek katmanlar", KATMAN_SECENEKLERI, default=KATMAN_SECENEKLERI[:2])


@st.fragment(run_every=OYNATMA_ADIM_S if st.session_state.oynuyor else None)
def zaman_paneli():
    # Saat, kaydırıcı çizilmeden ÖNCE ilerletilir: böylece kaydırıcı oynatmayla birlikte akar.
    if st.session_state.oynuyor:
        if not st.session_state.pop("oyun_yeni", False):
            st.session_state.saat = min(st.session_state.saat + 1, SAATLER[-1])
        if st.session_state.saat >= SAATLER[-1]:
            st.session_state.oynuyor = False
            st.rerun()  # zamanlayıcıyı kapatmak için tüm sayfayı yeniden çalıştır; son kare 20:00
    st.select_slider("Saat", options=SAATLER, key="saat", format_func=lambda s: f"{s:02d}:00",
                     label_visibility="collapsed")
    saat = st.session_state.saat
    c_harita, c_kart = st.columns([3.3, 1])
    tesis_k = tesisler if "Hassas tesisler (OSM)" in secenekler else None
    if cbs_secim is not None:
        dosya = os.path.join(CBS_KLASOR, CBS_HARITALAR[cbs_secim])
        try:
            c_harita.image(dosya + ".png", width="stretch")
        except Exception:
            c_harita.image(dosya + ".png", use_container_width=True)
        c_harita.caption("QGIS 3.34 ile üretilmiş sabit harita. Saat saat ısı stresi ve planın soğuma etkisi için "
                         "\"Uydu haritası\"nı seçin.")
        if os.path.exists(dosya + ".pdf") and not st.session_state.oynuyor:
            c_harita.download_button("Haritayı indir (PDF)", open(dosya + ".pdf", "rb").read(),
                                     file_name=f"coolscape_{CBS_HARITALAR[cbs_secim]}.pdf", mime="application/pdf",
                                     key="cbs_indir")
        with c_kart.container(border=True):
            saat_karti(saat, hava, plan, vpd_min)
        return
    if harita_turu == CANLI and yuzey is not None:
        m, bilgi = canli_harita(mod, saat, hava, yuzey, parklar, plan, secenekler, tesis_k, "Uydu (Esri World Imagery)")
        canli_harita_goster(c_harita, m)
        c_harita.caption(bilgi["not"] + ". Renk geçişi altıgen değerlerinin ~180 m yumuşatılmış hâlidir; üzerine gelince "
                         "altıgenin kendi değeri görünür. Altlığı (uydu / sokak), eş değer çizgilerini ve diğer katmanları sağ üstteki "
                         "kutudan açıp kapatın.")
        with c_kart.container(border=True):
            saat_karti(saat, hava, plan, vpd_min)
        return
    if harita_turu == "Tematik harita" and yuzey is not None:
        png, ist = tematik_harita(saat, hava, yuzey, parklar, plan, mod, secenekler, tesis_k, gun)
        try:
            c_harita.image(png, width="stretch")
        except Exception:
            c_harita.image(png, use_container_width=True)
        if ist["alan_km2"] == 0:
            c_harita.caption(f"{saat:02d}:00 · Bu saatte sulama yok; mahallelerde soğuma etkisi yok.")
        else:
            c_harita.caption(
                f"{saat:02d}:00 · Planın soğuttuğu alan (≥0,1 °C): ~{ist['alan_km2']:.1f} km², en çok {ist['azami']:.1f} °C. "
                "Soğuma nokta çevresinde yerel kalır (park ~150 m, ıslatma ~60 m, sisleme ~15 m ölçekle azalır; varsayım); "
                "kent ölçeğindeki haritada \"Önce\" ve \"Sonra\" bu yüzden birbirine yakın görünür.")
        if not st.session_state.oynuyor:
            c_harita.download_button("Haritayı indir (PNG)", png, file_name=f"coolscape_{mod.lower()}_{tarih}_{saat:02d}00.png",
                                     mime="image/png", key="harita_indir")
        with c_kart.container(border=True):
            saat_karti(saat, hava, plan, vpd_min)
        return
    mod_3b = "Önce" if mod == "LST" else mod   # 3B görünümde yüzey sıcaklığı yerine UTCI deseni gösterilir
    deck, ist = harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod_3b, secenekler, tesis_k, boyut3)
    c_harita.markdown(lejant_html(mod_3b, ist, sicak_pay, tesis_k is not None, boyut3), unsafe_allow_html=True)
    c_harita.pydeck_chart(deck, height=600, key="harita")
    if ist is not None:
        if ist["alan_km2"] == 0:
            c_harita.caption(f"{saat:02d}:00 · Bu saatte sulama yok; mahallelerde soğuma etkisi yok.")
        else:
            c_harita.caption(
                f"{saat:02d}:00 · Planın soğuttuğu alan (≥0,1 °C): ~{ist['alan_km2']:.1f} km², en çok "
                f"{ist['azami']:.1f} °C · sıcak nokta alanı {ist['once']} → {ist['sonra']} altıgen. Çevreye yayılım "
                f"varsayımdır (park 150 m, ıslatma 60 m, sisleme 15 m ölçekle azalır).")
    with c_kart.container(border=True):
        saat_karti(saat, hava, plan, vpd_min)


zaman_paneli()
if "Hassas tesisler (OSM)" in secenekler and tesisler is None:
    st.caption(f"OpenStreetMap tesisleri yüklenemedi ({osm_hata}); katman gösterilmiyor.")

# --- Noktalar: neden bu nokta? ---
st.subheader("Müdahale noktaları")
st.dataframe(park_ozet_tablosu(parklar, plan, tesisler), hide_index=True)
st.caption("Sisleme ve ıslatma noktaları, soğutma önceliği en yüksek yerleşim altıgenlerinden seçilen, birbirinden ve "
           "parklardan en az 1 km uzak ADAY konumlardır. Kesin yer (durak, meydan, cadde) sahada belediyeyle seçilir.")
if tesisler is not None:
    st.caption("Tesis sayıları noktanın 400 m'si (kuş uçuşu ~5 dk yürüme) içindedir; OpenStreetMap kaydı eksik olabilir. "
               + ("Nüfus dosyası olmadığı için okul/kreş, sağlık ve huzurevi sayısı kırılganlık göstergesi (vekil) olarak "
                  "optimizasyona girer." if nufus is None else "WorldPop 65+ katmanı ilçe düzeyinde olduğu için okul/kreş, sağlık ve "
                  "huzurevi sayısı yine kırılganlık göstergesi (vekil) olarak optimizasyona girer."))

# --- Plan matrisi ve operatör onayı ---
st.subheader("Günün su planı (nokta × saat, m³; • = sisleme açık)")
st.altair_chart(plan_matrisi(plan))

st.subheader("Operatör onay listesi")
komut = operator_komutlari(plan, parklar, aritilmis_senaryo)
if komut.empty:
    st.write("Bu koşullarda sulama önerilmiyor.")
else:
    komut.insert(0, "Onay", False)
    st.data_editor(komut, hide_index=True, disabled=["Nokta", "Yöntem", "Başlangıç", "Bitiş", "Su (m³)", "Kaynak"])
    st.download_button("Planı CSV olarak indir", komut.drop(columns="Onay").to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"coolscape_plan_{tarih}.csv", mime="text/csv")

# --- Bütçe – serinlik eğrisi ---
with st.expander("Bütçe–serinlik eğrisi: ek su ne kazandırır?"):
    if st.button("Eğriyi hesapla"):
        satirlar = []
        for b in sorted({int(butce * k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}):
            p_b, d_b = optimize_et(girdi, b, set(kirilgan_parklar), kirilgan_pay, vpd_min, enerji_opt,
                                   turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
            if d_b == "Optimal":
                satirlar.append({"Bütçe (m³)": b, "Senaryo": "CoolScape", "Fayda": etki(p_b)["agirlikli_fayda"].sum()})
            satirlar.append({"Bütçe (m³)": b, "Senaryo": "Gündüz eşit park sulaması",
                             "Fayda": etki(esit_dagitim(girdi, b, park_kesme, park_mm))["agirlikli_fayda"].sum()})
        egri = pd.DataFrame(satirlar)
        st.altair_chart(alt.Chart(egri).mark_line(point=True).encode(
            x="Bütçe (m³):Q", y=alt.Y("Fayda:Q", title="Öncelik ağırlıklı serinlik"),
            color=alt.Color("Senaryo:N", scale=alt.Scale(range=["#1e88e5", "#999999"]))))
        st.caption("Eğri düzleştiği yerde ek su az ek serinlik getirir: bütçe kararını destekler.")

with st.expander("Neden? (teknik ayrıntı ve varsayımlar)"):
    s = st.session_state.saat
    d = plan[plan["saat"] == s][["park", "tur", "utci_c", "vpd_kpa", "kisi", "oncelik", "emax", "su_m3", "su_mm",
                                 "utci_dusus_c"]].round(2)
    d["tur"] = d["tur"].map(lambda t: YONTEMLER[t][0])
    d.columns = ["Nokta", "Yöntem", "UTCI (°C)", "VPD (kPa)", "Erişim (kişi)", "Öncelik", "Azami etki (°C)", "Su (m³)",
                 "Su (mm = L/m²)", "UTCI düşüşü (°C)"]
    st.markdown(f"**{s:02d}:00 saatinin hesap tablosu**")
    st.dataframe(d, hide_index=True)
    st.dataframe(parklar_g[["park", "yuzey_farki_c", "yuzey_kaynak", "yesil_skoru", "yesil_kaynak", "nufus_skoru",
                            "nufus_kaynak"]]
                 .rename(columns={"park": "Park", "yuzey_farki_c": "Çevre yüzey farkı (°C)", "yuzey_kaynak": "Kaynak",
                                  "yesil_skoru": "Yeşil skoru", "yesil_kaynak": "Kaynak ",
                                  "nufus_skoru": "Nüfus skoru (1–10)", "nufus_kaynak": "Kaynak  "}),
                 hide_index=True)
    kd = parklar[["park", "kirilgan_endeks"]].copy()
    kd["Kırılgan sayılır"] = kd["park"].isin(kirilgan_parklar).map({True: "evet", False: "hayır"})
    st.markdown(f"**Kırılganlık göstergesi** (tüm noktalar aynı kuralla; kaynak: {kirilgan_kaynak}). "
                "Kırılgan nokta: göstergesi tüm noktaların medyanında ya da üstünde.")
    st.dataframe(kd.rename(columns={"park": "Nokta", "kirilgan_endeks": "Gösterge (0–1)"}), hide_index=True)
    st.markdown(
        "- **Öncelik** = max(UTCI − 32, 0) × erişim (kişi) / 100 × (1 + kırılganlık ağırlığı × kırılganlık göstergesi). "
        "Gösterge 0–1: 400 m içindeki okul/kreş, sağlık, huzurevi sayısının noktalar arasındaki en yüksek değere "
        "oranı (vekil). WorldPop 65+ payı kullanılmaz: pilot bölgede ilçe başına tek değerdir (%6,8 / %9,8 / %11,2).\n"
        "- **Yöntemler:** park sulama ve yol ıslatma azalan getirili eğriyle (aşağıda); sisleme aç/kapa, açıkken "
        f"{SISLEME_E} °C × min(VPD / 2,5 kPa, 1) UTCI düşüşü ve saatte {SISLEME_SU_M3_SAAT} m³ su (varsayım). "
        "Sisleme yalnızca içme suyuyla çalışır (lejyonella).\n"
        f"- **Çim güvenliği:** park sulaması {park_kesme}:00'dan sonra yapılmaz; park başına günlük en çok {park_mm} mm.\n"
        "- **Soğutma eğrisi:** UTCI düşüşü = azami etki × (1 − e^(−doz/x₀)). Doz, parkın sulanan alanına düşen su "
        "miktarıdır (mm = L/m²); x₀ = 0,5 mm/saat (duyarlılık 0,25–1,0 mm), yani ~1,5 mm/saatte etkinin %95'i alınır.\n"
        "- **Azami etki** 1,0 °C UTCI × min(VPD / 2,5 kPa, 1). Nemli saatlerde buharlaşma ve serinlik azalır. "
        "Belirsizlik bandı 0,5–2,0 °C.\n"
        "- **MILP:** öncelik ağırlıklı UTCI düşüşünü maksimize eder; kısıtlar su bütçesi, hat kapasitesi, en az doz, "
        "nem kapısı, kırılgan bölgelere asgari pay, akşam kesme saati ve günlük park üst sınırı.\n"
        "- **Karşılaştırma (gündüz eşit park sulaması):** aynı bütçe yalnızca parklara ve izin verilen gündüz saatlerine "
        "eşit bölünür, aynı kapasite ve çim güvenliği kurallarına uyar; "
        "nem, ısı stresi ve kırılganlık gözetilmez. Kazanç, aynı etki eğrileriyle hesaplanan bir simülasyondur.\n"
        f"- **MRT** = hava sıcaklığı + güneş katkısı + 0,5 × yüzey sıcaklığı farkı. Bu çalıştırmada güneş katkısı: "
        f"{mrt_yontem}. SolarCal varsayımları: ayakta kişi, güneşte, gökyüzü görüş oranı 0,5, güneş yandan, giysi "
        "yutuculuğu 0,7; DNI Open-Meteo'dan, güneş yüksekliği hesapla. Bina gölgesi ve ağaç gölgesi hesaba katılmaz.\n"
        "- **Soğutma önceliği:** yüzey ısısı (0,40), 65+ sayısı (0,25; yalnızca nufus_h3.csv varsa; WorldPop toplam nüfusu × ilçe 65+ oranı, mahalle farkını göstermez), parka uzaklık (0,20) "
        "ve 400 m içindeki okul/sağlık tesisi (0,15) ölçütlerinin ağırlıklı ortalaması (eksik ölçüt çıkarılıp ağırlıklar "
        "yeniden ölçeklenir). Seyrek yerleşim: nüfus varsa altıgende 50 kişinin altı, yoksa 1 km içinde okul/sağlık "
        "tesisi olmayan altıgenler. Park nüfus skoru 800 m içindeki nüfustan (yoksa okul/sağlık tesisi sayısından, vekil) "
        "parklar arasında göreli 1–10 ölçeğinde türetilir. "
        "Sınıflar yerleşim alanının beşte birlik dilimleridir. Ağırlıklar uzman varsayımıdır.\n"
        "- **Harita:** hesaplar H3 altıgenleriyle (~0,1 km²) yapılır. Tematik haritada altıgen değerleri ~60 m'lik "
        "ızgarada ~180 m Gauss süzgeciyle yumuşatılıp sınıf alanları eş değer eğrileriyle çizilir; yumuşatma yalnızca "
        "görseldir. UTCI sınıf aralıkları o saatin değer aralığına göre 1 °C (geniş aralıkta 2 °C) adımlıdır. "
        "Etkileşimli haritada 3B sütun yüksekliği değerle orantılıdır. "
        "Landsat yerel sabah geçer; harita öğleden sonraki pik sıcaklığı değil, yüzeyin göreli ısınma desenini gösterir.\n"
        "- **UTCI sınıf eşikleri** resmîdir (26/32/38/46 °C); renkler bu uygulamanın seçimidir.\n"
        f"- **Yürüme halkaları** kuş uçuşu {YURUME_5DK_M} ve {YURUME_10DK_M} m'dir, yol ağına göre değildir."
    )
    st.markdown("**Etki eğrilerinin dayandığı saha ölçümleri**")
    st.dataframe(pd.DataFrame([
        ["Hendel vd. 2016, Urban Climate", "Paris, Louvre", "Kaldırım ıslatma", "−0,79 °C", "−1,03 °C", "Merkezi değer"],
        ["Hendel vd. 2016, Urban Climate", "Paris, Belleville", "Kaldırım ıslatma", "−0,60 °C", "−0,93 °C", "Merkezi değer"],
        ["Parison vd. 2020", "Paris (2013–2015)", "Kaldırım ıslatma", "−1,02 °C", "−1,93 °C", "Üst band"],
        ["Traill vd. 2024, Urban Climate", "Melbourne", "Kaldırım ıslatma (~1 mm)", "en çok −0,6 °C", "−0,2 … −2,0 °C", "Band"],
        ["Broadbent vd. 2018, Urban Climate", "Adelaide (model)", "Zemin/çim sulama", "günlük ort. en çok −2,3 °C",
         "–", "Doyma: ~20 L/m²/gün üzeri ek fayda yok"],
        ["Vanos vd. 2022, Int. J. Biometeorology", "Tempe, Arizona", "Sisleme (restoran terasları)", "–",
         "ort. −4,4 °C (gölgeyle −9,7 °C)", "Sisleme etkisi"],
    ], columns=["Çalışma", "Yer", "Uygulama", "Hava sıcaklığı", "UTCI", "Modelde kullanımı"]), hide_index=True)
    st.caption("Sınırlar: Paris ve Melbourne ölçümleri kaldırım ıslatmadır; park çim sulamasında etki farklı olabilir. "
               "Sisleme etkisi yalnızca birkaç metrelik alanda hissedilir ve Arizona'nın kuru koşulunda ölçülmüştür. "
               "Antalya'nın kuru-sıcak koşulu bu ölçümlerin çoğundan daha serttir. Sulanan alanlar temsilidir. "
               "Bu nedenle eğri yerel ölçümle doğrulanmalıdır (2027 saha pilotu).")
