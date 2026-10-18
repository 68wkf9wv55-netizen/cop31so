"""
CoolScape AI – CBS katmanları ve öncelikli soğutma alanı analizi.

Girdi:  isi_yuzeyi_h3.csv (zorunlu, Landsat yüzey sıcaklığı H3 altıgenleri)
        tesisler_osm.csv  (isteğe bağlı, OpenStreetMap hassas tesisleri)
Çıktı:  coolscape_cbs.gpkg (UTM 36N, EPSG:32636) ve analiz_ozeti.csv

Ağırlıklı çakıştırma (her ölçüt 0–1 aralığına ölçeklenir):
  C1 Isı          : yüzey sıcaklığının kent medyanından farkı (Landsat)
  C2 Park boşluğu : en yakın parka uzaklık (400 m'de 0, 1500 m ve ötesi 1)
  C3 Hassas tesis : 400 m içindeki okul/kreş, sağlık ve huzurevi sayısı (OSM; yoksa ölçüt dışı)
  C4 65+ nüfus    : altıgendeki 65 yaş ve üstü kişi sayısı (WorldPop 2025; yoksa ölçüt dışı)
  Öncelik = Σ ağırlık × ölçüt / Σ ağırlık   (ağırlıklar: ısı 0,40 · 65+ 0,25 · park 0,20 · tesis 0,15)
  Sınıflar: öncelik değerinin beşte birlik dilimleri (quintile)
  Deniz/su altıgenleri (LST < 36 °C) analiz dışıdır.
"""

import argparse
import json
import os
import sys

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point

CRS_UTM = "EPSG:32636"
AGIRLIK = {"c_isi": 0.40, "c_yasli": 0.25, "c_park": 0.20, "c_tesis": 0.15}
OLCUT_AD = {"c_isi": "yüzey ısısı", "c_yasli": "65+ sayısı (ilçe oranıyla)", "c_park": "parka uzaklık", "c_tesis": "okul ve sağlık tesisi"}
YERLESIM_YARICAP_M = 1000   # nüfus verisi yoksa: bu mesafede en az bir okul/sağlık tesisi olan altıgen "yerleşim" sayılır
YERLESIM_NUFUS_ESIK = 50    # nüfus verisi varsa: altıgende en az bu kadar kişi yaşıyorsa "yerleşim"
SINIF_ADLARI = ["Çok düşük", "Düşük", "Orta", "Yüksek", "Çok yüksek"]


def uygulama_fonksiyonlari(app_yolu):
    """Parklar ve aday nokta kuralı uygulamayla birebir aynı olsun diye uygulama kodundan alınır."""
    kaynak = open(app_yolu, encoding="utf-8").read().split("# ARAYÜZ")[0]
    ns = {}
    exec(compile(kaynak, app_yolu, "exec"), ns)
    return ns


def h3_poligon(hex_id):
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(hex_id)])


def tesis_oku(yol, ns):
    if not yol or not os.path.exists(yol):
        return None
    oku = getattr(ns["tesis_dosyasi_oku"], "__wrapped__", ns["tesis_dosyasi_oku"])
    return oku(yol, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yuzey", default="isi_yuzeyi_h3.csv")
    ap.add_argument("--tesis", default="tesisler_osm.csv")
    ap.add_argument("--nufus", default="nufus_h3.csv")
    ap.add_argument("--app", default="coolscape_app.py")
    ap.add_argument("--cikti", default="coolscape_cbs.gpkg")
    a = ap.parse_args()

    ns = uygulama_fonksiyonlari(a.app)
    oku = getattr(ns["isi_yuzeyi_oku"], "__wrapped__", ns["isi_yuzeyi_oku"])
    yuzey = oku(a.yuzey, 0)
    if yuzey is None:
        sys.exit(f"{a.yuzey} okunamadı.")
    parklar = ns["parklari_guncelle"](ns["PARKLAR"], yuzey)
    tesis = tesis_oku(a.tesis, ns)
    # Aday noktalar uygulamadakiyle birebir aynı: önce uygulamanın öncelik analizi, sonra aday kuralı
    nf_app = pd.read_csv(a.nufus)[["hex", "nufus", "yasli_65"]] if a.nufus and os.path.exists(a.nufus) else None
    onc = getattr(ns["oncelik_hesapla"], "__wrapped__", ns["oncelik_hesapla"])
    y_app, _ = onc(yuzey, parklar, tesis, nf_app)
    adaylar = ns["aday_noktalar"](y_app, parklar)

    # --- Veri kapsamı ve su alanları (ham dosyadan) ---
    ham = pd.read_csv(a.yuzey).dropna(subset=["lst_c"])
    ham_g = gpd.GeoDataFrame(ham, geometry=[h3_poligon(h) for h in ham["hex"]], crs="EPSG:4326").to_crs(CRS_UTM)
    kapsam = gpd.GeoDataFrame({"ad": ["Uydu verisi kapsamı"]}, geometry=[ham_g.union_all()], crs=CRS_UTM)
    su = ham_g[ham_g["lst_c"] < ns["SU_ESIGI_C"]].dissolve().reset_index(drop=True)[["geometry"]]
    su["ad"] = "Deniz / su yüzeyi (analiz dışı)"
    _, kiyi_set = ns["kiyi_siniflari"](ham)
    kiyi = ham_g[ham_g["hex"].isin(kiyi_set)][["hex", "lst_c", "geometry"]].copy()
    kiyi["ad"] = "Kıyı karışık piksel (analiz dışı)"

    # --- Isı yüzeyi altıgenleri ---
    g = gpd.GeoDataFrame(yuzey.copy(), geometry=[h3_poligon(h) for h in yuzey["hex"]], crs="EPSG:4326").to_crs(CRS_UTM)
    merkez = g.geometry.centroid
    g["lst_c"] = g["lst_c"].round(2)
    g["lst_fark"] = g["lst_fark_c"].round(2)

    pk = gpd.GeoDataFrame(parklar.drop(columns=["yesil_kaynak"], errors="ignore"),
                          geometry=gpd.points_from_xy(parklar["lon"], parklar["lat"]), crs="EPSG:4326").to_crs(CRS_UTM)
    pk["veri_durumu"] = ["Uydu verisi içinde" if kapsam.geometry.iloc[0].contains(p) else "Uydu sahnesi dışında"
                         for p in pk.geometry]

    # C1 ısı: 5.–95. yüzdelik arasında doğrusal ölçek
    p5, p95 = np.nanpercentile(g["lst_fark"], [5, 95])
    g["c_isi"] = np.clip((g["lst_fark"] - p5) / (p95 - p5), 0, 1).round(3)
    # C2 park boşluğu
    d = np.min(np.stack([merkez.distance(p) for p in pk.geometry]), axis=0)
    g["d_park_m"] = d.round(0)
    g["c_park"] = np.clip((d - 400) / (1500 - 400), 0, 1).round(3)
    # C3 hassas tesis
    agirlik = dict(AGIRLIK)
    if tesis is not None and len(tesis):
        tg = gpd.GeoDataFrame(tesis, geometry=gpd.points_from_xy(tesis["lon"], tesis["lat"]), crs="EPSG:4326").to_crs(CRS_UTM)
        hassas = tg[tg["kategori"].isin(["Okul / kreş", "Sağlık", "Huzurevi"])]
        if len(hassas):
            xy = np.c_[hassas.geometry.x, hassas.geometry.y]
            say = [int((np.hypot(xy[:, 0] - c.x, xy[:, 1] - c.y) <= 400).sum()) for c in merkez]
            yakin = [int((np.hypot(xy[:, 0] - c.x, xy[:, 1] - c.y) <= YERLESIM_YARICAP_M).sum()) for c in merkez]
        else:
            say = [0] * len(g)
            yakin = [0] * len(g)
        g["n_tesis_1km"] = yakin
        g["n_tesis_400"] = say
        ust = max(np.percentile(say, 95), 1)
        g["c_tesis"] = np.clip(np.array(say) / ust, 0, 1).round(3)
    else:
        tg = None
        g["n_tesis_400"] = None
        g["n_tesis_1km"] = None
        g["c_tesis"] = None
        agirlik.pop("c_tesis")
    # C4 65+ nüfus
    if a.nufus and os.path.exists(a.nufus):
        nf = pd.read_csv(a.nufus)[["hex", "nufus", "yasli_65"]]
        g = g.merge(nf, on="hex", how="left")
        g[["nufus", "yasli_65"]] = g[["nufus", "yasli_65"]].fillna(0)
        ust_y = max(np.nanpercentile(g["yasli_65"], 95), 1)
        g["c_yasli"] = np.clip(g["yasli_65"] / ust_y, 0, 1).round(3)
        merkez = g.geometry.centroid
    else:
        g["nufus"] = None
        g["yasli_65"] = None
        g["c_yasli"] = None
        agirlik.pop("c_yasli")
    toplam = sum(agirlik.values())
    g["oncelik"] = (sum(g[k].astype(float) * w for k, w in agirlik.items()) / toplam).round(3)
    # 5 sınıf: beşte birlik dilimler (her sınıf kara alanının ~%20'si). Kentin neredeyse tamamı sıcak olduğundan
    # eşit aralıklı sınıflar alanın çoğunu "yüksek"e atar; dilimler göreli önceliği ayırt eder.
    # Yerleşim maskesi: seyrek yerleşimli sıcak alanlar (açık arazi, sanayi) önceliği şişirmesin
    if "c_yasli" in agirlik:
        g["yerlesim"] = g["nufus"].astype(float) >= YERLESIM_NUFUS_ESIK
        yerlesim_kurali = f"altıgende en az {YERLESIM_NUFUS_ESIK} kişi (WorldPop)"
    elif "c_tesis" in agirlik:
        g["yerlesim"] = g["n_tesis_1km"].astype(int) >= 1
        yerlesim_kurali = f"{YERLESIM_YARICAP_M} m içinde en az bir okul veya sağlık tesisi (OSM)"
    else:
        g["yerlesim"] = True
        yerlesim_kurali = "yok (tüm kara alanı)"
    y = g["yerlesim"]
    g["sinif_no"] = 0
    g.loc[y, "sinif_no"] = pd.qcut(g.loc[y, "oncelik"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["sinif"] = g["sinif_no"].map(lambda i: SINIF_ADLARI[i - 1] if i > 0 else "Seyrek yerleşim")
    g["park_400_icinde"] = g["d_park_m"] <= 400
    alan_km2 = float(g.geometry.area.mean() / 1e6)

    # --- Tamponlar ---
    tampon = []
    for _, r in pk.iterrows():
        for m, ad in ((400, "5 dk yürüme (400 m)"), (800, "10 dk yürüme (800 m)"), (500, "Soğuma etkisi (500 m)")):
            tampon.append({"park": r["park"], "mesafe_m": m, "tur": ad, "geometry": r.geometry.buffer(m, 64)})
    tampon = gpd.GeoDataFrame(tampon, crs=CRS_UTM)

    # --- Sıcak odaklar: "Çok yüksek" + "Yüksek" sınıfları birleştirilir ---
    odak = g[g["sinif_no"] >= 4].dissolve(by="sinif").reset_index()[["sinif", "geometry"]]
    odak = odak.explode(index_parts=False).reset_index(drop=True)
    odak["alan_km2"] = (odak.geometry.area / 1e6).round(3)
    odak["park_400_disi_pay"] = [
        round(float(1 - geom.intersection(tampon[tampon.mesafe_m == 400].union_all()).area / geom.area), 3)
        for geom in odak.geometry]

    # --- Aday noktalar (uygulamadaki kuralla) ---
    ad = gpd.GeoDataFrame(adaylar[["park", "tur", "ilce", "lat", "lon"]].rename(columns={"park": "ad"}),
                          geometry=gpd.points_from_xy(adaylar["lon"], adaylar["lat"]), crs="EPSG:4326").to_crs(CRS_UTM)
    ad["yontem"] = ad["tur"].map({"sisleme": "Sisleme noktası", "islatma": "Yol ıslatma"})
    ad = ad.sjoin(g[["oncelik", "sinif", "geometry"]], how="left", predicate="within").drop(columns="index_right")

    # --- Yaz ---
    if os.path.exists(a.cikti):
        os.remove(a.cikti)
    sutun = ["hex", "lst_c", "lst_fark", "d_park_m", "n_tesis_400", "nufus", "yasli_65", "c_isi", "c_park",
             "c_tesis", "c_yasli", "oncelik",
             "sinif_no", "sinif", "yerlesim", "park_400_icinde", "geometry"]
    g[sutun].to_file(a.cikti, layer="oncelik_alanlari", driver="GPKG")
    kapsam.to_file(a.cikti, layer="veri_kapsami", driver="GPKG")
    su.to_file(a.cikti, layer="su_alanlari", driver="GPKG")
    if len(kiyi):
        kiyi.to_file(a.cikti, layer="kiyi_karisik", driver="GPKG")
    pk[["park", "ilce", "tur", "veri_durumu", "yuzey_kaynak", "yuzey_farki_c", "nufus_skoru", "hassas_yuzde", "hat_kapasitesi_m3_saat",
        "sulanan_alan_ha", "geometry"]].to_file(a.cikti, layer="parklar", driver="GPKG")
    tampon.to_file(a.cikti, layer="park_tamponlari", driver="GPKG")
    odak.to_file(a.cikti, layer="sicak_odaklar", driver="GPKG")
    ad[["ad", "yontem", "ilce", "oncelik", "sinif", "geometry"]].to_file(a.cikti, layer="aday_noktalar", driver="GPKG")
    if tg is not None:
        tg[["kategori", "ad", "geometry"]].to_file(a.cikti, layer="hassas_tesisler", driver="GPKG")

    # --- Özet istatistik ---
    satir = []
    for no, adi in enumerate(SINIF_ADLARI, 1):
        s = g[g["sinif_no"] == no]
        satir.append({"Öncelik sınıfı": adi, "Altıgen": len(s), "Alan (km²)": round(len(s) * alan_km2, 2),
                      "Alan payı (%)": round(100 * len(s) / len(g), 1),
                      "Park 400 m içinde (%)": round(100 * s["park_400_icinde"].mean(), 1) if len(s) else 0.0})
    s0 = g[g["sinif_no"] == 0]
    satir.append({"Öncelik sınıfı": "Seyrek yerleşim (sıralama dışı)", "Altıgen": len(s0),
                  "Alan (km²)": round(len(s0) * alan_km2, 2), "Alan payı (%)": round(100 * len(s0) / len(g), 1),
                  "Park 400 m içinde (%)": round(100 * s0["park_400_icinde"].mean(), 1) if len(s0) else 0.0})
    ozet = pd.DataFrame(satir)
    ozet.to_csv("analiz_ozeti.csv", index=False, encoding="utf-8-sig")
    yuksek = g[g["sinif_no"] >= 4]
    print(ozet.to_string(index=False))
    print(f"Ağırlıklar: {agirlik} · Yüksek+çok yüksek öncelikli alan: {len(yuksek) * alan_km2:.2f} km², "
          f"bunun %{100 * (~yuksek['park_400_icinde']).mean():.0f}'i parkların 400 m'si dışında")
    bilgi = {
        "agirlik": agirlik, "tesis_var": tg is not None, "nufus_var": "c_yasli" in agirlik,
        "agirlik_metin": " · ".join(f"{OLCUT_AD[k]} {v:.2f}".replace(".", ",") for k, v in agirlik.items()),
        "eksik": [OLCUT_AD[k] for k in AGIRLIK if k not in agirlik],
        "nufus_toplam": int(g["nufus"].sum()) if "c_yasli" in agirlik else None,
        "yasli_yuksek_pct": round(float(100 * g.loc[g["sinif_no"] >= 4, "yasli_65"].sum() / max(g["yasli_65"].sum(), 1)), 0)
        if "c_yasli" in agirlik else None,
        "kara_altigen": int(len(g)), "su_altigen": int(len(ham) - len(g)),
        "yuksek_km2": round(len(yuksek) * alan_km2, 1),
        "cok_yuksek_km2": round(int((g["sinif_no"] == 5).sum()) * alan_km2, 1),
        "cok_yuksek_park_disi_pct": round(float(100 * (~g.loc[g["sinif_no"] == 5, "park_400_icinde"]).mean()), 0),
        "park_400_kapsam_pct": round(float(100 * g.loc[y, "park_400_icinde"].mean()), 1),
        "yerlesim_km2": round(int(y.sum()) * alan_km2, 1), "yerlesim_kurali": yerlesim_kurali, "toplam_km2": round(len(g) * alan_km2, 1),
        "yuksek_park_disi_pct": round(float(100 * (~yuksek["park_400_icinde"]).mean()), 0) if len(yuksek) else 0,
        "park_disi_veri": pk.loc[pk["veri_durumu"] != "Uydu verisi içinde", "park"].tolist(),
        "sahne": str(ham["landsat_sahne"].iloc[0]) if "landsat_sahne" in ham else "?",
    }
    json.dump(bilgi, open("analiz_bilgi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(bilgi)
    print(f"Yazıldı: {a.cikti}")


if __name__ == "__main__":
    main()
