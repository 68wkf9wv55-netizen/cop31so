"""
CoolScape AI – klasik tematik harita katmanları.
cbs_hazirla.py'nin ürettiği coolscape_cbs.gpkg içindeki H3 altıgenlerini sınıflara göre birleştirir ve
kenarlarını yumuşatarak düzgün poligonlara çevirir (altıgen görünümü kalkar). Ayrıca çalışma alanı sınırı
ve yönlendirme amaçlı yer adları katmanı ekler.

Çalıştırma: python cbs_poligon.py   (cbs_hazirla.py'den sonra)
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

GPKG = "coolscape_cbs.gpkg"
CRS = "EPSG:32636"
YUMUSATMA_M = 110      # kapanış (genişlet + daralt) yarıçapı; altıgen dişlerini giderir
SADELESTIRME_M = 20
EN_KUCUK_ALAN_M2 = 60_000  # bundan küçük adacıklar atılır (~yarım altıgen)

LST_SINIF = [(-99, -3, 1, "3 °C'den fazla serin"), (-3, -1, 2, "1–3 °C serin"), (-1, 1, 3, "Kent ortalaması (±1 °C)"),
             (1, 3, 4, "1–3 °C sıcak"), (3, 99, 5, "3 °C'den fazla sıcak")]

# Yönlendirme etiketleri: ilçe ve deniz adları, yaklaşık konumlar (sınır verisi değildir)
YER_ADLARI = [
    ("KEPEZ", 36.935, 30.700, "ilce"), ("MURATPAŞA", 36.880, 30.735, "ilce"), ("KONYAALTI", 36.885, 30.632, "ilce"),
    ("DÖŞEMEALTI", 36.968, 30.660, "komsu"), ("AKSU", 36.930, 30.798, "komsu"),
    ("ANTALYA KÖRFEZİ", 36.852, 30.690, "deniz"),
]


PIKSEL_M = 30          # ızgara çözünürlüğü
COGUNLUK_YARICAP_PX = 6  # çoğunluk süzgeci yarıçapı (~150 m): sınıf sınırlarını yuvarlar, boşluk bırakmaz


def raster_poligon(gdf, alan, adlar, deniz=None, kiyi=None):
    """Altıgen sınıflarını 30 m ızgaraya aktarır, çoğunluk süzgeciyle yumuşatır ve poligona çevirir.
    Komşu sınıflar arasında boşluk ya da çakışma oluşmaz (tek parça mozaik)."""
    import numpy as np
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    tum = gdf[[alan, "geometry"]].copy()
    if deniz is not None:
        tum = pd.concat([tum, gpd.GeoDataFrame({alan: [-1] * len(deniz)}, geometry=deniz.geometry.values, crs=CRS)])
    if kiyi is not None and len(kiyi):   # kıyı karışık altıgenler: değeri komşu sınıflardan (oylamaya girmez)
        tum = pd.concat([tum, gpd.GeoDataFrame({alan: [-3] * len(kiyi)}, geometry=kiyi.geometry.values, crs=CRS)])
    x0, y0, x1, y1 = tum.total_bounds
    gen, yuk = int((x1 - x0) / PIKSEL_M) + 2, int((y1 - y0) / PIKSEL_M) + 2
    tr = from_origin(x0 - PIKSEL_M, y1 + PIKSEL_M, PIKSEL_M, PIKSEL_M)
    r = features.rasterize(((g, int(v) + 2) for g, v in zip(tum.geometry, tum[alan])), out_shape=(yuk, gen),
                           transform=tr, fill=0, dtype="int16")   # 0 = veri yok, 1 = deniz, 2.. = sınıflar
    degerler = [int(v) for v in np.unique(r) if v != -1]  # 0 (veri dışı) da oylamaya katılır; -1 = kıyı karışık
    k = COGUNLUK_YARICAP_PX
    oy = np.zeros((len(degerler),) + r.shape, dtype="float32")
    for i, v in enumerate(degerler):
        m = np.pad((r == v).astype("float32"), k, mode="constant")
        c = np.cumsum(np.cumsum(m, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0)))
        oy[i] = c[2 * k + 1:, 2 * k + 1:] - c[:-2 * k - 1, 2 * k + 1:] - c[2 * k + 1:, :-2 * k - 1] + c[:-2 * k - 1, :-2 * k - 1]
    yeni = np.array(degerler, dtype="int16")[oy.argmax(0)]
    parca = []
    for geom, v in features.shapes(yeni, mask=yeni > 0, transform=tr):
        v = int(v) - 2
        parca.append({alan: v, "geometry": shape(geom).simplify(PIKSEL_M * 0.8)})
    out = gpd.GeoDataFrame(parca, crs=CRS)
    out = out[out.geometry.area >= EN_KUCUK_ALAN_M2].copy()
    out["ad"] = out[alan].map(lambda v: adlar.get(v, "Deniz" if v == -1 else str(v)))
    out["alan_km2"] = (out.geometry.area / 1e6).round(2)
    return out


def dis_sinir(gdf):
    """Poligonları tek sınırda birleştirir; ince boşlukları kapatır, iç delikleri (küçük adacık kalıntıları) siler."""
    from shapely.geometry import Polygon, MultiPolygon
    u = gdf.union_all().buffer(25).buffer(-25)
    parcalar = u.geoms if hasattr(u, "geoms") else [u]
    tut = [Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area > 1_000_000]) for p in parcalar
           if p.area > EN_KUCUK_ALAN_M2]
    return MultiPolygon(tut) if len(tut) > 1 else tut[0]


def yumusat(geom):
    g = geom.buffer(YUMUSATMA_M, join_style=1).buffer(-YUMUSATMA_M, join_style=1).simplify(SADELESTIRME_M)
    return g


def poligonlastir(gdf, alan, adlar):
    parca = []
    for deger, grup in gdf.groupby(alan):
        g = yumusat(grup.union_all())
        parca.append({alan: deger, "ad": adlar.get(deger, str(deger)), "geometry": g})
    out = gpd.GeoDataFrame(parca, crs=CRS).explode(index_parts=False).reset_index(drop=True)
    out = out[out.geometry.area >= EN_KUCUK_ALAN_M2].copy()
    out["alan_km2"] = (out.geometry.area / 1e6).round(2)
    return out


def main():
    g = gpd.read_file(GPKG, layer="oncelik_alanlari")
    # Yüzey sıcaklığı sınıfları
    g["lst_sinif"] = 3
    for a, b, no, _ in LST_SINIF:
        g.loc[(g["lst_fark"] > a) & (g["lst_fark"] <= b), "lst_sinif"] = no
    su = gpd.read_file(GPKG, layer="su_alanlari")
    import pyogrio
    kiyi = gpd.read_file(GPKG, layer="kiyi_karisik") if "kiyi_karisik" in [l[0] for l in pyogrio.list_layers(GPKG)] else None
    lst_t = raster_poligon(g, "lst_sinif", {no: ad for _, _, no, ad in LST_SINIF}, deniz=su, kiyi=kiyi)
    deniz = lst_t[lst_t["lst_sinif"] == -1][["ad", "geometry"]].dissolve().reset_index(drop=True)
    lst = lst_t[lst_t["lst_sinif"] >= 1].sort_values("lst_sinif")
    lst.to_file(GPKG, layer="lst_sinif_poligon", driver="GPKG")

    # Öncelik sınıfları (0 = seyrek yerleşim)
    adlar = {0: "Seyrek yerleşim (sıralama dışı)", 1: "Çok düşük", 2: "Düşük", 3: "Orta", 4: "Yüksek", 5: "Çok yüksek"}
    onc_t = raster_poligon(g, "sinif_no", adlar, deniz=su, kiyi=kiyi)
    onc = onc_t[onc_t["sinif_no"] >= 0].sort_values("sinif_no")
    onc.to_file(GPKG, layer="oncelik_poligon", driver="GPKG")

    # Yüksek öncelikli odaklar (yumuşatılmış)
    odak = onc[onc["sinif_no"] >= 4].copy()
    odak["grup"] = odak["sinif_no"].map({4: "Yüksek öncelik", 5: "Çok yüksek öncelik"})
    odak.to_file(GPKG, layer="odak_poligon", driver="GPKG")

    # Çalışma alanı (kara) sınırı
    kara = gpd.GeoDataFrame({"ad": ["Çalışma alanı"]}, geometry=[dis_sinir(onc)], crs=CRS)
    kara.to_file(GPKG, layer="calisma_alani", driver="GPKG")

    # Deniz (aynı mozaikten)
    deniz.to_file(GPKG, layer="deniz", driver="GPKG")

    # Yer adları
    ya = gpd.GeoDataFrame([{"ad": a, "tur": t} for a, _, _, t in YER_ADLARI],
                          geometry=[Point(lon, lat) for _, lat, lon, _ in YER_ADLARI], crs="EPSG:4326").to_crs(CRS)
    ya.to_file(GPKG, layer="yer_adlari", driver="GPKG")
    print("Eklendi:", len(lst), "sıcaklık poligonu,", len(onc), "öncelik poligonu,", len(odak), "odak poligonu")


if __name__ == "__main__":
    main()
