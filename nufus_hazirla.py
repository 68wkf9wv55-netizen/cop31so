"""
CoolScape AI – nüfus ve 65+ nüfus katmanı (WorldPop 2025, 100 m, constrained, R2025A).

WorldPop yaş-cinsiyet rasterlarından pilot bölge penceresini okur ve H3 altıgenlerine toplar.
Dosyaların tamamını indirmez: GDAL /vsicurl/ ile uzak dosyadan yalnızca Antalya penceresini okur.
İnternet uzak okumaya izin vermezse, aynı dosyaları elle indirip --klasor ile gösterin.

Çıktı: nufus_h3.csv  (hex, nufus, yasli_65, lat, lon)
Kurulum: pip install rasterio h3 pandas numpy
Çalıştırma: python nufus_hazirla.py
            python nufus_hazirla.py --klasor C:\\Users\\...\\Downloads\\worldpop   (elle indirdiyseniz)

Kaynak: WorldPop (www.worldpop.org), Global2 R2025A yaş-cinsiyet yapıları, CC BY 4.0.
"""

import argparse
import os

import h3
import numpy as np
import pandas as pd

TABAN = ("https://worldpop-public-data.soton.ac.uk/GIS/AgeSex_structures/Global_2015_2030/R2025A/"
         "2025/TUR/v1/100m/constrained/")
TOPLAM = ["tur_T_F_2025_CN_100m_R2025A_v1.tif", "tur_T_M_2025_CN_100m_R2025A_v1.tif"]   # kadın + erkek toplam
YASLI = [f"tur_t_{y}_2025_CN_100m_R2025A_v1.tif" for y in (65, 70, 75, 80, 85, 90)]     # 65 yaş ve üstü
BBOX = (30.62, 36.85, 30.80, 36.97)  # batı, güney, doğu, kuzey (uygulamadakiyle aynı)


def pencere_oku(yol):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.Env(GDAL_HTTP_MAX_RETRY="3", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(yol) as r:
            w = from_bounds(*BBOX, transform=r.transform).round_offsets().round_lengths()
            a = r.read(1, window=w).astype("float64")
            if r.nodata is not None:
                a[a == r.nodata] = 0
            a[~np.isfinite(a) | (a < 0)] = 0
            t = r.window_transform(w)
    satir, sutun = np.indices(a.shape)
    lon = t.c + (sutun + 0.5) * t.a
    lat = t.f + (satir + 0.5) * t.e
    return a.ravel(), lat.ravel(), lon.ravel()


def h3_topla(dosyalar, kaynak, cozunurluk):
    toplam = None
    for f in dosyalar:
        yol = os.path.join(kaynak, f) if not kaynak.startswith("http") else "/vsicurl/" + kaynak + f
        print("okunuyor:", f)
        v, lat, lon = pencere_oku(yol)
        toplam = v if toplam is None else toplam + v
    m = toplam > 0
    hexler = [h3.latlng_to_cell(a, b, cozunurluk) for a, b in zip(lat[m], lon[m])]
    return pd.Series(toplam[m]).groupby(pd.Series(hexler)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klasor", default=TABAN, help="Elle indirilen .tif dosyalarının klasörü (varsayılan: uzaktan oku)")
    ap.add_argument("--h3", type=int, default=9)
    ap.add_argument("--cikti", default="nufus_h3.csv")
    a = ap.parse_args()
    nufus = h3_topla(TOPLAM, a.klasor, a.h3)
    yasli = h3_topla(YASLI, a.klasor, a.h3)
    df = pd.DataFrame({"nufus": nufus, "yasli_65": yasli}).fillna(0).round(1)
    df.index.name = "hex"
    df = df.reset_index()
    m = [h3.cell_to_latlng(h) for h in df["hex"]]
    df["lat"] = [x for x, _ in m]
    df["lon"] = [y for _, y in m]
    df.to_csv(a.cikti, index=False)
    print(f"Yazıldı: {a.cikti} ({len(df)} altıgen, toplam nüfus ~{df['nufus'].sum():,.0f}, "
          f"65+ ~{df['yasli_65'].sum():,.0f} = %{100 * df['yasli_65'].sum() / max(df['nufus'].sum(), 1):.1f})")


if __name__ == "__main__":
    main()
