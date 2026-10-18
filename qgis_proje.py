"""
CoolScape AI – QGIS projesi ve harita sayfaları (PyQGIS 3.34).
coolscape_cbs.gpkg katmanlarını stillendirir, 3 baskı düzeni kurar, coolscape_cbs.qgz olarak kaydeder
ve haritaları PDF/PNG olarak dışa aktarır.

Çalıştırma (QGIS kurulu bilgisayarda): QGIS Python konsolunda bu dosyayı açıp çalıştırın
ya da:  python3 qgis_proje.py  (PyQGIS ortamında)
"""

import json
import os
import sys

from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsGraduatedSymbolRenderer, QgsRendererRange,
    QgsFillSymbol, QgsMarkerSymbol, QgsLineSymbol, QgsSymbol, QgsPalLayerSettings, QgsTextFormat,
    QgsTextBufferSettings, QgsVectorLayerSimpleLabeling, QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel,
    QgsLayoutItemLegend, QgsLayoutItemScaleBar, QgsLayoutItemPicture, QgsLayoutPoint, QgsLayoutSize,
    QgsUnitTypes, QgsLayoutExporter, QgsRectangle, QgsLegendStyle, QgsLayerTreeLayer,
)
from qgis.PyQt.QtGui import QColor, QFont

NUFUS_KAYNAK = ""
KLASOR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
GPKG = os.path.join(KLASOR, "coolscape_cbs.gpkg")
CIKTI = os.path.join(KLASOR, "haritalar")
BASLIK_ALT = "CoolScape AI · Antalya kent merkezi (Muratpaşa, Kepez ve çevresi)"
KAYNAK = ("Kaynak: Landsat 8/9 C2 L2 yüzey sıcaklığı (USGS, Microsoft Planetary Computer), H3 altıgenleri; "
          "{nufus}parklar: Google Haritalar koordinatları; tesisler: © OpenStreetMap katkıcıları (ODbL). "
          "Öncelik ağırlıkları ve aday noktalar CoolScape analiz varsayımlarıdır.")


def _f(boyut, kalin=False):
    f = QFont("DejaVu Sans")
    f.setPointSizeF(boyut)
    f.setBold(kalin)
    return f


def baslat():
    if QgsApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        QgsApplication.setPrefixPath("/usr", True)
        app = QgsApplication([], False)
        app.initQgis()
        return app
    return QgsApplication.instance()


def katman(ad, gorunen):
    k = QgsVectorLayer(f"{GPKG}|layername={ad}", gorunen, "ogr")
    return k if k.isValid() else None


def dolgu(renk, saydam=1.0, cizgi=None, kalin=0.0):
    s = QgsFillSymbol.createSimple({"color": renk, "outline_style": "no" if cizgi is None else "solid",
                                    "outline_color": cizgi or "0,0,0,0", "outline_width": str(kalin)})
    s.setOpacity(saydam)
    return s


def etiket(k, alan, boyut=8, renk="#1a1a1a", tampon="#ffffff", kalin=False, ifade=False):
    p = QgsPalLayerSettings()
    p.fieldName = alan
    p.isExpression = ifade
    tf = QgsTextFormat()
    f = QFont("DejaVu Sans")
    f.setBold(kalin)
    tf.setFont(f)
    tf.setSize(boyut)
    tf.setColor(QColor(renk))
    b = QgsTextBufferSettings()
    b.setEnabled(True)
    b.setSize(1.0)
    b.setColor(QColor(tampon))
    tf.setBuffer(b)
    p.setFormat(tf)
    p.placement = QgsPalLayerSettings.OverPoint
    p.yOffset = -3.5
    k.setLabeling(QgsVectorLayerSimpleLabeling(p))
    k.setLabelsEnabled(True)


def stiller(proje):
    L = {}
    osm = None if os.environ.get("ALTLIK_YOK") else QgsRasterLayer("type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
                         "OpenStreetMap (altlık)", "wms")
    if osm is not None and osm.isValid():
        osm.setOpacity(0.55)
        L["osm"] = osm

    # Harita 1: yüzey sıcaklığı farkı (diverging, 0 merkezli)
    lst = katman("oncelik_alanlari", "Yüzey sıcaklığı farkı (°C, kent medyanına göre)")
    araliklar = [(-99, -3, "#2c7bb6", "−3 °C'den serin"), (-3, -1, "#abd9e9", "−3 … −1"),
                 (-1, 1, "#ffffbf", "−1 … +1 (kent ortalaması)"), (1, 3, "#fdae61", "+1 … +3"),
                 (3, 99, "#d7191c", "+3 °C'den sıcak")]
    lst.setRenderer(QgsGraduatedSymbolRenderer("lst_fark", [
        QgsRendererRange(a, b, dolgu(r, 0.85), t) for a, b, r, t in araliklar]))
    L["lst"] = lst

    # Harita 2: öncelik sınıfları
    onc = katman("oncelik_alanlari", "Soğutma önceliği (ağırlıklı çakıştırma)")
    renk = ["#fff7bc", "#fee391", "#fec44f", "#ec7014", "#99000d"]
    adlar = ["Çok düşük", "Düşük", "Orta", "Yüksek", "Çok yüksek"]
    onc.setRenderer(QgsCategorizedSymbolRenderer("sinif_no", [
        QgsRendererCategory(i + 1, dolgu(renk[i], 0.85), adlar[i]) for i in range(5)] + [
        QgsRendererCategory(0, dolgu("#d9d9d9", 0.6), "Seyrek yerleşim (sıralama dışı)")]))
    L["onc"] = onc

    # Sıcak odaklar: kırmızı kontur
    odak = katman("sicak_odaklar", "Yüksek öncelikli odaklar")
    odak.setRenderer(QgsCategorizedSymbolRenderer("sinif", [
        QgsRendererCategory("Çok yüksek", dolgu("153,0,13,140", 1.0, "#99000d", 0.5), "Çok yüksek öncelik"),
        QgsRendererCategory("Yüksek", dolgu("236,112,20,90", 1.0, "#ec7014", 0.3), "Yüksek öncelik")]))
    L["odak"] = odak

    # Park tamponları
    tam = katman("park_tamponlari", "Park erişim halkaları")
    kat = []
    for m, ad, r, st, w in ((400, "5 dk yürüme (400 m)", "#1f3864", "solid", 0.5),
                            (800, "10 dk yürüme (800 m)", "#1f3864", "dash", 0.35),
                            (500, "Soğuma etkisi (500 m)", "#1e88e5", "dot", 0.4)):
        s = QgsFillSymbol.createSimple({"style": "no", "outline_color": r, "outline_style": st,
                                        "outline_width": str(w)})
        kat.append(QgsRendererCategory(m, s, ad))
    tam.setRenderer(QgsCategorizedSymbolRenderer("mesafe_m", kat))
    L["tampon"] = tam

    # Parklar
    pk = katman("parklar", "Pilot parklar")
    pk.setRenderer(QgsCategorizedSymbolRenderer("tur", [QgsRendererCategory(
        "park", QgsMarkerSymbol.createSimple({"name": "circle", "color": "#1e88e5", "size": "3.6",
                                              "outline_color": "#ffffff", "outline_width": "0.6"}), "Park (sulama)")]))
    etiket(pk, "CASE WHEN \"veri_durumu\" <> 'Uydu verisi içinde' THEN \"park\" || ' *' ELSE \"park\" END",
           8, kalin=True, ifade=True)
    L["park"] = pk

    # Aday noktalar
    ad = katman("aday_noktalar", "Aday müdahale noktaları")
    if ad is not None:
        ad.setRenderer(QgsCategorizedSymbolRenderer("yontem", [
            QgsRendererCategory("Sisleme noktası", QgsMarkerSymbol.createSimple(
                {"name": "triangle", "color": "#00acc1", "size": "3.6", "outline_color": "#ffffff",
                 "outline_width": "0.5"}), "Sisleme noktası (aday)"),
            QgsRendererCategory("Yol ıslatma", QgsMarkerSymbol.createSimple(
                {"name": "square", "color": "#7e57c2", "size": "3.2", "outline_color": "#ffffff",
                 "outline_width": "0.5"}), "Yol ıslatma (aday)")]))
        etiket(ad, "ad", 7)
        L["aday"] = ad

    # Hassas tesisler
    ts = katman("hassas_tesisler", "Hassas tesisler (OSM)")
    if ts is not None:
        renk_t = {"Okul / kreş": "#ffffff", "Sağlık": "#e91e63", "Huzurevi": "#8bc34a", "İçme suyu": "#80deea"}
        ts.setRenderer(QgsCategorizedSymbolRenderer("kategori", [QgsRendererCategory(
            k, QgsMarkerSymbol.createSimple({"name": "circle", "color": v, "size": "1.6", "outline_color": "#333333",
                                             "outline_width": "0.2"}), k) for k, v in renk_t.items()]))
        L["tesis"] = ts

    kp = katman("veri_kapsami", "Uydu verisi kapsamı (Landsat sahnesi)")
    if kp is not None:
        kp.setRenderer(QgsCategorizedSymbolRenderer("ad", [QgsRendererCategory(
            "Uydu verisi kapsamı", QgsFillSymbol.createSimple({"style": "no", "outline_color": "#555555",
                                                               "outline_style": "dash", "outline_width": "0.45"}),
            "Uydu verisi sınırı")]))
        L["kapsam"] = kp
    su = katman("su_alanlari", "Deniz / su yüzeyi")
    if su is not None:
        su.setRenderer(QgsCategorizedSymbolRenderer("ad", [QgsRendererCategory(
            "Deniz / su yüzeyi (analiz dışı)", dolgu("#cfdde8", 1.0), "Deniz / su (analiz dışı)")]))
        L["su"] = su

    # Katman ağacı sırası (üstten alta)
    sira = ["aday", "park", "tesis", "tampon", "odak", "kapsam", "onc", "lst", "su", "osm"]
    for anahtar in sira:
        if anahtar in L:
            proje.addMapLayer(L[anahtar], False)
            proje.layerTreeRoot().addLayer(L[anahtar])
    # Başlangıçta yalnızca öncelik haritası görünür olsun
    for anahtar in ("lst", "odak"):
        if anahtar in L:
            proje.layerTreeRoot().findLayer(L[anahtar].id()).setItemVisibilityChecked(False)
    return L


def kapsam(L):
    r = QgsRectangle(L["onc"].extent())
    r.grow(400)
    return r


def duzen(proje, L, ad, baslik, aciklama, anahtarlar, altlik=True):
    lay = QgsPrintLayout(proje)
    lay.initializeDefaults()
    lay.setName(ad)
    sayfa = lay.pageCollection().page(0)
    sayfa.setPageSize("A4", sayfa.Orientation.Landscape)

    harita = QgsLayoutItemMap(lay)
    harita.attemptMove(QgsLayoutPoint(8, 24, QgsUnitTypes.LayoutMillimeters))
    harita.attemptResize(QgsLayoutSize(205, 178, QgsUnitTypes.LayoutMillimeters))
    katmanlar = [L[k] for k in anahtarlar if k in L]
    if altlik and "osm" in L:
        katmanlar.append(L["osm"])
    harita.setLayers(katmanlar)
    harita.setKeepLayerSet(True)
    harita.setCrs(QgsCoordinateReferenceSystem("EPSG:32636"))
    harita.zoomToExtent(kapsam(L))
    harita.setFrameEnabled(True)
    harita.setBackgroundColor(QColor("#f4f6f8"))
    lay.addLayoutItem(harita)

    def yazi(metin, x, y, w, h, boyut, kalin=False, renk="#1f3864"):
        e = QgsLayoutItemLabel(lay)
        e.setText(metin)
        f = QFont("DejaVu Sans")
        f.setPointSizeF(boyut)
        f.setBold(kalin)
        e.setFont(f)
        e.setFontColor(QColor(renk))
        e.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
        e.attemptResize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
        lay.addLayoutItem(e)
        return e

    yazi(baslik, 8, 6, 280, 9, 15, True)
    yazi(BASLIK_ALT, 8, 15, 280, 6, 8.5, False, "#555555")

    lej = QgsLayoutItemLegend(lay)
    lej.setLinkedMap(harita)
    lej.setLegendFilterByMapEnabled(True)
    lej.setAutoUpdateModel(False)
    kok = lej.model().rootGroup()
    kok.removeAllChildren()
    for k in katmanlar:
        if k is not L.get("osm"):
            kok.addLayer(k)
    lej.setTitle("Lejant")
    lej.setStyleFont(QgsLegendStyle.Title, _f(10, True))
    lej.setStyleFont(QgsLegendStyle.Subgroup, _f(8, True))
    lej.setStyleFont(QgsLegendStyle.SymbolLabel, _f(7))
    lej.attemptMove(QgsLayoutPoint(218, 24, QgsUnitTypes.LayoutMillimeters))
    lej.setResizeToContents(True)
    lay.addLayoutItem(lej)

    lej.setSymbolHeight(3.5)
    lej.setSymbolWidth(6)
    yazi(aciklama, 218, 170, 74, 31, 6.6, False, "#333333")

    olcek = QgsLayoutItemScaleBar(lay)
    olcek.setStyle("Single Box")
    olcek.setLinkedMap(harita)
    olcek.setUnits(QgsUnitTypes.DistanceKilometers)
    olcek.setUnitLabel("km")
    olcek.setUnitsPerSegment(1)
    olcek.setNumberOfSegments(4)
    olcek.setNumberOfSegmentsLeft(0)
    olcek.setFont(_f(7))
    olcek.attemptMove(QgsLayoutPoint(12, 190, QgsUnitTypes.LayoutMillimeters))
    lay.addLayoutItem(olcek)

    ok = QgsLayoutItemPicture(lay)
    yol = "/usr/share/qgis/svg/arrows/NorthArrow_02.svg"
    for aday in (yol, os.path.join(QgsApplication.pkgDataPath(), "svg/arrows/NorthArrow_02.svg")):
        if os.path.exists(aday):
            ok.setPicturePath(aday)
            break
    ok.attemptMove(QgsLayoutPoint(198, 27, QgsUnitTypes.LayoutMillimeters))
    ok.attemptResize(QgsLayoutSize(11, 14, QgsUnitTypes.LayoutMillimeters))
    lay.addLayoutItem(ok)

    yazi(KAYNAK.format(nufus=NUFUS_KAYNAK) + " Koordinat sistemi: WGS 84 / UTM 36N (EPSG:32636).", 8, 203, 282, 6, 6, False, "#666666")
    proje.layoutManager().addLayout(lay)
    return lay, harita


def main(altlik_disa_aktar=False):
    app = baslat()
    proje = QgsProject.instance()
    proje.clear()
    proje.setCrs(QgsCoordinateReferenceSystem("EPSG:32636"))
    proje.setTitle("CoolScape AI – CBS analizi")
    L = stiller(proje)
    os.makedirs(CIKTI, exist_ok=True)

    b = json.load(open(os.path.join(KLASOR, "analiz_bilgi.json"), encoding="utf-8"))
    global NUFUS_KAYNAK
    NUFUS_KAYNAK = "nüfus: WorldPop 2025 (CC BY 4.0); " if b.get("nufus_var") else ""
    yildiz = (f" * {', '.join(b['park_disi_veri'])} uydu sahnesinin dışında kalıyor." if b["park_disi_veri"] else "")
    eksik = (f" Veri olmadığı için dışarıda: {', '.join(b['eksik'])}; ağırlıkları diğerlerine dağıtıldı."
             if b.get("eksik") else "")
    sayfalar = [
        ("Harita 1 – Yüzey sıcaklığı", "Harita 1 · Yüzey sıcaklığı farkı (Landsat, 25 Temmuz 2025)",
         f"Kara yüzeyinin sıcaklığının kent medyanından farkı ({b['kara_altigen']} altıgen, ~{b['toplam_km2']} km²). "
         "Kırmızı alanlar daha çok ısınan yapılaşmış bölgeler, mavi alanlar görece serin yeşil ve açık alanlar. "
         f"Deniz ({b['su_altigen']} altıgen) analiz dışıdır. Görüntü yerel sabah geçişidir; göreli ısınma desenini "
         "gösterir." + yildiz, ["kapsam", "park", "lst", "su"]),
        ("Harita 2 – Soğutma önceliği", "Harita 2 · Öncelikli soğutma alanları",
         f"Ağırlıklı çakıştırma: {b['agirlik_metin']}.{eksik} Yerleşim alanı: {b['yerlesim_kurali']}; "
         f"~{b['yerlesim_km2']} km². Sınıflar bu alanın beşte birlik dilimleridir; gri alanlar sıralama dışı." + yildiz,
         ["aday", "park", "tesis", "tampon", "onc", "su"]),
        ("Harita 3 – Müdahale planı", "Harita 3 · Park erişim boşluğu ve aday müdahale noktaları",
         f"Parkların 5 dakikalık (400 m) yürüme alanı yerleşim alanının yalnızca %{b['park_400_kapsam_pct']:.1f}'ini "
         f"kapsıyor; yüksek ve çok yüksek öncelikli {b['yuksek_km2']} km²'nin %{b['yuksek_park_disi_pct']:.0f}'i bu "
         "erişimin dışında. CoolScape bu alanlar için sıcak noktalardan seçilen aday sisleme ve yol ıslatma "
         "noktaları önerir; kesin yerler sahada belediyeyle belirlenir.", ["aday", "park", "tampon", "odak", "kapsam", "su"]),
    ]
    for ad, baslik, aciklama, anahtarlar in sayfalar:
        lay, harita = duzen(proje, L, ad, baslik, aciklama, anahtarlar, altlik=altlik_disa_aktar)
        dosya = ad.split(" – ")[0].replace(" ", "_").lower()
        print("dışa aktarılıyor:", ad, flush=True)
        ex = QgsLayoutExporter(lay)
        ex.exportToPdf(os.path.join(CIKTI, f"{dosya}.pdf"), QgsLayoutExporter.PdfExportSettings())
        if not altlik_disa_aktar and "osm" in L:  # projede altlık açık kalsın
            harita.setLayers(harita.layers() + [L["osm"]])
    proje.write(os.path.join(KLASOR, "coolscape_cbs.qgz"))
    print("Kaydedildi: coolscape_cbs.qgz ve haritalar/")


if __name__ == "__main__":
    main(altlik_disa_aktar="--altlik" in sys.argv)
