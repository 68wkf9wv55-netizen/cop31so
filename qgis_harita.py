"""
CoolScape AI – klasik CBS tematik haritaları (PyQGIS 3.34).
coolscape_cbs.gpkg katmanlarından (cbs_hazirla.py + cbs_poligon.py) üç A4 harita üretir:
düzgün sınırlı renkli bölgeler, enlem-boylam koordinat çerçevesi, "İŞARETLER" lejantı, taramalar,
nokta sembolleri, yer adları, kuzey oku, ölçek ve çerçeveli başlık. Projeyi coolscape_cbs.qgz olarak kaydeder.

Çalıştırma: QGIS Python konsolunda bu dosyayı açıp çalıştırın ya da PyQGIS ortamında: python3 qgis_harita.py
"""

import json
import os

from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsSingleSymbolRenderer, QgsFillSymbol, QgsMarkerSymbol,
    QgsLinePatternFillSymbolLayer, QgsPointPatternFillSymbolLayer, QgsSimpleLineSymbolLayer, QgsLineSymbol,
    QgsPalLayerSettings, QgsTextFormat, QgsTextBufferSettings, QgsVectorLayerSimpleLabeling,
    QgsPrintLayout, QgsLayoutItemMap, QgsLayoutItemLabel, QgsLayoutItemLegend, QgsLayoutItemScaleBar,
    QgsLayoutItemPicture, QgsLayoutItemShape, QgsLayoutItemMapGrid, QgsLayoutPoint, QgsLayoutSize, QgsUnitTypes,
    QgsLayoutExporter, QgsRectangle, QgsLegendStyle, QgsSimpleFillSymbolLayer,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont

KLASOR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
GPKG = os.path.join(KLASOR, "coolscape_cbs.gpkg")
CIKTI = os.path.join(KLASOR, "haritalar")
KIRMIZI = "#b71c1c"


def baslat():
    if QgsApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        QgsApplication.setPrefixPath("/usr", True)
        a = QgsApplication([], False)
        a.initQgis()
        return a
    return QgsApplication.instance()


def font(boyut, kalin=False, italik=False, aile="DejaVu Sans"):
    f = QFont(aile)
    f.setPointSizeF(boyut)
    f.setBold(kalin)
    f.setItalic(italik)
    return f


def metin_bicimi(boyut, renk="#111111", kalin=False, italik=False, tampon="#ffffff", aralik=0.0):
    tf = QgsTextFormat()
    f = font(boyut, kalin, italik)
    f.setLetterSpacing(QFont.AbsoluteSpacing, aralik)
    tf.setFont(f)
    tf.setSize(boyut)
    tf.setColor(QColor(renk))
    if tampon:
        b = QgsTextBufferSettings()
        b.setEnabled(True)
        b.setSize(0.9)
        b.setColor(QColor(tampon))
        tf.setBuffer(b)
    return tf


def katman(ad, gorunen):
    k = QgsVectorLayer(f"{GPKG}|layername={ad}", gorunen, "ogr")
    return k if k.isValid() else None


def dolgu(renk, cizgi="#1a1a1a", kalin=0.15, saydam=1.0):
    s = QgsFillSymbol.createSimple({"color": renk, "outline_color": cizgi, "outline_width": str(kalin),
                                    "outline_style": "solid" if cizgi else "no"})
    s.setOpacity(saydam)
    return s


def tarama(taban, cizgi_renk, aci=45, aralik=1.4, kalin=0.18, kontur="#1a1a1a", kontur_kalin=0.15):
    s = dolgu(taban, kontur, kontur_kalin)
    lp = QgsLinePatternFillSymbolLayer()
    lp.setLineAngle(aci)
    lp.setDistance(aralik)
    lp.setLineWidth(kalin)
    lp.setColor(QColor(cizgi_renk))
    s.appendSymbolLayer(lp)
    return s


def noktali(taban, nokta_renk, aralik=1.6, boyut=0.45, kontur="#1a1a1a"):
    s = dolgu(taban, kontur, 0.15)
    pp = QgsPointPatternFillSymbolLayer()
    pp.setDistanceX(aralik)
    pp.setDistanceY(aralik)
    m = QgsMarkerSymbol.createSimple({"name": "circle", "color": nokta_renk, "size": str(boyut), "outline_style": "no"})
    pp.setSubSymbol(m)
    s.appendSymbolLayer(pp)
    return s


def etiket(k, alan, tf, ifade=False, yer=QgsPalLayerSettings.OverPoint, dy=-3.0):
    p = QgsPalLayerSettings()
    p.fieldName = alan
    p.isExpression = ifade
    p.setFormat(tf)
    p.placement = yer
    p.yOffset = dy
    p.priority = 10
    try:  # QGIS 3.26+: etiketler çakışsa da gösterilsin (nokta adları kaybolmasın)
        from qgis.core import Qgis, QgsLabelPlacementSettings
        ps = p.placementSettings()
        ps.setOverlapHandling(Qgis.LabelOverlapHandling.AllowOverlapIfRequired)
        p.setPlacementSettings(ps)
    except Exception:
        p.displayAll = True
    k.setLabeling(QgsVectorLayerSimpleLabeling(p))
    k.setLabelsEnabled(True)


def katmanlar():
    L = {}
    deniz = katman("deniz", "Deniz")
    deniz.setRenderer(QgsSingleSymbolRenderer(tarama("#d9ecf7", "#9cc7e3", aci=0, aralik=2.2, kalin=0.12,
                                                     kontur="#5b9bc8", kontur_kalin=0.3)))
    L["deniz"] = deniz

    lst = katman("lst_sinif_poligon", "YÜZEY SICAKLIĞI FARKI (°C)")
    renk = {1: "#2166ac", 2: "#92c5de", 3: "#fff7bc", 4: "#f4a582", 5: "#d6604d"}
    adlar = {1: "3 °C'den fazla serin", 2: "1–3 °C serin", 3: "Kent ortalaması (±1 °C)", 4: "1–3 °C sıcak",
             5: "3 °C'den fazla sıcak"}
    lst.setRenderer(QgsCategorizedSymbolRenderer("lst_sinif", [
        QgsRendererCategory(i, dolgu(renk[i]), adlar[i]) for i in (5, 4, 3, 2, 1)]))
    L["lst"] = lst

    onc = katman("oncelik_poligon", "SOĞUTMA ÖNCELİĞİ (ağırlıklı çakıştırma)")
    renk = {5: "#99000d", 4: "#e6550d", 3: "#fdae6b", 2: "#fdd49e", 1: "#fff5e6"}
    adlar = {5: "Çok yüksek", 4: "Yüksek", 3: "Orta", 2: "Düşük", 1: "Çok düşük"}
    kat = [QgsRendererCategory(i, dolgu(renk[i]), adlar[i]) for i in (5, 4, 3, 2, 1)]
    kat.append(QgsRendererCategory(0, tarama("#ffffff", "#9e9e9e", 45, 1.3, 0.15), "Seyrek yerleşim (sıralama dışı)"))
    onc.setRenderer(QgsCategorizedSymbolRenderer("sinif_no", kat))
    L["onc"] = onc

    od = katman("odak_poligon", "YÜKSEK ÖNCELİKLİ ODAKLAR")
    od.setRenderer(QgsCategorizedSymbolRenderer("grup", [
        QgsRendererCategory("Çok yüksek öncelik", tarama("#e57373", "#7f0000", 45, 1.1, 0.25, "#7f0000", 0.35),
                            "Çok yüksek öncelik"),
        QgsRendererCategory("Yüksek öncelik", noktali("#ffe0b2", "#e65100", 1.5, 0.5, "#e65100"),
                            "Yüksek öncelik")]))
    L["odak"] = od

    ca = katman("calisma_alani", "Çalışma alanı sınırı")
    s = QgsFillSymbol.createSimple({"style": "no", "outline_color": "#000000", "outline_width": "0.7"})
    ca.setRenderer(QgsSingleSymbolRenderer(s))
    L["sinir"] = ca

    tam = katman("park_tamponlari", "PARK ERİŞİMİ")
    s4 = QgsFillSymbol.createSimple({"style": "no", "outline_color": "#1b5e20", "outline_style": "dash",
                                     "outline_width": "0.45"})
    s8 = QgsFillSymbol.createSimple({"style": "no", "outline_color": "#1b5e20", "outline_style": "dot",
                                     "outline_width": "0.35"})
    tam.setRenderer(QgsCategorizedSymbolRenderer("mesafe_m", [
        QgsRendererCategory(400, s4, "5 dk yürüme (400 m)"), QgsRendererCategory(800, s8, "10 dk yürüme (800 m)")]))
    tam.setSubsetString('"mesafe_m" IN (400, 800)')
    L["tampon"] = tam

    ts = katman("hassas_tesisler", "TESİSLER (OpenStreetMap)")
    if ts is not None:
        ts.setRenderer(QgsCategorizedSymbolRenderer("kategori", [
            QgsRendererCategory("Okul / kreş", QgsMarkerSymbol.createSimple(
                {"name": "diamond", "color": "#111111", "size": "1.9", "outline_style": "no"}), "Okul / kreş"),
            QgsRendererCategory("Sağlık", QgsMarkerSymbol.createSimple(
                {"name": "cross_fill", "color": "#d32f2f", "size": "2.4", "outline_color": "#ffffff",
                 "outline_width": "0.15"}), "Hastane / sağlık merkezi"),
            QgsRendererCategory("İçme suyu", QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": "#1565c0", "size": "1.5", "outline_color": "#ffffff",
                 "outline_width": "0.15"}), "Çeşme / içme suyu")]))
        L["tesis"] = ts

    pk = katman("parklar", "PİLOT PARKLAR")
    pk.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple(
        {"name": "circle", "color": "#2e7d32", "size": "3.6", "outline_color": "#000000", "outline_width": "0.35"})))
    pk.renderer().symbol()  # tek sembol
    etiket(pk, "park", metin_bicimi(7.5, "#0b3d0b", kalin=True), dy=-3.6)
    L["park"] = pk

    ad = katman("aday_noktalar", "ADAY MÜDAHALE NOKTALARI")
    if ad is not None:
        ad.setRenderer(QgsCategorizedSymbolRenderer("yontem", [
            QgsRendererCategory("Sisleme noktası", QgsMarkerSymbol.createSimple(
                {"name": "triangle", "color": "#00838f", "size": "3.4", "outline_color": "#000000",
                 "outline_width": "0.3"}), "Sisleme noktası (aday)"),
            QgsRendererCategory("Yol ıslatma", QgsMarkerSymbol.createSimple(
                {"name": "square", "color": "#6a1b9a", "size": "2.8", "outline_color": "#000000",
                 "outline_width": "0.3"}), "Yol ıslatma (aday)")]))
        etiket(ad, "replace(replace(\"ad\", ' (aday)', ''), 'Islatma', 'Islatma')", metin_bicimi(6.5, "#222222"),
               ifade=True, dy=-3.0)
        L["aday"] = ad

    ya = katman("yer_adlari", "Yer adları")
    p = QgsPalLayerSettings()
    p.fieldName = "ad"
    p.placement = QgsPalLayerSettings.OverPoint
    p.setFormat(metin_bicimi(11, "#3a3a3a", kalin=True, aralik=2.5))
    # türe göre biçim: kural tabanlı yerine veri tanımlı renk/italik
    from qgis.core import QgsPropertyCollection, QgsProperty
    pc = p.dataDefinedProperties()
    pc.setProperty(QgsPalLayerSettings.Color, QgsProperty.fromExpression(
        "CASE WHEN \"tur\"='deniz' THEN '#2b6c99' WHEN \"tur\"='komsu' THEN '#777777' ELSE '#3a3a3a' END"))
    pc.setProperty(QgsPalLayerSettings.Italic, QgsProperty.fromExpression("\"tur\"='deniz'"))
    pc.setProperty(QgsPalLayerSettings.Size, QgsProperty.fromExpression(
        "CASE WHEN \"tur\"='ilce' THEN 12 ELSE 10 END"))
    p.setDataDefinedProperties(pc)
    ya.setLabeling(QgsVectorLayerSimpleLabeling(p))
    ya.setLabelsEnabled(True)
    ya.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({"name": "circle", "size": "0",
                                                                          "outline_style": "no",
                                                                          "color": "0,0,0,0"})))
    L["yer"] = ya
    return L


def duzen(proje, L, ad, baslik, alt_baslik, aciklama, anahtarlar):
    lay = QgsPrintLayout(proje)
    lay.initializeDefaults()
    lay.setName(ad)
    sayfa = lay.pageCollection().page(0)
    sayfa.setPageSize("A4", sayfa.Orientation.Landscape)

    harita = QgsLayoutItemMap(lay)
    harita.attemptMove(QgsLayoutPoint(14, 10, QgsUnitTypes.LayoutMillimeters))
    harita.attemptResize(QgsLayoutSize(200, 182, QgsUnitTypes.LayoutMillimeters))
    kat = [L[k] for k in anahtarlar if k in L]
    harita.setLayers(kat)
    harita.setKeepLayerSet(True)
    harita.setCrs(QgsCoordinateReferenceSystem("EPSG:32636"))
    r = QgsRectangle(L["sinir"].extent())
    r.grow(700)
    harita.zoomToExtent(r)
    harita.setFrameEnabled(True)
    harita.setFrameStrokeWidth(QgsLayoutMeasurement_mm(0.5))
    harita.setBackgroundColor(QColor("#ffffff"))
    lay.addLayoutItem(harita)

    grid = QgsLayoutItemMapGrid("Coğrafi koordinatlar", harita)
    grid.setCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    grid.setIntervalX(0.03)
    grid.setIntervalY(0.02)
    grid.setStyle(QgsLayoutItemMapGrid.FrameAnnotationsOnly)
    grid.setFrameStyle(QgsLayoutItemMapGrid.ExteriorTicks)
    grid.setFrameWidth(1.8)
    grid.setFramePenSize(0.3)
    grid.setAnnotationEnabled(True)
    grid.setAnnotationFormat(QgsLayoutItemMapGrid.DegreeMinuteSecond)
    grid.setAnnotationPrecision(0)
    grid.setAnnotationTextFormat(metin_bicimi(6.5, "#222222", tampon=None))
    for kenar in (QgsLayoutItemMapGrid.Left, QgsLayoutItemMapGrid.Right, QgsLayoutItemMapGrid.Top,
                  QgsLayoutItemMapGrid.Bottom):
        grid.setAnnotationPosition(QgsLayoutItemMapGrid.OutsideMapFrame, kenar)
        grid.setAnnotationDirection(QgsLayoutItemMapGrid.Vertical if kenar in (
            QgsLayoutItemMapGrid.Left, QgsLayoutItemMapGrid.Right) else QgsLayoutItemMapGrid.Horizontal, kenar)
    grid.setAnnotationFrameDistance(1.0)
    harita.grids().addGrid(grid)

    # Çerçeveli başlık kutusu (sol üst, haritanın içinde)
    kutu = QgsLayoutItemShape(lay)
    kutu.setShapeType(QgsLayoutItemShape.Rectangle)
    kutu.attemptMove(QgsLayoutPoint(16.5, 12.5, QgsUnitTypes.LayoutMillimeters))
    kutu.attemptResize(QgsLayoutSize(86, 25, QgsUnitTypes.LayoutMillimeters))
    kutu.setSymbol(QgsFillSymbol.createSimple({"color": "#ffffff", "outline_color": KIRMIZI, "outline_width": "0.7"}))
    lay.addLayoutItem(kutu)

    def yazi(metin, x, y, w, h, boyut, kalin=False, renk="#111111", hizala=Qt.AlignLeft, italik=False):
        e = QgsLayoutItemLabel(lay)
        e.setText(metin)
        e.setTextFormat(metin_bicimi(boyut, renk, kalin, italik, tampon=None))
        e.setHAlign(hizala)
        e.attemptMove(QgsLayoutPoint(x, y, QgsUnitTypes.LayoutMillimeters))
        e.attemptResize(QgsLayoutSize(w, h, QgsUnitTypes.LayoutMillimeters))
        lay.addLayoutItem(e)
        return e

    yazi("COOLSCAPE AI", 18.5, 13.5, 82, 8, 15, True, KIRMIZI)
    yazi(baslik, 18.5, 21.5, 82, 7, 9.5, True, "#111111")
    yazi(alt_baslik, 18.5, 28.5, 82, 8, 6.8, False, "#444444")

    # İŞARETLER lejantı (sağ, çerçeveli)
    lej = QgsLayoutItemLegend(lay)
    lej.setLinkedMap(harita)
    lej.setAutoUpdateModel(False)
    kok = lej.model().rootGroup()
    kok.removeAllChildren()
    for k in kat:
        if k.name() not in ("Yer adları", "Çalışma alanı sınırı", "Deniz"):
            kok.addLayer(k)
    kok.addLayer(L["deniz"])
    kok.addLayer(L["sinir"])
    lej.setTitle("İŞARETLER")
    lej.setStyleFont(QgsLegendStyle.Title, font(13, True))
    lej.setStyleFont(QgsLegendStyle.Subgroup, font(7.5, True))
    lej.setStyleFont(QgsLegendStyle.SymbolLabel, font(6.8))
    lej.setSymbolHeight(3.6)
    lej.setSymbolWidth(7)
    lej.setFrameEnabled(True)
    lej.setFrameStrokeWidth(QgsLayoutMeasurement_mm(0.5))
    lej.setBackgroundColor(QColor("#ffffff"))
    lej.setBoxSpace(2.5)
    lej.setColumnSpace(2)
    lej.attemptMove(QgsLayoutPoint(222, 10, QgsUnitTypes.LayoutMillimeters))
    lej.setResizeToContents(True)
    lay.addLayoutItem(lej)

    # Açıklama kutusu (sağ alt)
    kutu2 = QgsLayoutItemShape(lay)
    kutu2.setShapeType(QgsLayoutItemShape.Rectangle)
    kutu2.attemptMove(QgsLayoutPoint(222, 152, QgsUnitTypes.LayoutMillimeters))
    kutu2.attemptResize(QgsLayoutSize(68, 40, QgsUnitTypes.LayoutMillimeters))
    kutu2.setSymbol(QgsFillSymbol.createSimple({"color": "#ffffff", "outline_color": "#000000", "outline_width": "0.5"}))
    lay.addLayoutItem(kutu2)
    yazi(aciklama, 224, 154, 64, 37, 6.3, False, "#222222")

    # Kuzey oku ve ölçek (sol alt, haritanın içinde)
    ok = QgsLayoutItemPicture(lay)
    for aday in ("/usr/share/qgis/svg/arrows/NorthArrow_11.svg", "/usr/share/qgis/svg/arrows/NorthArrow_02.svg",
                 os.path.join(QgsApplication.pkgDataPath(), "svg/arrows/NorthArrow_11.svg")):
        if os.path.exists(aday):
            ok.setPicturePath(aday)
            break
    ok.attemptMove(QgsLayoutPoint(18, 166, QgsUnitTypes.LayoutMillimeters))
    ok.attemptResize(QgsLayoutSize(10, 14, QgsUnitTypes.LayoutMillimeters))
    lay.addLayoutItem(ok)
    yazi("N", 20.5, 161.5, 6, 5, 9, True, "#000000", Qt.AlignHCenter)

    olcek = QgsLayoutItemScaleBar(lay)
    olcek.setStyle("Single Box")
    olcek.setLinkedMap(harita)
    olcek.setUnits(QgsUnitTypes.DistanceKilometers)
    olcek.setUnitLabel("km")
    olcek.setUnitsPerSegment(1)
    olcek.setNumberOfSegments(4)
    olcek.setNumberOfSegmentsLeft(0)
    olcek.setTextFormat(metin_bicimi(6.5, "#000000", tampon="#ffffff"))
    olcek.setHeight(2)
    olcek.attemptMove(QgsLayoutPoint(32, 182, QgsUnitTypes.LayoutMillimeters))
    lay.addLayoutItem(olcek)

    yazi("Kaynak: Landsat 8/9 C2 L2 yüzey sıcaklığı, 25 Temmuz 2025 (USGS / Microsoft Planetary Computer); tesisler © "
         "OpenStreetMap katkıcıları (ODbL); nüfus: WorldPop 2025, CC BY 4.0 (65+ ilçe oranıyla); "
         "parklar: Google Haritalar koordinatları. İlçe adları yönlendirme amaçlıdır, "
         "sınır verisi değildir. Projeksiyon: WGS 84 / UTM 36N (EPSG:32636). CoolScape AI, Siemens Türkiye Climathon 2026.",
         14, 199, 276, 8, 5.6, False, "#555555")
    proje.layoutManager().addLayout(lay)
    return lay


def QgsLayoutMeasurement_mm(v):
    from qgis.core import QgsLayoutMeasurement
    return QgsLayoutMeasurement(v, QgsUnitTypes.LayoutMillimeters)


def main():
    global _UYG
    _UYG = baslat()  # referans tutulmazsa uygulama nesnesi silinir
    proje = QgsProject.instance()
    proje.clear()
    proje.setCrs(QgsCoordinateReferenceSystem("EPSG:32636"))
    proje.setTitle("CoolScape AI – CBS tematik haritaları")
    L = katmanlar()
    sira = ["yer", "aday", "park", "tesis", "tampon", "sinir", "odak", "onc", "lst", "deniz"]
    for k in sira:
        if k in L:
            proje.addMapLayer(L[k], False)
            proje.layerTreeRoot().addLayer(L[k])
    for k in ("lst", "odak", "tesis"):
        if k in L:
            proje.layerTreeRoot().findLayer(L[k].id()).setItemVisibilityChecked(False)
    osm = QgsRasterLayer("type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
                         "OpenStreetMap (isteğe bağlı altlık)", "wms")
    if osm.isValid():
        proje.addMapLayer(osm, False)
        proje.layerTreeRoot().addLayer(osm).setItemVisibilityChecked(False)

    b = json.load(open(os.path.join(KLASOR, "analiz_bilgi.json"), encoding="utf-8"))
    os.makedirs(CIKTI, exist_ok=True)
    alt = "Antalya kent merkezi (Muratpaşa, Kepez ve çevresi) · Ölçek 1:60.000 civarı"
    sayfalar = [
        ("Harita 1 – Yüzey sıcaklığı", "YÜZEY SICAKLIĞI FARKI HARİTASI",
         "Landsat 8/9, 25 Temmuz 2025 · kentin ortalamasına göre",
         f"Kara yüzeyinin sıcaklığının kent medyanından farkı. Kırmızı tonlar daha çok ısınan yapılaşmış alanları, mavi "
         f"tonlar görece serin yeşil ve açık alanları gösterir. Analiz alanı ~{b['toplam_km2']:.0f} km²; deniz analiz "
         "dışıdır. Görüntü yerel sabah geçişidir; göreli ısınma desenini gösterir.",
         ["yer", "park", "sinir", "lst", "deniz"]),
        ("Harita 2 – Soğutma önceliği", "SOĞUTMA ÖNCELİĞİ HARİTASI",
         "Ağırlıklı çakıştırma analizi · CoolScape AI",
         f"Ölçütler ve ağırlıklar: {b['agirlik_metin']}. Yerleşim alanı: {b.get('yerlesim_kurali', '')} "
         f"(~{b.get('yerlesim_km2', 0):.0f} km²); taralı alanlar sıralama dışıdır. Sınıflar yerleşim alanının beşte "
         "birlik dilimleridir. Ağırlıklar uzman varsayımıdır.",
         ["yer", "aday", "park", "tesis", "tampon", "sinir", "onc", "deniz"]),
        ("Harita 3 – Müdahale planı", "ERİŞİM BOŞLUĞU VE MÜDAHALE PLANI",
         "Yüksek öncelikli odaklar ve aday sisleme / yol ıslatma noktaları",
         f"Beş parkın 5 dakikalık (400 m) yürüme alanı yerleşim alanının yalnızca %{str(round(b['park_400_kapsam_pct'], 1)).replace('.', ',')}'ini "
         f"kapsıyor; yüksek ve çok yüksek öncelikli {b['yuksek_km2']:.0f} km²'nin %{b['yuksek_park_disi_pct']:.0f}'i bu "
         "erişimin dışında. Aday noktalar bu odaklardan seçilmiştir; kesin yerler sahada belediyeyle belirlenir.",
         ["yer", "aday", "park", "tampon", "sinir", "odak", "deniz"]),
    ]
    for ad, baslik, alt_b, aciklama, anahtarlar in sayfalar:
        lay = duzen(proje, L, ad, baslik, alt_b, aciklama, anahtarlar)
        dosya = ad.split(" – ")[0].replace(" ", "_").lower()
        print("dışa aktarılıyor:", ad, flush=True)
        QgsLayoutExporter(lay).exportToPdf(os.path.join(CIKTI, f"{dosya}.pdf"), QgsLayoutExporter.PdfExportSettings())
    proje.write(os.path.join(KLASOR, "coolscape_cbs.qgz"))
    print("Kaydedildi: coolscape_cbs.qgz ve haritalar/")


if __name__ == "__main__":
    main()
