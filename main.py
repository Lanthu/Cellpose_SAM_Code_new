"""
╔══════════════════════════════════════════════════════════════════════╗
║       Cellpose-SAM  →  Elongation Index  |  Vollständige Pipeline   ║
╚══════════════════════════════════════════════════════════════════════╝
 
Workflow:
  1. Alle .tif-Bilder aus einem Eingabeordner laden
  2. Segmentierung mit Cellpose-SAM (lokal, GPU-beschleunigt)
  3. Segmentierungsbilder als PNG speichern (visuelle Kontrolle)
  4. Elongation Index (a−b)/(a+b) pro Zelle berechnen
  5. CSV-Tabelle + Plot ausgeben
 
Installation (einmalig, in VS Code Terminal):
  pip install cellpose scikit-image tifffile pandas matplotlib pillow
 
Für GPU-Unterstützung zusätzlich das passende PyTorch installieren:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  (cu121 = CUDA 12.1 – prüfe deine CUDA-Version mit: nvidia-smi)
"""
 
# ──────────────────────────────────────────────────────────────────────
# EINSTELLUNGEN  ←  Hier anpassen!
# ──────────────────────────────────────────────────────────────────────
 
# Ordner mit deinen .tif-Originalbildern
# INPUT_ORDNER = r"C:\Users\Biofluid\Documents\Ole\Masterarbeit\Ergebnisse\Rohdaten_und_Ergebnisse\HemoMT09.06.26"
INPUT_ORDNER = r"C:\Users\Biofluid\Downloads\Henris Erys"
 
# Ausgabeordner (wird automatisch erstellt)
# OUTPUT_ORDNER = r"C:\Users\Biofluid\Documents\Ole\Masterarbeit\Ergebnisse\Rohdaten_und_Ergebnisse\Cellpose_ErgebnisseHemoMT09.06.26"
OUTPUT_ORDNER = r"C:\Users\Biofluid\Documents\Ole\Masterarbeit\Ergebnisse\Rohdaten_und_Ergebnisse\Cellpose_Ergebnisse_Henri"
 
# ── Cellpose-Parameter ─────────────────────────────────────────────────
 
MODELL = "cpsam"
# Welches Cellpose-Modell wird verwendet?
# "cpsam"  → Cellpose-SAM (neuestes, bestes Modell – empfohlen)
# "cyto3"  → Vorherige Generation, gut für runde Zellen
# "cyto2"  → Älteres Modell
# Beim ersten Start wird das Modell automatisch heruntergeladen.
 
KANAL_ZELLE = 0
# In welchem Kanal sind deine Erythrozyten sichtbar?
# 0 = Graustufenbild ODER alle Kanäle werden gemittelt (Standard)
# 1 = Rot-Kanal, 2 = Grün-Kanal, 3 = Blau-Kanal
# Für typische Phasenkontrastbilder von Erythrozyten: 0
 
KANAL_KERN = 0
# In welchem Kanal sind Zellkerne sichtbar? (für Erythrozyten: 0 = kein Kern)
# 0 = kein Kernkanal vorhanden
# Nur relevant wenn du fluoreszenz-Dual-Channel-Bilder hast
 
DURCHMESSER = None
# Geschätzter Zelldurchmesser in Pixeln.
# None  → Cellpose-SAM schätzt automatisch (empfohlen für cpsam)
# z.B. 30  → wenn die automatische Schätzung schlecht ist
# Tipp: Miss in deinem Bild ein paar Zellen per Hand aus
 
FLOW_THRESHOLD = 0.4
# Wie streng ist die Qualitätskontrolle der Segmentierung?
# Cellpose berechnet intern Flussfelder; dieser Wert filtert
# Masken heraus, deren Flussfelder zu ungenau sind.
# 0.0  → sehr streng (weniger, aber sicherere Masken)
# 0.4  → Standard (gute Balance)
# 0.9  → locker (mehr Masken, aber mehr Fehler möglich)
 
CELLPROB_THRESHOLD = 0.0
# Schwellenwert für die Zellwahrscheinlichkeit.
# Bestimmt, ab welcher Wahrscheinlichkeit ein Pixel als Zelle gilt.
# 0.0   → Standard
# -1.0  → mehr Zellen werden erkannt (sensibler, mehr false positives)
# +1.0  → nur sehr eindeutige Zellen (weniger, aber präziser)
 
GPU = True
# True  → NVIDIA-GPU verwenden (viel schneller, GTX reicht)
# False → CPU verwenden (langsamer, aber immer möglich)
# Cellpose prüft automatisch ob eine GPU verfügbar ist
 
# ── Elongation Index Parameter ─────────────────────────────────────────
 
MIN_PIXEL = 500
# Minimale Zellgröße in Pixeln – alles darunter wird als Artefakt
# ignoriert. Passe diesen Wert an deine Bildauflösung an.
# Bei hochauflösenden Mikroskopbildern eher 500-2000,
# bei kleineren Bildern eher 50-200.
 
# ──────────────────────────────────────────────────────────────────────
# AB HIER NICHTS MEHR ÄNDERN
# ──────────────────────────────────────────────────────────────────────
 
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from skimage.measure import regionprops, label
from skimage.color import label2rgb
from tifffile import imread as tiff_read
import warnings
warnings.filterwarnings("ignore")
 
 
# ──────────────────────────────────────────────
# HILFSFUNKTION: Bild laden & normalisieren
# ──────────────────────────────────────────────
 
def lade_bild(pfad):
    """Lädt ein Bild (TIF, TIFF, PNG, BMP) und gibt es als numpy-Array zurück."""
    ext = os.path.splitext(pfad)[1].lower()

    # Format-abhängiger Loader
    if ext in (".tif", ".tiff"):
        img = tiff_read(pfad)
    else:
        # PNG, BMP, JPG etc. → Pillow verwenden
        from PIL import Image
        img = np.array(Image.open(pfad))

    # Falls 3D-Stack (z, y, x) → erste Ebene nehmen
    if img.ndim == 3 and img.shape[0] < 10 and img.shape[2] > 10:
        print(f"    ⚠ 3D-Stack erkannt ({img.shape[0]} Ebenen) → verwende erste Ebene")
        img = img[0]

    # RGBA (4 Kanäle, z.B. aus Paint) → RGB konvertieren
    if img.ndim == 3 and img.shape[2] == 4:
        print(f"    ⚠ RGBA-Bild erkannt → Alpha-Kanal wird entfernt")
        img = img[:, :, :3]

    # Float → uint8 normalisieren falls nötig
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    return img
 
 
# ──────────────────────────────────────────────
# SCHRITT 1: CELLPOSE-SAM SEGMENTIERUNG
# ──────────────────────────────────────────────
 
def segmentiere_bilder(bild_pfade, ausgabe_masken_ordner):
    """
    Führt Cellpose-SAM Segmentierung auf allen Bildern durch.
    Gibt ein Dict {dateipfad: masken_array} zurück.
    """
    from cellpose import models
 
    print("\n" + "═"*60)
    print("  SCHRITT 1: Cellpose-SAM Segmentierung")
    print("═"*60)
 
    # Modell laden
    print(f"\n  Lade Modell: '{MODELL}' (GPU={'ja' if GPU else 'nein'})")
    modell = models.CellposeModel(
        gpu=GPU,
        pretrained_model=MODELL   # "cpsam", "cyto3", etc.
    )
    print("  ✓ Modell geladen\n")
 
    os.makedirs(ausgabe_masken_ordner, exist_ok=True)
    alle_masken = {}
 
    for i, pfad in enumerate(bild_pfade):
        dateiname = os.path.basename(pfad)
        print(f"  [{i+1}/{len(bild_pfade)}] {dateiname}")
 
        # Bild laden
        img = lade_bild(pfad)
        print(f"    Bildgröße: {img.shape}, dtype: {img.dtype}")
 
        # ── Cellpose-Segmentierung ──
        masken, flows, styles = modell.eval(
            img,
 
            diameter=DURCHMESSER,
            # Zelldurchmesser in Pixeln.
            # None = automatische Schätzung durch Cellpose-SAM
 
            channels=[KANAL_ZELLE, KANAL_KERN],
            # [Zellkanal, Kernkanal]
            # [0, 0] = Graustufenbild, kein Kernkanal
 
            flow_threshold=FLOW_THRESHOLD,
            # Qualitätsschwelle für Flussfeld-Konsistenz
 
            cellprob_threshold=CELLPROB_THRESHOLD,
            # Wahrscheinlichkeitsschwelle für Zellpixel
 
            normalize=True,
            # True = Bild wird automatisch normalisiert (Perzentil-basiert)
            # Empfohlen für Mikroskopiebilder mit variablen Helligkeiten
 
            augment=False,
            # True = Test-Time-Augmentation (4x langsamer, manchmal genauer)
            # False = Standard, für Batch-Verarbeitung empfohlen
 
            #tile=True,
            # True  = große Bilder werden in Kacheln aufgeteilt
            # False = Bild wird als Ganzes verarbeitet (mehr VRAM nötig)
 
            #tile_overlap=0.1,
            # Überlappung der Kacheln (0.0–0.5)
            # 0.1 = 10% Überlappung, verhindert Artefakte an Kachelrändern
            # Höher = weniger Artefakte, aber langsamer
        )
 
        n_zellen = len(np.unique(masken)) - 1
        print(f"    → {n_zellen} Zellen erkannt")
 
        alle_masken[pfad] = masken
 
        # Masken als TIF speichern (für EI-Berechnung später nutzbar)
        basis = os.path.splitext(dateiname)[0]
        masken_pfad = os.path.join(ausgabe_masken_ordner, f"{basis}_masks.tif")
        from tifffile import imwrite
        imwrite(masken_pfad, masken.astype(np.int32))
 
    print(f"\n  ✓ Segmentierung abgeschlossen – Masken gespeichert in:\n    {ausgabe_masken_ordner}")
    return alle_masken
 
 
# ──────────────────────────────────────────────
# SCHRITT 2: SEGMENTIERUNG VISUELL AUSGEBEN
# ──────────────────────────────────────────────
 
def speichere_overlay_bilder(bild_pfade, alle_masken, ausgabe_overlay_ordner):
    """
    Erstellt für jedes Bild ein Overlay-PNG:
    Originalbild + farbige Masken + rote Umrandungen.
    """
    print("\n" + "═"*60)
    print("  SCHRITT 2: Overlay-Bilder speichern")
    print("═"*60 + "\n")
 
    os.makedirs(ausgabe_overlay_ordner, exist_ok=True)
 
    for pfad in bild_pfade:
        dateiname = os.path.basename(pfad)
        basis = os.path.splitext(dateiname)[0]
        img = lade_bild(pfad)
        masken = alle_masken[pfad]
 
        # NEU:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.patch.set_facecolor("#111111")
        fig.suptitle(dateiname, color="white", fontsize=13, y=1.01)
 
        # Panel 1: Originalbild
        ax = axes[0]
        ax.set_facecolor("#111111")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")
        ax.set_title("Original", color="white", fontsize=11)
        ax.axis("off")
 
        # Panel 2: Farbige Masken-Overlay
        ax = axes[1]
        ax.set_facecolor("#111111")
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img if img.ndim == 3 else img
        overlay = label2rgb(masken, image=img_rgb, bg_label=0, alpha=0.4)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(f"Masken-Overlay  ({len(np.unique(masken))-1} Zellen)",
                     color="white", fontsize=11)
        ax.axis("off")
 
        # Panel 3: Nur Masken (bunt, ohne Hintergrund)
        ax = axes[2]
        ax.set_facecolor("#111111")
        masken_bunt = label2rgb(masken, bg_label=0)
        ax.imshow(masken_bunt, interpolation="nearest")
 
        # Umrandungen einzeichnen
        from skimage.segmentation import find_boundaries
        grenzen = find_boundaries(masken, mode="outer")
        ax.imshow(np.ma.masked_where(~grenzen, grenzen),
                  cmap="Reds", alpha=0.8, interpolation="nearest")
        ax.set_title("Masken + Umrandungen", color="white", fontsize=11)
        ax.axis("off")
 
        plt.tight_layout(pad=1.5)
        overlay_pfad = os.path.join(ausgabe_overlay_ordner, f"{basis}_overlay.png")
        # Panel 4: Major- und Minor-Achsen eingezeichnet
        ax = axes[3]
        ax.set_facecolor("#111111")
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", interpolation="nearest")
        else:
            ax.imshow(img, interpolation="nearest")

        from skimage.measure import regionprops
        import matplotlib.patches as mpatches
        props = regionprops(masken)
        for prop in props:
            if prop.area < MIN_PIXEL:
                continue
            cy, cx = prop.centroid
            a = prop.axis_major_length / 2   # halbe Länge für Darstellung
            b = prop.axis_minor_length / 2
            winkel = prop.orientation        # in Radians

            ## Major-Achse (rot)
            # skimage orientation: Winkel von X-Achse, aber Y-Achse zeigt nach unten
            dx_major = a * np.sin(winkel)
            dy_major = a * np.cos(winkel)
            ax.plot([cx - dx_major, cx + dx_major],
                    [cy - dy_major, cy + dy_major],
                    color="#45e950", linewidth=1.2, alpha=0.9)

            # Minor-Achse (blau) – steht senkrecht zur Major-Achse
            dx_minor = b * np.cos(winkel)
            dy_minor = b * np.sin(winkel)
            ax.plot([cx - dx_minor, cx + dx_minor],
                    [cy + dy_minor, cy - dy_minor],
                    color="#00b4d8", linewidth=1.2, alpha=0.9)

        # Legende
        legende = [
            mpatches.Patch(color="#e94560", label="Major Axis (a)"),
            mpatches.Patch(color="#00b4d8", label="Minor Axis (b)"),
        ]
        ax.legend(handles=legende, loc="upper right",
                  facecolor="#111111", labelcolor="white", fontsize=8)
        ax.set_title("Major / Minor Achsen", color="white", fontsize=11)
        ax.axis("off")
        plt.savefig(overlay_pfad, dpi=120, bbox_inches="tight",
                    facecolor="#111111")
        plt.close()
        print(f"  ✓ {basis}_overlay.png")
 
    print(f"\n  Overlay-Bilder gespeichert in:\n    {ausgabe_overlay_ordner}")
 
 
# ──────────────────────────────────────────────
# SCHRITT 3: ELONGATION INDEX BERECHNEN
# ──────────────────────────────────────────────
 
def berechne_elongation(masken, min_pixels=500):
    """
    Berechnet EI = (a - b) / (a + b) für jede segmentierte Zelle.
    a = lange Achse (Major Axis), b = kurze Achse (Minor Axis)
    der an die Zelle gefitteten Ellipse.
 
    Wertebereich: 0 (Kreis) bis 1 (Linie)
    """
    ergebnisse = []
    props = regionprops(masken)
 
    for prop in props:
        if prop.area < min_pixels:
            continue  # Artefakte ignorieren
 
        a = prop.axis_major_length   # lange Halbachse
        b = prop.axis_minor_length   # kurze Halbachse
 
        if (a + b) < 1e-6:
            ei = np.nan
        else:
            ei = (a - b) / (a + b)
 
        ergebnisse.append({
            "Zell_ID":           prop.label,
            "Elongation_Index":  round(ei, 4) if not np.isnan(ei) else np.nan,
            "Flaeche_px":        prop.area,
            "Umfang_px":         round(prop.perimeter, 2),
            "Major_Axis_px":     round(a, 2),
            "Minor_Axis_px":     round(b, 2),
            "Exzentrizitaet":    round(prop.eccentricity, 4),
            "Kompaktheit":       round((4 * np.pi * prop.area) / (prop.perimeter**2), 4)
                                 if prop.perimeter > 0 else np.nan,
            "Zentroid_Y":        round(prop.centroid[0], 1),
            "Zentroid_X":        round(prop.centroid[1], 1),
        })
 
    return ergebnisse
 
 
# ──────────────────────────────────────────────
# SCHRITT 4: PLOTS ERSTELLEN
# ──────────────────────────────────────────────
 
def erstelle_ei_plot(masken, df, ausgabe_pfad, titel=""):
    """EI-Histogramm + EI-Farboverlay pro Bild."""
    ei_werte = df["Elongation_Index"].dropna()
    if len(ei_werte) == 0:
        return
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")
    if titel:
        fig.suptitle(titel, color="white", fontsize=12, y=1.01)
 
    # ── Histogramm ──
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    norm_h = Normalize(vmin=ei_werte.min(), vmax=max(ei_werte.max(), 0.01))
    n, bins, patches = ax1.hist(ei_werte, bins=30, edgecolor="#0f3460", linewidth=0.5)
    for patch, bv in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.plasma(norm_h(bv)))
    ax1.axvline(ei_werte.median(), color="#e94560", lw=2, ls="--",
                label=f"Median: {ei_werte.median():.3f}")
    ax1.axvline(ei_werte.mean(), color="#f5a623", lw=2, ls=":",
                label=f"Mittelwert: {ei_werte.mean():.3f}")
    ax1.set_xlabel("Elongation Index  (a − b) / (a + b)", color="white", fontsize=12)
    ax1.set_ylabel("Anzahl Zellen", color="white", fontsize=12)
    ax1.set_title("EI-Verteilung", color="white", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 1)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#0f3460")
    ax1.legend(facecolor="#16213e", labelcolor="white", fontsize=10)
 
    laenge_mean = df["Major_Axis_px"].mean()
    laenge_std  = df["Major_Axis_px"].std()
    breite_mean = df["Minor_Axis_px"].mean()
    breite_std  = df["Minor_Axis_px"].std()
    flaeche_mean = df["Flaeche_px"].mean()
    flaeche_std  = df["Flaeche_px"].std()
 
    stats = (f"n = {len(df)} Zellen\n"
             f"Median EI: {ei_werte.median():.4f}\n"
             f"Mean EI:   {ei_werte.mean():.4f} ± {ei_werte.std():.4f}\n"
             f"Min/Max EI: {ei_werte.min():.4f} / {ei_werte.max():.4f}\n"
             f"\n"
             f"Länge (a):  {laenge_mean:.1f} ± {laenge_std:.1f} px\n"
             f"Breite (b): {breite_mean:.1f} ± {breite_std:.1f} px\n"
             f"Fläche:     {flaeche_mean:.1f} ± {flaeche_std:.1f} px²")
    ax1.text(0.97, 0.97, stats, transform=ax1.transAxes, color="white",
             fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f3460", alpha=0.85))
 
    # ── EI-Overlay ──
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    ei_dict = dict(zip(df["Zell_ID"], df["Elongation_Index"]))
    ei_max_vis = min(df["Elongation_Index"].quantile(0.95), 1.0)
    cmap = plt.cm.plasma
    rgb = np.zeros((*masken.shape, 3), dtype=np.float32)
    for zell_id, ei in ei_dict.items():
        if np.isnan(ei):
            continue
        px = masken == zell_id
        c = np.clip(ei / (ei_max_vis + 1e-6), 0, 1)
        rgb[px] = cmap(c)[:3]
    ax2.imshow(rgb, interpolation="nearest")
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=ei_max_vis))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Elongation Index", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax2.set_title("EI-Farboverlay  (gelb = verformt, lila = rund)",
                  color="white", fontsize=13, fontweight="bold")
    ax2.axis("off")
 
    plt.tight_layout(pad=2)
    plt.savefig(ausgabe_pfad, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
 
 
def erstelle_gesamt_plot(gesamt_df, ausgabe_pfad):
    """Gesamthistogramm über alle Bilder."""
    ei_vals = gesamt_df["Elongation_Index"].dropna()
    if len(ei_vals) == 0:
        return
 
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
 
    norm = Normalize(vmin=0, vmax=1)
    n, bins, patches = ax.hist(ei_vals, bins=40, edgecolor="#0f3460")
    for patch, bv in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.plasma(norm(bv)))
 
    ax.axvline(ei_vals.median(), color="#e94560", lw=2.5, ls="--",
               label=f"Median: {ei_vals.median():.4f}")
    ax.axvline(ei_vals.mean(), color="#f5a623", lw=2.5, ls=":",
               label=f"Mittelwert: {ei_vals.mean():.4f}")
 
    ax.set_xlabel("Elongation Index  (a − b) / (a + b)", color="white", fontsize=13)
    ax.set_ylabel("Anzahl Zellen", color="white", fontsize=13)
    ax.set_title(f"EI-Gesamtverteilung  |  {len(ei_vals)} Zellen aus "
                 f"{gesamt_df['Quelldatei'].nunique()} Bildern",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#0f3460")
    ax.legend(facecolor="#16213e", labelcolor="white", fontsize=11)
 
    laenge_mean = gesamt_df["Major_Axis_px"].mean()

    laenge_std  = gesamt_df["Major_Axis_px"].std()

    breite_mean = gesamt_df["Minor_Axis_px"].mean()

    breite_std  = gesamt_df["Minor_Axis_px"].std()

    flaeche_mean = gesamt_df["Flaeche_px"].mean()

    flaeche_std  = gesamt_df["Flaeche_px"].std()
 
    stats = (f"n = {len(ei_vals)}\n"

             f"Median EI: {ei_vals.median():.4f}\n"

             f"Mean EI:   {ei_vals.mean():.4f} ± {ei_vals.std():.4f}\n"

             f"Min/Max EI: {ei_vals.min():.4f} / {ei_vals.max():.4f}\n"

             f"\n"

             f"Länge (a):  {laenge_mean:.1f} ± {laenge_std:.1f} px\n"

             f"Breite (b): {breite_mean:.1f} ± {breite_std:.1f} px\n"

             f"Fläche:     {flaeche_mean:.1f} ± {flaeche_std:.1f} px²")
 
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, color="white",
            fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f3460", alpha=0.9))
 
    plt.tight_layout()
    plt.savefig(ausgabe_pfad, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"  ✓ Gesamt-Plot: {ausgabe_pfad}")
 
 
# ──────────────────────────────────────────────
# HAUPTPROGRAMM
# ──────────────────────────────────────────────
 
def main():
    print("\n" + "═"*60)
    print("  Cellpose-SAM → Elongation Index  |  Pipeline")
    print("═"*60)
 
    # Ausgabeordner definieren
    ordner_masken   = os.path.join(OUTPUT_ORDNER, "1_Masken")
    ordner_overlay  = os.path.join(OUTPUT_ORDNER, "2_Overlay_Bilder")
    ordner_ei       = os.path.join(OUTPUT_ORDNER, "3_Elongation_Index")
    for o in [ordner_masken, ordner_overlay, ordner_ei]:
        os.makedirs(o, exist_ok=True)
 
    # Bilder einlesen
    bild_pfade = sorted(
    glob.glob(os.path.join(INPUT_ORDNER, "*.tif")) +
    glob.glob(os.path.join(INPUT_ORDNER, "*.tiff")) +
    glob.glob(os.path.join(INPUT_ORDNER, "*.bmp")) +
    glob.glob(os.path.join(INPUT_ORDNER, "*.png")) +
    glob.glob(os.path.join(INPUT_ORDNER, "*.jpg"))  # ← neu
)
    if not bild_pfade:
        print(f"\n✗ Keine .tif-Dateien gefunden in:\n  {INPUT_ORDNER}")
        return
    print(f"\n  {len(bild_pfade)} TIF-Bild(er) gefunden")
 
    # ── SCHRITT 1: Segmentierung ──────────────────────
    alle_masken = segmentiere_bilder(bild_pfade, ordner_masken)
 
    # ── SCHRITT 2: Overlay-Bilder ─────────────────────
    speichere_overlay_bilder(bild_pfade, alle_masken, ordner_overlay)
 
    # ── SCHRITT 3 + 4: EI berechnen & ausgeben ────────
    print("\n" + "═"*60)
    print("  SCHRITT 3: Elongation Index berechnen")
    print("═"*60 + "\n")
 
    alle_dfs = []
    for pfad in bild_pfade:
        dateiname = os.path.basename(pfad)
        basis = os.path.splitext(dateiname)[0]
        masken = alle_masken[pfad]
 
        ergebnisse = berechne_elongation(masken, min_pixels=MIN_PIXEL)
        if not ergebnisse:
            print(f"  ⚠ {dateiname}: keine Zellen nach Größenfilter")
            continue
 
        df = pd.DataFrame(ergebnisse)
        df["Quelldatei"] = dateiname
        alle_dfs.append(df)
 
        ei = df["Elongation_Index"].dropna()
        print(f"  {dateiname}: {len(df)} Zellen  |  "
              f"EI Median={ei.median():.4f}  Mean={ei.mean():.4f}")
 
        # CSV pro Bild
        csv_pfad = os.path.join(ordner_ei, f"{basis}_EI.csv")
        df.to_csv(csv_pfad, index=False, sep=";", decimal=".")
 
        # Plot pro Bild
        plot_pfad = os.path.join(ordner_ei, f"{basis}_EI_plot.png")
        erstelle_ei_plot(masken, df, plot_pfad, titel=dateiname)
        print(f"    → CSV + Plot gespeichert")
 
    # ── Gesamtauswertung ──────────────────────────────
    if alle_dfs:
        gesamt_df = pd.concat(alle_dfs, ignore_index=True)
        gesamt_csv = os.path.join(ordner_ei, "GESAMT_EI.csv")
        gesamt_df.to_csv(gesamt_csv, index=False, sep=";", decimal=".")
 
        gesamt_plot = os.path.join(ordner_ei, "GESAMT_EI_plot.png")
        erstelle_gesamt_plot(gesamt_df, gesamt_plot)
 
        print("\n" + "═"*60)
        print(f"  GESAMTAUSWERTUNG  ({len(gesamt_df)} Zellen, {len(alle_dfs)} Bilder)")
        print("═"*60)
        ei_all = gesamt_df["Elongation_Index"].dropna()
        print(f"  EI Median:     {ei_all.median():.4f}")
        print(f"  EI Mittelwert: {ei_all.mean():.4f} ± {ei_all.std():.4f}")
        print(f"  EI Min / Max:  {ei_all.min():.4f} / {ei_all.max():.4f}")
        print(f"\n  Ausgabe-Ordner:")
        print(f"    Masken:        {ordner_masken}")
        print(f"    Overlay-Bilder:{ordner_overlay}")
        print(f"    EI-Ergebnisse: {ordner_ei}")
 
    print("\n✓ Pipeline abgeschlossen!\n")
 
 
if __name__ == "__main__":
    main()