"""
======================================================================
 ACTIVIDAD 04 - Segmentación de Imágenes
 PUNTO B: Segmentación de vasos sanguíneos retinianos (DRIVE)
          mediante thresholding.
======================================================================

 ENUNCIADO
   "Desarrolle una aplicación que resuelva un problema de segmentación
    de imágenes médicas usando la idea de thresholding. ¿Qué mecanismo
    de thresholding le ha permitido resolver el problema? Explique los
    resultados."

 DATASET
   DRIVE: Digital Retinal Images for Vessel Extraction.
   - 20 imágenes de fondo de ojo + 20 máscaras manuales anotadas por
     oftalmólogos (ground truth clínico real).
   - Fuente: https://drive.grand-challenge.org/
   - Referencia: Staal et al., "Ridge-based vessel segmentation in
     color images of the retina", IEEE TMI, 2004.

 JUSTIFICACIÓN DEL PROBLEMA
   Los vasos sanguíneos son estructuras oscuras de ancho variable
   (1-10 px) sobre un fondo claro con iluminación no uniforme. La
   variabilidad de calibre y la presencia de vasos capilares finos
   hacen que este problema NO sea trivialmente bimodal, lo que
   permite comparar Otsu (global) vs. adaptativo (local) con
   resultados significativos.

 PIPELINE
   RGB → canal verde → CLAHE → invertir → top-hat → thresholding
       → morfología (POSTPROCESAMIENTO) → métricas con FOV

 EVALUACIÓN
   Se procesan las 20 imágenes de entrenamiento y se reporta:
     - métricas por imagen
     - promedio ± desviación estándar
     - robustez frente al ruido (promediada sobre 3 semillas)

 EJECUCIÓN
   python tarea1_drive.py --carpeta "C:/ruta/a/training"
   python tarea1_drive.py --carpeta "..." --caso 21_training
======================================================================
"""

import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. CARGA
# ----------------------------------------------------------------------
def cargar_caso(carpeta, caso, verbose=True):
    ruta_img = os.path.join(carpeta, "images", f"{caso}.tif")
    ruta_gt = os.path.join(carpeta, "1st_manual",
                            f"{caso.replace('_training', '')}_manual1.gif")
    ruta_fov = os.path.join(carpeta, "mask", f"{caso}_mask.gif")

    if not os.path.exists(ruta_img):
        raise FileNotFoundError(f"No existe: {ruta_img}")
    if not os.path.exists(ruta_gt):
        raise FileNotFoundError(f"No existe: {ruta_gt}")

    img_bgr = cv2.imread(ruta_img, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer: {ruta_img}")

    gt_raw = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)
    if gt_raw is None:
        raise ValueError(f"No se pudo leer GT: {ruta_gt}")
    gt = gt_raw > 0

    fov = None
    if os.path.exists(ruta_fov):
        fov_raw = cv2.imread(ruta_fov, cv2.IMREAD_GRAYSCALE)
        if fov_raw is not None:
            if fov_raw.shape != gt.shape:
                fov_raw = cv2.resize(fov_raw, (gt.shape[1], gt.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            fov = fov_raw > 0

    if verbose:
        print(f"[OK] {caso}: shape={img_bgr.shape[:2]}  "
              f"GT={100*gt.mean():.2f}% vasos")
    return img_bgr, gt, fov


# ----------------------------------------------------------------------
# 2. PREPROCESAMIENTO
# ----------------------------------------------------------------------
def preprocesar(img_bgr, fov=None):
    """
    Canal verde + CLAHE + inversión + top-hat + anulación fuera de FOV.

    Resultado: vasos BRILLANTES sobre fondo oscuro. Por eso los
    métodos de thresholding usan THRESH_BINARY (no INV).
    """
    verde = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    verde = clahe.apply(verde)
    inv = 255 - verde
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.normalize(tophat, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    if fov is not None:
        tophat = tophat.copy()
        tophat[~fov] = 0
    return tophat


# ----------------------------------------------------------------------
# 3. RUIDO
# ----------------------------------------------------------------------
def agregar_ruido(img_u8, sigma, seed):
    rng = np.random.default_rng(seed)
    ruidosa = img_u8.astype(np.float64) + rng.normal(0, sigma, img_u8.shape)
    return np.clip(ruidosa, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 4. MÉTODOS
# ----------------------------------------------------------------------
def segmentar_otsu(img_u8):
    valor, mascara = cv2.threshold(
        img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return valor, mascara


def segmentar_adaptativo(img_u8, block_size=15, C=2):
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )


def segmentar_kmeans(img_u8, k=3, seed=0):
    """K-means (NO es thresholding; comparación). k=3 = fondo/ruido/vaso."""
    cv2.setRNGSeed(seed)
    Z = img_u8.reshape(-1, 1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centros = cv2.kmeans(
        Z, k, None, crit, 8, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(img_u8.shape)
    etiqueta_vaso = int(np.argmax(centros.flatten()))
    return (np.where(labels == etiqueta_vaso, 255, 0).astype(np.uint8),
            centros.flatten())


# ----------------------------------------------------------------------
# 5. POSTPROCESAMIENTO
# ----------------------------------------------------------------------
def limpiar_mascara(mascara, kernel_open=2, kernel_close=2, fov=None):
    limpia = mascara.copy()
    if kernel_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_open, kernel_open))
        limpia = cv2.morphologyEx(limpia, cv2.MORPH_OPEN, k)
    if kernel_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_close, kernel_close))
        limpia = cv2.morphologyEx(limpia, cv2.MORPH_CLOSE, k)
    if fov is not None:
        limpia[~fov] = 0
    return limpia


# ----------------------------------------------------------------------
# 6. MÉTRICAS
# ----------------------------------------------------------------------
def metricas(mascara, ground_truth, fov=None):
    pred = mascara > 0
    gt = ground_truth > 0
    if fov is not None:
        pred = pred & fov
        gt = gt & fov
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union > 0 else 0.0
    tp = inter
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1": f1}


# ----------------------------------------------------------------------
# 7. EVALUACIÓN DE UN CASO
# ----------------------------------------------------------------------
def evaluar_caso(nombre, img_bgr, gt, fov, ko=2, kc=2):
    """Devuelve dict con métricas de Otsu, adaptativo y K-means."""
    img = preprocesar(img_bgr, fov)

    otsu_val, otsu_raw = segmentar_otsu(img)
    otsu_mask = limpiar_mascara(otsu_raw, ko, kc, fov)
    m_otsu = metricas(otsu_mask, gt, fov)

    adap_raw = segmentar_adaptativo(img, 15, 2)
    adap_mask = limpiar_mascara(adap_raw, ko, kc, fov)
    m_adap = metricas(adap_mask, gt, fov)

    km_raw, _ = segmentar_kmeans(img, k=3)
    km_mask = limpiar_mascara(km_raw, ko, kc, fov)
    m_km = metricas(km_mask, gt, fov)

    return {
        "caso": nombre,
        "otsu_val": otsu_val,
        "otsu": m_otsu,
        "adap": m_adap,
        "kmeans": m_km,
        "img_realce": img,
        "mascaras": {"otsu": otsu_mask, "adap": adap_mask, "km": km_mask},
    }


# ----------------------------------------------------------------------
# 8. SECCIÓN A: EVALUAR TODAS LAS IMÁGENES
# ----------------------------------------------------------------------
def seccion_A(carpeta, casos, ko=2, kc=2):
    print("\n" + "=" * 72)
    print(f" SECCIÓN A — Comparación principal sobre {len(casos)} imágenes")
    print("=" * 72)

    filas = []
    for caso in casos:
        try:
            img_bgr, gt, fov = cargar_caso(carpeta, caso, verbose=False)
            r = evaluar_caso(caso, img_bgr, gt, fov, ko, kc)
            filas.append({
                "Caso": caso,
                "Otsu_IoU": r["otsu"]["IoU"],
                "Otsu_F1": r["otsu"]["F1"],
                "Adap_IoU": r["adap"]["IoU"],
                "Adap_F1": r["adap"]["F1"],
                "KM_IoU": r["kmeans"]["IoU"],
                "KM_F1": r["kmeans"]["F1"],
            })
            print(f"  {caso:15s}  "
                  f"Otsu IoU={r['otsu']['IoU']:.3f}  "
                  f"Adap IoU={r['adap']['IoU']:.3f}  "
                  f"KM IoU={r['kmeans']['IoU']:.3f}")
        except Exception as e:
            print(f"  [WARN] {caso}: {e}")

    df = pd.DataFrame(filas)

    # Resumen
    media = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    print("\n  RESUMEN (media ± std sobre", len(df), "imágenes):")
    print(f"    Otsu  IoU = {media['Otsu_IoU']:.3f} ± {std['Otsu_IoU']:.3f}  "
          f"F1 = {media['Otsu_F1']:.3f} ± {std['Otsu_F1']:.3f}")
    print(f"    Adap  IoU = {media['Adap_IoU']:.3f} ± {std['Adap_IoU']:.3f}  "
          f"F1 = {media['Adap_F1']:.3f} ± {std['Adap_F1']:.3f}")
    print(f"    KM    IoU = {media['KM_IoU']:.3f} ± {std['KM_IoU']:.3f}  "
          f"F1 = {media['KM_F1']:.3f} ± {std['KM_F1']:.3f}")

    return df, media, std


# ----------------------------------------------------------------------
# 9. SECCIÓN B: BARRIDO SOBRE 1 IMAGEN (más eficiente)
# ----------------------------------------------------------------------
def seccion_B(carpeta, caso, fov, img_u8, gt, ko=2, kc=2):
    print("\n" + "=" * 72)
    print(f" SECCIÓN B — Barrido de parámetros del adaptativo ({caso})")
    print("=" * 72)

    block_sizes = [7, 11, 15, 25, 35]
    valores_C = [10, 5, 2, 0, -2, -5]

    filas = []
    for bs in block_sizes:
        for C in valores_C:
            raw = segmentar_adaptativo(img_u8, bs, C)
            m = metricas(limpiar_mascara(raw, ko, kc, fov), gt, fov)
            m["block_size"] = bs
            m["C"] = C
            filas.append(m)
    df = pd.DataFrame(filas)[["block_size", "C", "IoU", "Precision",
                              "Recall", "F1"]]
    mejor = df.loc[df["IoU"].idxmax()]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nMejor: block_size={int(mejor.block_size)}, C={int(mejor.C)} "
          f"-> IoU={mejor.IoU:.3f}")
    return df, (int(mejor.block_size), int(mejor.C), float(mejor.IoU))


# ----------------------------------------------------------------------
# 10. SECCIÓN C: ROBUSTEZ SOBRE 1 IMAGEN
# ----------------------------------------------------------------------
def seccion_C(img_base_u8, gt, fov, mejor_bs, mejor_C,
              ko=2, kc=2, reps=3):
    print("\n" + "=" * 72)
    print(" SECCIÓN C — Robustez frente al ruido")
    print(f" (promedio de {reps} semillas por punto)")
    print("=" * 72)

    sigmas = [2, 5, 10, 20]
    filas = []
    for sigma in sigmas:
        acum = {"Otsu": [],
                "Adaptativo fijo (15,2)": [],
                f"Adaptativo afinado ({mejor_bs},{mejor_C})": [],
                "K-means": []}
        for rep in range(reps):
            r = agregar_ruido(img_base_u8, sigma, seed=100 + rep)

            _, o = segmentar_otsu(r)
            acum["Otsu"].append(
                metricas(limpiar_mascara(o, ko, kc, fov), gt, fov)["IoU"])

            a = segmentar_adaptativo(r, 15, 2)
            acum["Adaptativo fijo (15,2)"].append(
                metricas(limpiar_mascara(a, ko, kc, fov), gt, fov)["IoU"])

            at = segmentar_adaptativo(r, mejor_bs, mejor_C)
            acum[f"Adaptativo afinado ({mejor_bs},{mejor_C})"].append(
                metricas(limpiar_mascara(at, ko, kc, fov), gt, fov)["IoU"])

            k, _ = segmentar_kmeans(r, k=3, seed=rep)
            acum["K-means"].append(
                metricas(limpiar_mascara(k, ko, kc, fov), gt, fov)["IoU"])

        for metodo, vals in acum.items():
            filas.append({"sigma": sigma, "Método": metodo,
                           "IoU_promedio": float(np.mean(vals)),
                           "IoU_std": float(np.std(vals))})
    df = pd.DataFrame(filas)[["sigma", "Método", "IoU_promedio", "IoU_std"]]
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


# ----------------------------------------------------------------------
# 11. CONCLUSIÓN
# ----------------------------------------------------------------------
def seccion_D(df_A, media, std, df_C, mejor_bs, mejor_C, mejor_iou):
    print("\n" + "=" * 72)
    print(" SECCIÓN D — Conclusión automática")
    print("=" * 72)

    iou_otsu = media["Otsu_IoU"]
    iou_adap = media["Adap_IoU"]
    iou_km = media["KM_IoU"]

    if iou_otsu > iou_adap:
        ganador = "Otsu (global)"
        compara = (f"Otsu superó al adaptativo fijo por "
                   f"{iou_otsu - iou_adap:.3f} de IoU en promedio.")
    elif iou_otsu < iou_adap:
        ganador = "Adaptativo (local)"
        compara = (f"El adaptativo superó a Otsu por "
                   f"{iou_adap - iou_otsu:.3f} de IoU en promedio.")
    else:
        ganador = "Empate"
        compara = "Empate entre Otsu y el adaptativo."

    if mejor_iou > iou_otsu:
        comp_af = (f"El adaptativo afinado ({mejor_bs}, {mejor_C}) alcanzó "
                   f"IoU={mejor_iou:.3f} en el caso de prueba, superando "
                   f"al Otsu de ese caso.")
    else:
        comp_af = (f"El adaptativo afinado ({mejor_bs}, {mejor_C}) alcanzó "
                   f"IoU={mejor_iou:.3f} en el barrido, sin superar el "
                   f"promedio de Otsu ({iou_otsu:.3f}).")

    fila_s20 = df_C[(df_C["sigma"] == 20) & (df_C["Método"] == "Otsu")]
    iou_otsu_s20 = (float(fila_s20["IoU_promedio"].iloc[0])
                    if len(fila_s20) else float("nan"))

    print(
        "¿Qué mecanismo de thresholding permitió resolver el problema "
        "y por qué?\n\n"
        f"  RESULTADOS PROMEDIO ({len(df_A)} imágenes):\n"
        f"    Otsu (global)        : IoU={iou_otsu:.3f} ± "
        f"{std['Otsu_IoU']:.3f}   F1={media['Otsu_F1']:.3f}\n"
        f"    Adaptativo fijo      : IoU={iou_adap:.3f} ± "
        f"{std['Adap_IoU']:.3f}   F1={media['Adap_F1']:.3f}\n"
        f"    K-means (NO thresh.) : IoU={iou_km:.3f} ± "
        f"{std['KM_IoU']:.3f}   F1={media['KM_F1']:.3f}\n\n"
        f"  COMPARACIÓN:\n"
        f"    {compara}\n"
        f"    {comp_af}\n\n"
        f"  ROBUSTEZ AL RUIDO:\n"
        f"    Otsu mantiene IoU={iou_otsu_s20:.3f} con sigma=20.\n\n"
        f"  RESPUESTA AL ENUNCIADO:\n"
        f"    El mecanismo que mejor resolvió el problema fue: {ganador}.\n"
        f"    Otsu explota la bimodalidad del histograma tras el\n"
        f"    preprocesamiento (verde + CLAHE + top-hat), que separa\n"
        f"    vasos de fondo con un único umbral global, sin necesidad\n"
        f"    de ajustar hiperparámetros.\n\n"
        f"  LIMITACIONES:\n"
        f"    - Thresholding clásico (sin deep learning): el estado del\n"
        f"      arte en DRIVE ronda IoU 0.65-0.75 con U-Net.\n"
        f"    - Vasos capilares finos se pierden con Otsu global.\n"
        f"    - GT anotado con criterio >=70% de certeza del oftalmólogo.\n"
        f"    - La morfología es POSTPROCESAMIENTO, no segmentación.\n"
        f"    - K-means NO es thresholding; es solo comparación."
    )


# ----------------------------------------------------------------------
# FIGURAS
# ----------------------------------------------------------------------
def figura_caso(img_bgr, r, gt, fov, salida):
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    axs[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"1. Retina original ({r['caso']})")
    axs[0, 0].axis("off")
    axs[0, 1].imshow(r["img_realce"], cmap="gray")
    axs[0, 1].set_title("2. Realce (verde+CLAHE+top-hat)")
    axs[0, 1].axis("off")
    axs[0, 2].hist(r["img_realce"].ravel(), bins=256, range=(0, 255),
                    color="steelblue")
    axs[0, 2].axvline(r["otsu_val"], color="red", linestyle="--",
                       label=f"Otsu = {r['otsu_val']:.0f}")
    axs[0, 2].set_title("3. Histograma"); axs[0, 2].legend(fontsize=8)
    axs[1, 0].imshow(r["mascaras"]["otsu"], cmap="gray")
    axs[1, 0].set_title(f"4. Otsu (IoU={r['otsu']['IoU']:.3f})")
    axs[1, 0].axis("off")
    axs[1, 1].imshow(r["mascaras"]["adap"], cmap="gray")
    axs[1, 1].set_title(f"5. Adaptativo (IoU={r['adap']['IoU']:.3f})")
    axs[1, 1].axis("off")
    gt_vis = gt if fov is None else (gt & fov)
    axs[1, 2].imshow(gt_vis, cmap="gray")
    axs[1, 2].set_title("6. GT (oftalmólogo)")
    axs[1, 2].axis("off")
    plt.tight_layout(); plt.savefig(salida, dpi=150); plt.close()
    print(f"[OK] Figura caso: {salida}")


def figura_overlay(img_bgr, r, gt, fov, salida):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ov = rgb.copy(); ov[r["mascaras"]["otsu"] > 0] = [255, 0, 0]
    axs[0].imshow(ov); axs[0].set_title("Otsu (rojo)")
    axs[0].axis("off")
    ov = rgb.copy(); ov[r["mascaras"]["adap"] > 0] = [0, 255, 0]
    axs[1].imshow(ov); axs[1].set_title("Adaptativo (verde)")
    axs[1].axis("off")
    ov = rgb.copy(); m = gt if fov is None else (gt & fov)
    ov[m] = [0, 0, 255]
    axs[2].imshow(ov); axs[2].set_title("GT (azul)")
    axs[2].axis("off")
    plt.tight_layout(); plt.savefig(salida, dpi=150); plt.close()
    print(f"[OK] Figura overlay: {salida}")


def figura_barrido(df_B, salida):
    tabla = df_B.pivot(index="block_size", columns="C", values="IoU")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(tabla.values, cmap="viridis", vmin=0, vmax=1,
                    aspect="auto")
    ax.set_xticks(range(len(tabla.columns))); ax.set_xticklabels(tabla.columns)
    ax.set_yticks(range(len(tabla.index))); ax.set_yticklabels(tabla.index)
    ax.set_xlabel("C"); ax.set_ylabel("block_size")
    ax.set_title("IoU del adaptativo según (block_size, C)")
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            ax.text(j, i, f"{tabla.values[i,j]:.2f}", ha="center",
                     va="center", fontsize=8,
                     color="white" if tabla.values[i,j] < 0.5 else "black")
    plt.colorbar(im, ax=ax, label="IoU")
    plt.tight_layout(); plt.savefig(salida, dpi=150); plt.close()
    print(f"[OK] Figura barrido: {salida}")


def figura_robustez(df_C, salida):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in df_C["Método"].unique():
        s = df_C[df_C["Método"] == m].sort_values("sigma")
        ax.errorbar(s["sigma"], s["IoU_promedio"], yerr=s["IoU_std"],
                    marker="o", capsize=3, label=m)
    ax.set_xlabel("Sigma del ruido"); ax.set_ylabel("IoU promedio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robustez frente al ruido")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(salida, dpi=150); plt.close()
    print(f"[OK] Figura robustez: {salida}")


def figura_distribucion(df_A, salida):
    """Boxplot de IoU por método sobre las 20 imágenes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    datos = [df_A["Otsu_IoU"], df_A["Adap_IoU"], df_A["KM_IoU"]]
    bp = ax.boxplot(
        datos,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="red")
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Otsu", "Adaptativo", "K-means"])
    ax.set_ylabel("IoU por imagen")
    ax.set_title(f"Distribución de IoU sobre {len(df_A)} imágenes")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(salida, dpi=150); plt.close()
    print(f"[OK] Figura distribución: {salida}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Actividad 04 - Segmentación de vasos retinianos DRIVE."
    )
    p.add_argument("--carpeta", required=True)
    p.add_argument("--caso", default="21_training",
                   help="Caso individual para barrido y robustez.")
    p.add_argument("--salida-prefijo", default="resultado_DRIVE")
    args = p.parse_args()

    print("=" * 72)
    print(" ACTIVIDAD 04 - PUNTO B (DRIVE)")
    print(" Segmentación de vasos retinianos con thresholding")
    print("=" * 72)

    casos = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(args.carpeta, "images"))
        if f.endswith(".tif")
    ])
    print(f"[INFO] Encontradas {len(casos)} imágenes en {args.carpeta}")

    # Sección A: todas las imágenes
    df_A, media, std = seccion_A(args.carpeta, casos)

    # Cargar caso individual para B, C, figuras
    img_bgr, gt, fov = cargar_caso(args.carpeta, args.caso, verbose=True)
    img_realce = preprocesar(img_bgr, fov)
    r_caso = evaluar_caso(args.caso, img_bgr, gt, fov)

    # Sección B: barrido
    df_B, (mb, mC, mi) = seccion_B(args.carpeta, args.caso, fov,
                                     img_realce, gt)

    # Sección C: robustez
    df_C = seccion_C(img_realce, gt, fov, mb, mC)

    # Sección D: conclusión
    seccion_D(df_A, media, std, df_C, mb, mC, mi)

    # Figuras
    figura_caso(img_bgr, r_caso, gt, fov,
                 f"{args.salida_prefijo}_caso.png")
    figura_overlay(img_bgr, r_caso, gt, fov,
                    f"{args.salida_prefijo}_overlay.png")
    figura_barrido(df_B, f"{args.salida_prefijo}_barrido.png")
    figura_robustez(df_C, f"{args.salida_prefijo}_robustez.png")
    figura_distribucion(df_A, f"{args.salida_prefijo}_distribucion.png")

    # CSVs
    df_A.to_csv(f"{args.salida_prefijo}_por_imagen.csv", index=False)
    df_B.to_csv(f"{args.salida_prefijo}_barrido.csv", index=False)
    df_C.to_csv(f"{args.salida_prefijo}_robustez.csv", index=False)
    resumen = pd.DataFrame({
        "Métrica": ["Otsu_IoU", "Otsu_F1", "Adap_IoU", "Adap_F1",
                     "KM_IoU", "KM_F1"],
        "Media": [media["Otsu_IoU"], media["Otsu_F1"],
                  media["Adap_IoU"], media["Adap_F1"],
                  media["KM_IoU"], media["KM_F1"]],
        "Std": [std["Otsu_IoU"], std["Otsu_F1"],
                std["Adap_IoU"], std["Adap_F1"],
                std["KM_IoU"], std["KM_F1"]],
    })
    resumen.to_csv(f"{args.salida_prefijo}_resumen.csv", index=False)
    print(f"\n[OK] CSVs: por_imagen, barrido, robustez, resumen")


if __name__ == "__main__":
    main() 