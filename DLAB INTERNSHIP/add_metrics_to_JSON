# ATM26 — Script para completar summary JSON con clPrecision y clSensitivity faltantes
# ------------------------------------------------------------------------------
# Ejecucion por terminal:
# python add_mterics_to_JSON.py <ruta_GT> <ruta_pred> <ruta_JSON>
# ------------------------------------------------------------------------------

import json
import argparse
import SimpleITK as sitk
import numpy as np
from skimage.morphology import skeletonize

# ── ARGPARSE ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Calcula clPrecision y clSensitivity y completa un JSON de summary existente"
)
parser.add_argument("ruta_GT", help="Directorio con las labels (ground truth)")
parser.add_argument("ruta_pred", help="Directorio con las predicciones")
parser.add_argument("ruta_JSON", help="Ruta al JSON existente que se quiere completar")

args = parser.parse_args()

GT_FOLDER   = args.ruta_GT
PRED_FOLDER = args.ruta_pred
JSON_PATH   = args.ruta_JSON


# ── FUNCIONES ─────────────────────────────────────────────────────────────────
def img_to_array(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)

def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)

def clDice(pred, label, smooth=1e-5):
    tprec = cl_score(pred, skeletonize(label))
    tsens = cl_score(label, skeletonize(pred))
    cldice = 2 * tprec * tsens / (tprec + tsens + smooth)
    return cldice, tprec, tsens


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    clprec_list = []
    clsens_list = []

    for case in data["metric_per_case"]:
        pred_path  = case["prediction_file"]
        label_path = case["reference_file"]

        print(f"Procesando {pred_path}...")

        pred  = img_to_array(pred_path).astype(np.uint8)
        label = img_to_array(label_path).astype(np.uint8)

        _, clprec, clsens = clDice(pred, label)

        # Actualiza las metricas del caso (asume clave "1", igual que el resto del JSON)
        case["metrics"]["1"]["clPrecision"]   = clprec
        case["metrics"]["1"]["clSensitivity"] = clsens

        clprec_list.append(clprec)
        clsens_list.append(clsens)

        print(f"  clPrecision={clprec:.4f} | clSensitivity={clsens:.4f}")

    # ── Recalcular medias y std solo para las metricas nuevas ─────────────────
    new_means = {
        "clPrecision":   float(np.mean(clprec_list)),
        "clSensitivity": float(np.mean(clsens_list)),
    }
    new_stds = {
        "clPrecision":   float(np.std(clprec_list)),
        "clSensitivity": float(np.std(clsens_list)),
    }

    data["foreground_mean"].update(new_means)
    data["mean"]["1"].update(new_means)
    data["std"]["1"].update(new_stds)

    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nJSON actualizado en: {JSON_PATH}")


if __name__ == "__main__":
    main()
