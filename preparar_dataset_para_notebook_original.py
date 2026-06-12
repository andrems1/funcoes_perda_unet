from pathlib import Path
import shutil
from PIL import Image


ORIGEM = Path(r"G:/TCC/OralEpitheliumDB/original")
DESTINO = Path(r"G:/TCC/Norm_HE_ProtoSeg/datasets/Originais")

CLASSES = ["healthy", "mild", "moderate", "severe"]
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

LIMPAR_DESTINO = True


def listar_arquivos(pasta, extensoes):
    arquivos = []
    for ext in extensoes:
        arquivos.extend(pasta.glob(f"*{ext}"))
        arquivos.extend(pasta.glob(f"*{ext.upper()}"))
    return sorted(arquivos)


def criar_pastas():
    if LIMPAR_DESTINO and DESTINO.exists():
        shutil.rmtree(DESTINO)

    for split in SPLITS:
        (DESTINO / split / "mascaras").mkdir(parents=True, exist_ok=True)


def encontrar_mascara(mask_dir, stem):
    for ext in MASK_EXTENSIONS:
        candidato = mask_dir / f"{stem}{ext}"
        if candidato.exists():
            return candidato

        candidato = mask_dir / f"{stem}{ext.upper()}"
        if candidato.exists():
            return candidato

    return None


def salvar_png_256(img_path, mask_path, dst_img, dst_mask):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = img.resize((256, 256), Image.Resampling.BILINEAR)
    mask = mask.resize((256, 256), Image.Resampling.NEAREST)

    img.save(dst_img)
    mask.save(dst_mask)


def main():
    if not ORIGEM.exists():
        raise FileNotFoundError(f"Pasta de origem não encontrada: {ORIGEM}")

    criar_pastas()

    total_geral = 0

    for split in SPLITS:
        total_split = 0

        for classe in CLASSES:
            img_dir = ORIGEM / split / "images" / classe
            mask_dir = ORIGEM / split / "masks" / classe

            if not img_dir.exists():
                print(f"[WARN] Pasta de imagens não encontrada: {img_dir}")
                continue

            if not mask_dir.exists():
                print(f"[WARN] Pasta de máscaras não encontrada: {mask_dir}")
                continue

            imagens = listar_arquivos(img_dir, IMAGE_EXTENSIONS)

            for img_path in imagens:
                mask_path = encontrar_mascara(mask_dir, img_path.stem)

                if mask_path is None:
                    print(f"[WARN] Máscara não encontrada para: {img_path}")
                    continue

                nome_saida = f"{classe}_{img_path.stem}.png"

                dst_img = DESTINO / split / nome_saida
                dst_mask = DESTINO / split / "mascaras" / nome_saida

                salvar_png_256(img_path, mask_path, dst_img, dst_mask)
                total_split += 1

        total_geral += total_split
        print(f"{split}: {total_split} pares imagem/máscara")

    print(f"\nTotal geral: {total_geral}")
    print(f"Dataset adaptado salvo em: {DESTINO}")


if __name__ == "__main__":
    main()