from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from collections import defaultdict
import random
import shutil
import csv
import re


PASTA_IMAGENS = Path(r"G:\TCC\Original ROI images")
PASTA_MASCARAS = Path(r"G:\TCC\Mascaras")

DESTINO = Path(r"G:\TCC\Norm_HE_ProtoSeg\datasets\Originais")

SEED = 42
LIMPAR_DESTINO = True

APLICAR_RCAUG_TREINO = True

EXTENSOES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

CLASSES_ORDEM = ["healthy", "mild", "moderate", "severe"]

APELIDOS_CLASSES = {
    "healthy": ["healthy", "saudavel", "saudável", "normal"],
    "mild": ["mild", "leve"],
    "moderate": ["moderate", "moderada", "moderado"],
    "severe": ["severe", "severa", "severo"],
}


def normalizar_nome(nome):
    nome = nome.lower()

    sufixos = [
        "_mask", "-mask", " mask",
        "_masks", "-masks", " masks",
        "_mascara", "-mascara", " mascara",
        "_máscara", "-máscara", " máscara"
    ]

    prefixos = [
        "mask_", "mask-",
        "masks_", "masks-",
        "mascara_", "mascara-",
        "máscara_", "máscara-"
    ]

    for sufixo in sufixos:
        if nome.endswith(sufixo):
            nome = nome[:-len(sufixo)]

    for prefixo in prefixos:
        if nome.startswith(prefixo):
            nome = nome[len(prefixo):]

    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    nome = nome.strip("_")

    return nome


def detectar_classe(caminho):
    texto = " ".join(caminho.parts).lower() + " " + caminho.stem.lower()

    for classe in CLASSES_ORDEM:
        for apelido in APELIDOS_CLASSES[classe]:
            if apelido in texto:
                return classe

    return "all"


def listar_arquivos(pasta):
    arquivos = []
    for arquivo in pasta.rglob("*"):
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES:
            arquivos.append(arquivo)
    return sorted(arquivos)


def criar_indice_mascaras():
    indice = defaultdict(list)

    for mascara in listar_arquivos(PASTA_MASCARAS):
        chave = normalizar_nome(mascara.stem)
        indice[chave].append(mascara)

    return indice


def encontrar_mascara(imagem, indice_mascaras):
    chave = normalizar_nome(imagem.stem)
    candidatos = indice_mascaras.get(chave, [])

    if not candidatos:
        return None

    classe_img = detectar_classe(imagem)

    if classe_img != "all":
        candidatos_mesma_classe = [
            m for m in candidatos
            if detectar_classe(m) == classe_img
        ]

        if candidatos_mesma_classe:
            return candidatos_mesma_classe[0]

    return candidatos[0]


def carregar_pares():
    indice_mascaras = criar_indice_mascaras()
    imagens = listar_arquivos(PASTA_IMAGENS)

    pares = []
    sem_mascara = []

    for imagem in imagens:
        mascara = encontrar_mascara(imagem, indice_mascaras)

        if mascara is None:
            sem_mascara.append(str(imagem))
            continue

        pares.append({
            "imagem": imagem,
            "mascara": mascara,
            "classe": detectar_classe(imagem),
            "nome": normalizar_nome(imagem.stem)
        })

    if sem_mascara:
        print("\n[AVISO] Imagens sem máscara correspondente:")
        for item in sem_mascara[:20]:
            print(item)
        print(f"Total sem máscara: {len(sem_mascara)}")

    return pares


def alocar_por_classe(grupos, total_alvo):
    total = sum(len(v) for v in grupos.values())

    alocacao = {}
    restos = []

    soma = 0

    for classe, itens in grupos.items():
        valor_exato = len(itens) * total_alvo / total
        valor_base = int(valor_exato)
        alocacao[classe] = valor_base
        soma += valor_base
        restos.append((valor_exato - valor_base, classe))

    faltam = total_alvo - soma

    restos.sort(reverse=True)

    for i in range(faltam):
        classe = restos[i][1]
        alocacao[classe] += 1

    return alocacao


def separar_pares(pares):
    random.seed(SEED)

    total = len(pares)

    if total != 456:
        print(f"\n[AVISO] Foram encontrados {total} pares imagem/máscara.")
        print("O artigo usa 456 ROIs. Vou separar por proporção aproximada 70/10/20.")
        pares = pares[:]
        random.shuffle(pares)

        n_test = round(total * 0.20)
        n_val = round(total * 0.10)

        test = pares[:n_test]
        val = pares[n_test:n_test + n_val]
        train = pares[n_test + n_val:]

        return train, val, test

    grupos = defaultdict(list)

    for par in pares:
        grupos[par["classe"]].append(par)

    for classe in grupos:
        random.shuffle(grupos[classe])

    if len(grupos) > 1 and "all" not in grupos:
        n_train_roi = 319
        n_val_roi = 45
        n_test_roi = 92

        val_por_classe = alocar_por_classe(grupos, n_val_roi)
        test_por_classe = alocar_por_classe(grupos, n_test_roi)

        train = []
        val = []
        test = []

        for classe in sorted(grupos.keys()):
            itens = grupos[classe]

            qtd_test = test_por_classe[classe]
            qtd_val = val_por_classe[classe]

            test.extend(itens[:qtd_test])
            val.extend(itens[qtd_test:qtd_test + qtd_val])
            train.extend(itens[qtd_test + qtd_val:])

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return train, val, test

    pares = pares[:]
    random.shuffle(pares)

    test = pares[:92]
    val = pares[92:92 + 45]
    train = pares[92 + 45:]

    return train, val, test


def preparar_pastas():
    if LIMPAR_DESTINO and DESTINO.exists():
        shutil.rmtree(DESTINO)

    for split in ["train", "val", "test"]:
        (DESTINO / split / "mascaras").mkdir(parents=True, exist_ok=True)


def binarizar_mascara(mascara):
    return mascara.point(lambda p: 255 if p > 0 else 0)


def salvar_crops(par, split):
    img = Image.open(par["imagem"]).convert("RGB")
    mask = Image.open(par["mascara"]).convert("L")

    img = img.resize((512, 284), Image.Resampling.BILINEAR)
    mask = mask.resize((512, 284), Image.Resampling.NEAREST)
    mask = binarizar_mascara(mask)

    caixas = [
        (0, 0, 256, 256),
        (256, 0, 512, 256)
    ]

    nomes_gerados = []

    for i, caixa in enumerate(caixas, start=1):
        nome_saida = f"{par['classe']}_{par['nome']}_crop{i}.png"

        img_crop = img.crop(caixa)
        mask_crop = mask.crop(caixa)

        img_crop.save(DESTINO / split / nome_saida)
        mask_crop.save(DESTINO / split / "mascaras" / nome_saida)

        nomes_gerados.append(nome_saida)

    return nomes_gerados


def transformar_par(img, mask, operacoes):
    for op in operacoes:
        if op == "hflip":
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)

        elif op == "vflip":
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)

        elif op == "rot90":
            img = img.rotate(90, expand=False)
            mask = mask.rotate(90, expand=False)

        elif op == "rot180":
            img = img.rotate(180, expand=False)
            mask = mask.rotate(180, expand=False)

        elif op == "rot270":
            img = img.rotate(270, expand=False)
            mask = mask.rotate(270, expand=False)

        elif op == "transpose":
            img = img.transpose(Image.Transpose.TRANSPOSE)
            mask = mask.transpose(Image.Transpose.TRANSPOSE)

    return img, mask


def aplicar_rcaug_treino():
    random.seed(SEED)

    pasta_train = DESTINO / "train"
    pasta_mask = DESTINO / "train" / "mascaras"

    imagens_train = sorted([
        p for p in pasta_train.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ])

    operacoes_possiveis = ["hflip", "vflip", "rot90", "rot180", "rot270", "transpose"]

    total_aug = 0

    for img_path in imagens_train:
        mask_path = pasta_mask / img_path.name

        if not mask_path.exists():
            continue

        img_original = Image.open(img_path).convert("RGB")
        mask_original = Image.open(mask_path).convert("L")

        for rodada in range(1, 4):
            operacoes = []

            for op in operacoes_possiveis:
                if random.random() < 0.5:
                    operacoes.append(op)

            if not operacoes:
                operacoes = [random.choice(operacoes_possiveis)]

            img_aug, mask_aug = transformar_par(img_original, mask_original, operacoes)

            nome_aug = img_path.stem + f"_rcaug{rodada}.png"

            img_aug.save(pasta_train / nome_aug)
            mask_aug.save(pasta_mask / nome_aug)

            total_aug += 1

    print(f"Imagens aumentadas adicionadas ao treino: {total_aug}")


def contar_split(split):
    pasta = DESTINO / split

    imagens = [
        p for p in pasta.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ]

    mascaras = [
        p for p in (pasta / "mascaras").iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    ]

    return len(imagens), len(mascaras)


def salvar_relatorio(train, val, test):
    relatorio = DESTINO / "split_rois.csv"

    with open(relatorio, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "classe", "imagem", "mascara"])

        for split, lista in [("train", train), ("val", val), ("test", test)]:
            for par in lista:
                writer.writerow([
                    split,
                    par["classe"],
                    str(par["imagem"]),
                    str(par["mascara"])
                ])

    print(f"Relatório salvo em: {relatorio}")


def main():
    if not PASTA_IMAGENS.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {PASTA_IMAGENS}")

    if not PASTA_MASCARAS.exists():
        raise FileNotFoundError(f"Pasta de máscaras não encontrada: {PASTA_MASCARAS}")

    pares = carregar_pares()

    print(f"\nPares imagem/máscara encontrados: {len(pares)}")

    por_classe = defaultdict(int)
    for par in pares:
        por_classe[par["classe"]] += 1

    print("\nROIs por classe detectada:")
    for classe, qtd in sorted(por_classe.items()):
        print(f"{classe}: {qtd}")

    train, val, test = separar_pares(pares)

    print("\nSplit em ROIs antes do crop:")
    print("Train ROIs:", len(train))
    print("Val ROIs:", len(val))
    print("Test ROIs:", len(test))

    preparar_pastas()

    for par in train:
        salvar_crops(par, "train")

    for par in val:
        salvar_crops(par, "val")

    for par in test:
        salvar_crops(par, "test")

    print("\nApós crop 256x256:")
    for split in ["train", "val", "test"]:
        qtd_img, qtd_mask = contar_split(split)
        print(f"{split}: {qtd_img} imagens / {qtd_mask} máscaras")

    if APLICAR_RCAUG_TREINO:
        print("\nAplicando aumento de dados no treino...")
        aplicar_rcaug_treino()

        print("\nApós RCAug:")
        for split in ["train", "val", "test"]:
            qtd_img, qtd_mask = contar_split(split)
            print(f"{split}: {qtd_img} imagens / {qtd_mask} máscaras")

    salvar_relatorio(train, val, test)

    print("\nDataset final salvo em:")
    print(DESTINO)


if __name__ == "__main__":
    main()