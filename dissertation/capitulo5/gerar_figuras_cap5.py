"""
Regera as figuras do Capítulo 5 direto de dissertation/capitulo5/, sem escrever
nada dentro de studies/.

O manifesto NÃO é uma lista fixa: é extraído dos `\\includegraphics` de
resultados.tex, que é a fonte da verdade sobre quais figuras entram no capítulo.
Cada entrada é mapeada para (módulo de estudo, tema de log) e, ao final, só os
PDFs do manifesto são publicados nas duas árvores que o FIGURAS.md documenta:

    images/nvnl-slidhe/<rel>   caminho usado pelos \\includegraphics
    figures/<rel>              cópia de nome estável

Os módulos de studies/general são carregados por importlib e têm suas constantes
de saída (OUT_BASE / OUT_FIG_STUDY / OUT_FIG_CURVES / OUT_FIG_BARS / OUT_TAB)
rebindadas em memória para um diretório de staging antes de `main()` ser chamada.
Todas as `main()` leem essas constantes no momento da chamada, então nenhum
arquivo de studies/ é tocado.

O estudo 26 também grava o CSV de resumo que gerar_tabelas_fedphe.py consome;
ele é publicado em data/26_union_coverage/<tema>.csv.

Uso:
    cd dissertation/capitulo5
    python3 gerar_figuras_cap5.py                    # regera as 22 figuras
    python3 gerar_figuras_cap5.py --dry-run          # só mostra o manifesto
    python3 gerar_figuras_cap5.py --only 27_three_panel
    python3 gerar_figuras_cap5.py --keep-staging
"""

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
STUDIES_GENERAL = (HERE / ".." / ".." / "studies" / "general").resolve()
LOGS_ROOT = (HERE / ".." / ".." / "logs").resolve()
TEX = HERE / "resultados.tex"

IMAGES_ROOT = HERE / "images" / "nvnl-slidhe"
FIGURES_ROOT = HERE / "figures"
DATA_ROOT = HERE / "data" / "26_union_coverage"
STAGING = HERE / ".build_figuras"

# Prefixo dos \includegraphics no .tex; o que vem depois é o caminho relativo
# dentro de images/nvnl-slidhe/.
TEX_PREFIX = "sessoes/capitulo5/images/nvnl-slidhe/"
_INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
# Sufixo _cf<frac> emitido por solution_style.cf_suffix() apenas quando cf != 1.0.
_CF_RE = re.compile(r"_cf[0-9.]+")

# Diretório da figura -> (módulo em studies/general, atributos de saída a rebindar).
# O primeiro atributo é o que recebe o subdiretório desta figura.
STUDY_BY_DIR = {
    "27_three_panel":                        ("27_three_panel", "OUT_BASE"),
    "capitulo5_pt/15_bytes_maxacc_pt":       ("15_bytes_vs_maxacc_pt", "OUT_FIG_STUDY"),
    "capitulo5_pt/16_overhead_maxacc_pt":    ("16_overhead_vs_maxacc_pt", "OUT_FIG_STUDY"),
    "capitulo5_pt/20_pack_frequency_pt":     ("20_pack_frequency_pt", "OUT_FIG_STUDY"),
    "capitulo5_pt/26_union_coverage_curves_pt": ("26_union_coverage_downlink_pt", "OUT_FIG_CURVES"),
    "capitulo5_pt/26_downlink_inflation_pt":    ("26_union_coverage_downlink_pt", "OUT_FIG_BARS"),
    "capitulo5_pt/28_download_maxacc_pt":    ("28_download_vs_maxacc_pt", "OUT_FIG_STUDY"),
}

# Todos os atributos de saída de cada módulo, com o subdiretório de staging que
# cada um recebe. "" = a raiz do staging do tema.
MODULE_OUTPUTS = {
    "27_three_panel":               {"OUT_BASE": "27_three_panel"},
    "15_bytes_vs_maxacc_pt":        {"OUT_FIG_STUDY": "capitulo5_pt/15_bytes_maxacc_pt"},
    "16_overhead_vs_maxacc_pt":     {"OUT_FIG_STUDY": "capitulo5_pt/16_overhead_maxacc_pt"},
    "20_pack_frequency_pt":         {"OUT_FIG_STUDY": "capitulo5_pt/20_pack_frequency_pt"},
    "28_download_vs_maxacc_pt":     {"OUT_FIG_STUDY": "capitulo5_pt/28_download_maxacc_pt"},
    "26_union_coverage_downlink_pt": {
        "OUT_FIG_CURVES": "capitulo5_pt/26_union_coverage_curves_pt",
        "OUT_FIG_BARS":   "capitulo5_pt/26_downlink_inflation_pt",
        "OUT_TAB":        "tables",
    },
}

# O estudo 26 grava este CSV; gerar_tabelas_fedphe.py o consome.
SUMMARY_CSV_NAME = "26_union_coverage_downlink_summary_pt.csv"


def _load(name):
    """Carrega um módulo de studies/general sem poluir sys.modules do processo."""
    spec = importlib.util.spec_from_file_location(name, STUDIES_GENERAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_manifest():
    """Figuras referenciadas por resultados.tex, com estudo e tema de log.

    Retorna lista de dicts: rel, subdir, fname, module, theme.
    """
    if not TEX.exists():
        raise FileNotFoundError(f"resultados.tex não encontrado em {TEX}")
    text = TEX.read_text(encoding="utf-8")

    entries, seen = [], set()
    for path in _INCLUDE_RE.findall(text):
        if not path.startswith(TEX_PREFIX):
            continue
        rel = path[len(TEX_PREFIX):]
        if rel in seen:
            continue
        seen.add(rel)

        subdir, fname = rel.rsplit("/", 1)
        if subdir not in STUDY_BY_DIR:
            raise KeyError(
                f"Figura '{rel}' está em '{subdir}', que não tem estudo mapeado "
                f"em STUDY_BY_DIR. Acrescente o mapeamento antes de regerar."
            )
        module, _ = STUDY_BY_DIR[subdir]
        # cf_suffix() só aparece quando client_fraction != 1.0, e o tema
        # client_frac nunca roda com 1.0 — logo o sufixo identifica o tema.
        theme = "client_frac" if _CF_RE.search(fname) else "sparsification_percentage"
        entries.append({"rel": rel, "subdir": subdir, "fname": fname,
                        "module": module, "theme": theme})
    return sorted(entries, key=lambda e: e["rel"])


def run_study(module_name, theme, staging_root):
    """Roda um estudo com as saídas redirecionadas para o staging do tema."""
    logs_dir = LOGS_ROOT / theme
    if not logs_dir.exists():
        raise FileNotFoundError(f"Diretório de logs não encontrado: {logs_dir}")

    mod = _load(module_name)
    outputs = MODULE_OUTPUTS[module_name]
    for attr, subdir in outputs.items():
        if not hasattr(mod, attr):
            raise AttributeError(
                f"{module_name}.py não define '{attr}'. O estudo mudou de layout; "
                f"atualize MODULE_OUTPUTS."
            )
        target = staging_root / theme / subdir if subdir else staging_root / theme
        target.mkdir(parents=True, exist_ok=True)
        setattr(mod, attr, target)

    print(f"\n─── {module_name}  ({theme}) ───", flush=True)
    mod.main(str(logs_dir))


def publish(entries, staging_root):
    """Copia os PDFs do manifesto para images/nvnl-slidhe/ e figures/."""
    results = []
    for e in entries:
        src = staging_root / e["theme"] / e["subdir"] / e["fname"]
        if not src.exists():
            results.append((e["rel"], False))
            continue
        for root in (IMAGES_ROOT, FIGURES_ROOT):
            dst = root / e["subdir"] / e["fname"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        results.append((e["rel"], True))
    return results


def publish_summary_csvs(themes, staging_root):
    """Publica os CSVs do estudo 26 lidos por gerar_tabelas_fedphe.py."""
    published = []
    for theme in sorted(themes):
        src = staging_root / theme / "tables" / SUMMARY_CSV_NAME
        if not src.exists():
            continue
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        dst = DATA_ROOT / f"{theme}.csv"
        shutil.copy2(src, dst)
        published.append(dst)
    return published


def main():
    parser = argparse.ArgumentParser(
        description="Regera as figuras do Capítulo 5 a partir de dissertation/capitulo5.")
    parser.add_argument("--dry-run", action="store_true",
                        help="só imprime o manifesto e os estudos que rodariam")
    parser.add_argument("--only", default=None,
                        help="regera apenas figuras cujo caminho contenha esta substring")
    parser.add_argument("--keep-staging", action="store_true",
                        help="não apaga .build_figuras/ ao final")
    args = parser.parse_args()

    entries = build_manifest()
    if args.only:
        entries = [e for e in entries if args.only in e["rel"]]
        if not entries:
            print(f"Nenhuma figura do manifesto contém '{args.only}'.", file=sys.stderr)
            return 1

    # Pares (módulo, tema) necessários, sem repetir: um estudo roda uma vez por
    # tema e produz de uma vez todas as figuras daquele tema.
    jobs = sorted({(e["module"], e["theme"]) for e in entries})

    print(f"Manifesto: {len(entries)} figura(s) referenciada(s) por resultados.tex")
    for e in entries:
        print(f"  {e['rel']:<72s} {e['module']}  [{e['theme']}]")
    print(f"\nEstudos a rodar: {len(jobs)}")
    for module, theme in jobs:
        print(f"  {module}  [{theme}]")

    if args.dry_run:
        return 0

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    try:
        for module, theme in jobs:
            run_study(module, theme, STAGING)

        results = publish(entries, STAGING)
        csvs = publish_summary_csvs({t for _, t in jobs}, STAGING)
    finally:
        if not args.keep_staging and STAGING.exists():
            shutil.rmtree(STAGING)

    print("\n" + "=" * 78)
    ok = [r for r, done in results if done]
    missing = [r for r, done in results if not done]
    for rel, done in results:
        print(f"  {'regerada' if done else 'AUSENTE '}  {rel}")
    for path in csvs:
        print(f"  CSV       {path.relative_to(HERE)}")
    print("=" * 78)
    print(f"{len(ok)} regerada(s), {len(missing)} ausente(s) — "
          f"publicadas em images/nvnl-slidhe/ e figures/")

    if missing:
        print("\nFiguras do manifesto que o estudo não produziu:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        print("Confira se o nome do arquivo no .tex ainda casa com o gerado pelo "
              "estudo (regras _tok/_tok_pct).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
