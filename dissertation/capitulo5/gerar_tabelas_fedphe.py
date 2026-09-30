"""
Gera as 3 tabelas LaTeX pequenas usadas na subseção "Comparação direta com o
FedPHE: um experimento progressivo" de resultados.tex, a partir dos dados já
processados pelos estudos 15pt, 16pt e 26pt (studies/general/).

Não duplica a lógica de cálculo — importa e reutiliza diretamente as funções
de compute_cumulative_bytes/merge_accuracy (15pt) e compute_cumulative_overhead
(16pt) via o mesmo padrão de import por importlib usado nos scripts _pt.py,
e filtra o CSV já gerado pelo estudo 26pt para o resumo de cobertura/inflação.

Também gera as 2 tabelas usadas na subseção "Efeito da participação parcial
sobre custo e acurácia" (tema client_frac): acurácia máxima por fração de
clientes ($f_c$) e inflação de downlink por $f_c$, mesma reutilização de
funções, agora sobre logs/client_frac.

Uso:
    cd dissertation/capitulo5
    python3 gerar_tabelas_fedphe.py

Também gera tables/tab_study27_summary.tex: acurácia final NVNL vs. FedPHE
para os 5 pares modelo/base × 5 porcentagens (não-IID), resumindo as
combinações do Estudo 27 (`27_three_panel.py`) que não entraram como figura
individual em resultados.tex.

Saída: tables/tab_fedphe_bytes.tex, tables/tab_fedphe_overhead.tex,
tables/tab_fedphe_downlink.tex, tables/tab_cf_bytes.tex,
tables/tab_cf_downlink.tex, tables/tab_study27_summary.tex
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
STUDIES_GENERAL = (HERE / ".." / ".." / "studies" / "general").resolve()
LOGS_DIR = (HERE / ".." / ".." / "logs" / "sparsification_percentage").resolve()
CF_LOGS_DIR = (HERE / ".." / ".." / "logs" / "client_frac").resolve()


def _summary_csv(theme):
    """Resumo de cobertura/inflação do Estudo 26 para um tema de log.

    Prefere a cópia local, gravada por `gerar_figuras_cap5.py`, de modo que esta
    pasta seja reprodutível sem rodar nada dentro de studies/. Cai para a saída
    do pipeline em studies/<tema>/tables/ quando a cópia local ainda não existe.
    """
    local = HERE / "data" / "26_union_coverage" / f"{theme}.csv"
    if local.exists():
        return local.resolve()
    return (HERE / ".." / ".." / "studies" / theme / "tables"
            / "26_union_coverage_downlink_summary_pt.csv").resolve()


SUMMARY_CSV = _summary_csv("sparsification_percentage")
CF_SUMMARY_CSV = _summary_csv("client_frac")
OUT_TAB = HERE / "tables"
OUT_TAB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(STUDIES_GENERAL))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, STUDIES_GENERAL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load_data = _load("load_data")
sstyle = _load("solution_style")
s15 = _load("15_bytes_vs_maxacc_pt")
s16 = _load("16_overhead_vs_maxacc_pt")
s27 = _load("27_three_panel")

# Pares (model, dataset) e níveis de niid usados na subseção — mesmos já
# usados no restante do capítulo (Figuras fig:sparsification_cifar10 e
# fig:sparsification_fashionmnist), porcentagem fixa em 30%.
PAIRS = [("resnet20", "CIFAR10", "ResNet-20 / CIFAR-10"),
         ("lenet5", "FASHION_MNIST", "LeNet-5 / FashionMNIST")]
NIIDS = [(1.0, "IID"), (0.1, "Não-IID")]
PCT = 0.3

SOL_ROWS = ["nvnl", "fedphe", "sw", "rw"]


def _fmt_pct(v):
    return f"{v * 100:.2f}".replace(".", ",")


def _fmt2(v):
    return f"{v:.2f}".replace(".", ",")


def _fmt_round(v):
    return f"{int(round(v))}"


# ── Tabela 0: comparativo NVNL-FL vs. métodos de referência ────────────────
# (tab:results_legacy — antes escrita à mão dentro de resultados.tex)

LEGACY_PAIRS = [("lenet5", "FASHION_MNIST", "FashionMNIST (LeNet-5)"),
                ("resnet20", "CIFAR10", "CIFAR-10 (ResNet-20)")]
# (solução, rótulo, porcentagem) — None = baseline de modelo completo
LEGACY_ROWS = [("plaintext", "FedAvg", None), ("bfv", "BFV", None),
               ("ckks", "CKKS", None), ("batchcrypt", "BatchCrypt", None),
               ("fedphe", "FedPHE", 0.9), ("nvnl", "NVNL-FL", 0.9)]
LEGACY_NIIDS = [(1.0, "IID"), (0.1, "Não-IID")]


def _legacy_cell(train, server_eval, model, dataset, solution, pct, niid):
    """Acurácia final, TX bytes (MB) e tempos acumulados de uma célula da tabela.

    TX bytes usa o estimador de tamanho de fio do Estudo 27 (`_est_wire_bytes`),
    e não a coluna `payload_bytes` crua: para as soluções de empacotamento essa
    coluna registra apenas os metadados de máscara (~72 B por pack), não o texto
    cifrado — ver "Payload bytes caveat" em docs/log_schema.md. Foi exatamente
    essa troca que produziu os antigos 0,22 MB e 0,88 MB do FedPHE/NVNL-FL.

    Tempos e bytes são a média entre clientes em cada rodada, somada ao longo das
    rodadas; acurácia é a da última rodada.
    """
    def _sel(df):
        m = (df["model"] == model) & (df["dataset"] == dataset) \
            & (df["solution"] == solution) & (df["niid"] == niid)
        m &= df["percentage"].isna() if pct is None else (df["percentage"] == pct)
        return df[m]

    tr = _sel(train).copy()
    if tr.empty:
        return None
    for col in ["payload_bytes", "n_params", "packs_sent", "poly_modulus_degree",
                "train_time_s", "encrypt_time_s", "decrypt_time_s", "round"]:
        if col in tr.columns:
            tr[col] = pd.to_numeric(tr[col], errors="coerce")
    tr["wire_bytes"] = tr.apply(s27._est_wire_bytes, axis=1)

    per_round = tr.groupby("round")[
        ["wire_bytes", "train_time_s", "encrypt_time_s", "decrypt_time_s"]
    ].mean()
    totals = per_round.sum()

    se = _sel(server_eval)
    acc = float(se.sort_values("round")["mean_acc"].iloc[-1]) if not se.empty else float("nan")

    return {
        "rounds": int(per_round.index.max()),
        "acc": acc,
        "tx_mb": totals["wire_bytes"] / 1e6,
        "tempo": totals[["train_time_s", "encrypt_time_s", "decrypt_time_s"]].sum(),
        "cifra": totals["encrypt_time_s"],
        "decifra": totals["decrypt_time_s"],
    }


def build_results_legacy_table():
    dfs = load_data.load_all(str(LOGS_DIR))
    train, server_eval = dfs["train"], dfs["server_eval"]

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Comparativo de acurácia, volume de dados transmitidos e "
        r"latência computacional entre o NVNL-FL e os métodos de referência para "
        r"os pares LeNet-5/FashionMNIST e ResNet-20/CIFAR-10, avaliados sob "
        r"partições de dados IID e não-IID (FedPHE e NVNL-FL reportados em "
        r"$p=90\%$).}",
        r"\label{tab:results_legacy}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}cccccccccccc@{}}",
        r"\toprule",
        r"\multicolumn{1}{l}{\textbf{}} & \multicolumn{1}{l}{\textbf{}} & "
        + " & ".join(rf"\multicolumn{{5}}{{c}}{{\textbf{{{lbl}}}}}"
                     for _, _, lbl in LEGACY_PAIRS) + r" \\ \midrule",
        r"\textbf{Dados} & \textbf{Soluções} & "
        + " & ".join([r"\textbf{Acurácia} & \textbf{TX Bytes} & \textbf{Tempo} & "
                      r"\textbf{Cifragem} & \textbf{Descifragem}"] * len(LEGACY_PAIRS))
        + r" \\ \midrule",
    ]

    for i, (niid, niid_label) in enumerate(LEGACY_NIIDS):
        for j, (sol, sol_label, pct) in enumerate(LEGACY_ROWS):
            row = []
            for model, dataset, _ in LEGACY_PAIRS:
                cell = _legacy_cell(train, server_eval, model, dataset, sol, pct, niid)
                if cell is None:
                    row += ["--"] * 5
                    continue
                if cell["rounds"] != 200:
                    print(f"  aviso: {model}/{sol}/niid{niid} tem {cell['rounds']} "
                          f"rodadas, não 200", file=sys.stderr)
                row += [f"{cell['acc']:.2f}".replace(".", ","),
                        _fmt2(cell["tx_mb"]), _fmt2(cell["tempo"]),
                        _fmt2(cell["cifra"]), _fmt2(cell["decifra"])]
            head = (rf"\multirow{{{len(LEGACY_ROWS)}}}{{*}}{{\textbf{{{niid_label}}}}}"
                    if j == 0 else "")
            lines.append(" & ".join([head, rf"\textbf{{{sol_label}}}"] + row) + r" \\")
        if i < len(LEGACY_NIIDS) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\midrule",
        r"\multicolumn{12}{l}{*As unidades são acurácia (\%), TX bytes (Mbytes) e "
        r"Tempo (s). O tempo total inclui treino, cifragem e decifragem.} \\",
        r"\multicolumn{12}{l}{*TX bytes é o volume acumulado de \textit{upload} por "
        r"cliente ao longo das $200$ rodadas. Para FedPHE e NVNL-FL corresponde ao "
        r"texto cifrado} \\",
        r"\multicolumn{12}{l}{da biblioteca do Capítulo~\ref{cap:4} "
        r"($\text{pacotes enviados} \times 2N \cdot 8$ bytes); nas demais linhas, "
        r"ao \textit{payload} serializado medido.} \\",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_results_legacy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_results_legacy.tex")


# ── Tabela 0b: síntese por porcentagem de envio ────────────────────────────
# (tab:final_results_consolidation — antes escrita à mão dentro de resultados.tex)

CONSOL_GROUPS = [("resnet20", "CIFAR10", "CIFAR10"),
                 ("lenet5", "FASHION_MNIST", "FashionMNIST"),
                 ("vit_tiny", "tanganke_gtsrb", "GTSRB")]
CONSOL_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9]


def _consol_rows():
    yield ("ckks", None, "CKKS")
    for sol, label in [("nvnl", "NVNL"), ("fedphe", "FedPHE")]:
        for pct in CONSOL_PCTS:
            yield (sol, pct, rf"{label} {int(round(pct * 100))}\%")


def build_final_results_consolidation_table():
    dfs = load_data.load_all(str(LOGS_DIR))
    train, server_eval = dfs["train"], dfs["server_eval"]

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Síntese das métricas absolutas de acurácia, declínio da função "
        r"de perda (\textit{loss}) e tempo total de processamento para CIFAR-10 "
        r"(ResNet-20), FashionMNIST (LeNet-5) e GTSRB (ViT-Tiny), sob as cinco "
        r"porcentagens de envio paramétrico avaliadas.}",
        r"\label{tab:final_results_consolidation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}ll *{10}{c}@{}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Técnica}} & "
        r"\multicolumn{5}{c}{\textbf{IID ($\alpha = 1.0$)}} & "
        r"\multicolumn{5}{c}{\textbf{Não-IID ($\alpha = 0.1$)}} \\",
        r"\cmidrule(lr){3-7} \cmidrule(l){8-12}",
        r"& & " + " & ".join(
            ["Acurácia (\\%) & Loss & Total (s) & Cifra (s) & Decifra (s)"] * 2
        ) + r" \\ \midrule",
    ]

    for gi, (model, dataset, ds_label) in enumerate(CONSOL_GROUPS):
        for ri, (sol, pct, row_label) in enumerate(_consol_rows()):
            row = []
            for niid, _ in LEGACY_NIIDS:
                cell = _legacy_cell(train, server_eval, model, dataset, sol, pct, niid)
                loss = _last_loss(server_eval, model, dataset, sol, pct, niid)
                if cell is None:
                    row += ["--"] * 5
                    continue
                row += [f"{cell['acc']:.2f}".replace(".", ","),
                        _fmt2(loss), _fmt2(cell["tempo"]),
                        _fmt2(cell["cifra"]), _fmt2(cell["decifra"])]
            head = rf"\textbf{{{ds_label}}}" if ri == 0 else ""
            lines.append(" & ".join([head, rf"\textbf{{{row_label}}}"] + row) + r" \\")
        if gi < len(CONSOL_GROUPS) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_final_results_consolidation.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_final_results_consolidation.tex")


def _last_loss(server_eval, model, dataset, solution, pct, niid):
    m = (server_eval["model"] == model) & (server_eval["dataset"] == dataset) \
        & (server_eval["solution"] == solution) & (server_eval["niid"] == niid)
    m &= server_eval["percentage"].isna() if pct is None \
        else (server_eval["percentage"] == pct)
    sub = server_eval[m]
    if sub.empty:
        return float("nan")
    return float(sub.sort_values("round")["mean_loss"].iloc[-1])


# ── Tabela 1: custo em bytes até a acurácia máxima (Estudo 15) ──────────────

def build_bytes_table():
    dfs = load_data.load_all(str(LOGS_DIR))
    cum = s15.compute_cumulative_bytes(dfs["train"])
    data = s15.merge_accuracy(cum, dfs["server_eval"])
    data = data[data["solution"] != "plaintext"]

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Acurácia máxima atingida e custo acumulado em \textit{upload} "
        r"(dado bruto \textit{float32} pré-cifragem, MB) no ponto de pico, para "
        r"$p=30\%$, considerando NVNL-FL, FedPHE, SW e RW.}",
        r"\label{tab:fedphe_bytes}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{Não-IID}} \\",
        r"\textbf{Par} & \textbf{Solução} & \textbf{Acc. máx.} & \textbf{Rodada} & "
        r"\textbf{MB} & \textbf{Acc. máx.} & \textbf{Rodada} & \textbf{MB} \\ \midrule",
    ]
    for i, (model, dataset, label) in enumerate(PAIRS):
        for j, sol in enumerate(SOL_ROWS):
            row = [sstyle.label(sol)]
            for niid, _ in NIIDS:
                sub = data[(data.model == model) & (data.dataset == dataset)
                           & (data.niid == niid) & (data.percentage == PCT)
                           & (data.solution == sol)].sort_values("round")
                if sub.empty:
                    row += ["--", "--", "--"]
                    continue
                macc = s15._max_acc(sub)
                rnd = s15._peak_round(sub)
                cost = s15._cost_at_peak(sub)
                row += [_fmt_pct(macc), _fmt_round(rnd), _fmt2(cost)]
            par_cell = rf"\multirow{{4}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(" & ".join([par_cell] + row) + r" \\")
        if i < len(PAIRS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_fedphe_bytes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_fedphe_bytes.tex")


# ── Tabela 2: custo em tempo até a acurácia máxima (Estudo 16) ──────────────

def build_overhead_table():
    dfs = load_data.load_all(str(LOGS_DIR))
    cum = s16.compute_cumulative_overhead(dfs["train"])
    data = s16.merge_accuracy(cum, dfs["server_eval"])
    data = data[data["solution"] != "plaintext"]

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Acurácia máxima atingida e \textit{overhead} de processamento "
        r"acumulado (empacotamento + seleção + cifragem + serialização + "
        r"decifragem, s) no ponto de pico, para $p=30\%$, considerando NVNL-FL, "
        r"FedPHE, SW e RW.}",
        r"\label{tab:fedphe_overhead}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{Não-IID}} \\",
        r"\textbf{Par} & \textbf{Solução} & \textbf{Acc. máx.} & \textbf{Rodada} & "
        r"\textbf{s} & \textbf{Acc. máx.} & \textbf{Rodada} & \textbf{s} \\ \midrule",
    ]
    for i, (model, dataset, label) in enumerate(PAIRS):
        for j, sol in enumerate(SOL_ROWS):
            row = [sstyle.label(sol)]
            for niid, _ in NIIDS:
                sub = data[(data.model == model) & (data.dataset == dataset)
                           & (data.niid == niid) & (data.percentage == PCT)
                           & (data.solution == sol)].sort_values("round")
                if sub.empty:
                    row += ["--", "--", "--"]
                    continue
                macc = s16._max_acc(sub)
                rnd = s16._peak_round(sub)
                cost = s16._cost_at_peak(sub)
                row += [_fmt_pct(macc), _fmt_round(rnd), _fmt2(cost)]
            par_cell = rf"\multirow{{4}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(" & ".join([par_cell] + row) + r" \\")
        if i < len(PAIRS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_fedphe_overhead.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_fedphe_overhead.tex")


# ── Tabela 3: cobertura da união e inflação de downlink (Estudo 26) ─────────

def build_downlink_table():
    df = pd.read_csv(SUMMARY_CSV)

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Cobertura por cliente (\textit{upload}), cobertura da união "
        r"no servidor (\textit{downlink}) e fator de inflação "
        r"($\text{união}/\text{upload}$), para $p=30\%$, considerando NVNL-FL, "
        r"FedPHE, SW e RW.}",
        r"\label{tab:fedphe_downlink}",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{\textbf{IID}} & \multicolumn{3}{c}{\textbf{Não-IID}} \\",
        r"\textbf{Par} & \textbf{Solução} & \textbf{Upload (\%)} & \textbf{Downlink (\%)} & "
        r"\textbf{Inflação} & \textbf{Upload (\%)} & \textbf{Downlink (\%)} & "
        r"\textbf{Inflação} \\ \midrule",
    ]
    for i, (model, dataset, label) in enumerate(PAIRS):
        for j, sol in enumerate(SOL_ROWS):
            row = [sstyle.label(sol)]
            for niid, _ in NIIDS:
                sub = df[(df.model == model) & (df.dataset == dataset)
                         & (df.niid == niid) & (df.percentage == PCT)
                         & (df.solution == sol)]
                if sub.empty:
                    row += ["--", "--", "--"]
                    continue
                r0 = sub.iloc[0]
                row += [_fmt2(r0.client_cov_pct), _fmt2(r0.union_cov_pct), _fmt2(r0.inflation)]
            par_cell = rf"\multirow{{4}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(" & ".join([par_cell] + row) + r" \\")
        if i < len(PAIRS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_fedphe_downlink.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_fedphe_downlink.tex")


# ── Tabela 4: acurácia máxima por fração de clientes (tema client_frac) ─────

CF_VALUES = [0.25, 0.5, 0.75]
CF_PAIR = ("resnet20", "CIFAR10", "ResNet-20 / CIFAR-10")
CF_NIID = 0.1
CF_PCT_PRIMARY = 0.1


def build_cf_bytes_table():
    dfs = load_data.load_all(str(CF_LOGS_DIR))
    cum = s15.compute_cumulative_bytes(dfs["train"])
    data = s15.merge_accuracy(cum, dfs["server_eval"])
    data = data[data["solution"] != "plaintext"]

    model, dataset, label = CF_PAIR
    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Acurácia máxima atingida por fração de clientes "
        r"participantes ($f_c$), " + label + r", não-IID ($\alpha=0{,}1$), "
        r"$p=10\%$.}",
        r"\label{tab:cf_bytes}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"\textbf{Solução} & \boldmath{$f_c=0,25$} & \boldmath{$f_c=0,5$} & "
        r"\boldmath{$f_c=0,75$} \\ \midrule",
    ]
    for sol in SOL_ROWS:
        row = [sstyle.label(sol)]
        for cf in CF_VALUES:
            sub = data[(data.model == model) & (data.dataset == dataset)
                       & (data.niid == CF_NIID) & (data.percentage == CF_PCT_PRIMARY)
                       & (data.client_fraction == cf) & (data.solution == sol)]
            if sub.empty:
                row.append("--")
                continue
            macc = s15._max_acc(sub)
            row.append(_fmt_pct(macc))
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_cf_bytes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_cf_bytes.tex")


# ── Tabela 5: inflação de downlink por fração de clientes ───────────────────

def build_cf_downlink_table():
    df = pd.read_csv(CF_SUMMARY_CSV)
    model, dataset, label = CF_PAIR

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Fator de inflação de \textit{downlink} "
        r"($\text{união}/\text{upload}$) por fração de clientes participantes "
        r"($f_c$), " + label + r", não-IID ($\alpha=0{,}1$).}",
        r"\label{tab:cf_downlink}",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"\textbf{$p$} & \textbf{Solução} & \boldmath{$f_c=0,25$} & "
        r"\boldmath{$f_c=0,5$} & \boldmath{$f_c=0,75$} \\ \midrule",
    ]
    pcts = [(0.1, "10\\%"), (0.5, "50\\%")]
    for i, (pct, pct_lbl) in enumerate(pcts):
        for j, sol in enumerate(SOL_ROWS):
            row = [sstyle.label(sol)]
            for cf in CF_VALUES:
                sub = df[(df.model == model) & (df.dataset == dataset)
                         & (df.niid == CF_NIID) & (df.percentage == pct)
                         & (df.client_fraction == cf) & (df.solution == sol)]
                if sub.empty:
                    row.append("--")
                    continue
                row.append(_fmt2(sub.iloc[0].inflation) + r"$\times$")
            pct_cell = rf"\multirow{{4}}{{*}}{{{pct_lbl}}}" if j == 0 else ""
            lines.append(" & ".join([pct_cell] + row) + r" \\")
        if i < len(pcts) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_cf_downlink.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_cf_downlink.tex")


# ── Tabela 6: acurácia final NVNL vs. FedPHE, todos os pares × porcentagens ──
# (Estudo 27 — cobre as combinações não escolhidas como figura individual)

STUDY27_PAIRS = [
    ("dnn_tiny", "FASHION_MNIST", "DNN / FashionMNIST"),
    ("lenet5", "FASHION_MNIST", "LeNet-5 / FashionMNIST"),
    ("resnet20", "CIFAR10", "ResNet-20 / CIFAR-10"),
    ("mobilenetv2", "CIFAR10", "MobileNetV2 / CIFAR-10"),
    ("vit_tiny", "tanganke_gtsrb", "ViT / GTSRB"),
]
STUDY27_PCTS = [0.1, 0.3, 0.5, 0.7, 0.9]
STUDY27_NIID = 0.1


def build_study27_summary_table():
    dfs = load_data.load_all(str(LOGS_DIR))
    se = dfs["server_eval"].copy()
    grp = (
        se.groupby(["model", "dataset", "niid", "percentage", "solution", "round"], dropna=False)
        ["mean_acc"].mean().reset_index()
    )
    last = grp.sort_values("round").groupby(
        ["model", "dataset", "niid", "percentage", "solution"], dropna=False
    ).tail(1)

    lines = [
        r"\begin{table}[htpb]",
        r"\centering",
        r"\caption{Acurácia final (última rodada) do NVNL-FL e do FedPHE por "
        r"par modelo/base e porcentagem de envio, não-IID ($\alpha=0{,}1$).}",
        r"\label{tab:study27_summary}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"\textbf{Par} & \textbf{Solução} & \boldmath{$p=10\%$} & \boldmath{$p=30\%$} & "
        r"\boldmath{$p=50\%$} & \boldmath{$p=70\%$} & \boldmath{$p=90\%$} \\ \midrule",
    ]
    for i, (model, dataset, label) in enumerate(STUDY27_PAIRS):
        for j, sol in enumerate(["nvnl", "fedphe"]):
            row = [sstyle.label(sol)]
            for pct in STUDY27_PCTS:
                sub = last[(last.model == model) & (last.dataset == dataset)
                           & (last.niid == STUDY27_NIID) & (last.percentage == pct)
                           & (last.solution == sol)]
                if sub.empty:
                    row.append("--")
                    continue
                row.append(_fmt_pct(float(sub.iloc[0].mean_acc)))
            par_cell = rf"\multirow{{2}}{{*}}{{{label}}}" if j == 0 else ""
            lines.append(" & ".join([par_cell] + row) + r" \\")
        if i < len(STUDY27_PAIRS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (OUT_TAB / "tab_study27_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Salvo tables/tab_study27_summary.tex")


if __name__ == "__main__":
    build_results_legacy_table()
    build_final_results_consolidation_table()
    build_bytes_table()
    build_overhead_table()
    build_downlink_table()
    build_cf_bytes_table()
    build_cf_downlink_table()
    build_study27_summary_table()
