"""
Smoke tests for experiments/check_status.py.

Tests cover:
  1. parse_bash_script  — scalar vars, inline arrays, multi-line arrays,
                          CKKS_CONFIGS with spaces, commented-out array entries
  2. enumerate_expected — packing vs. non-packing solutions, correct key fields
  3. get_max_round      — missing csv, header-only csv, normal csv
  4. classify           — done / partial / missing cases
  5. discover_scripts   — finds run_theme callers, skips lib/template/test
  6. CLI               — basic invocation, --summary-only, --scripts, --no-color

No Docker, no cluster, no GPU required.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_EXP_DIR = _ROOT / "experiments"
sys.path.insert(0, str(_EXP_DIR))

from check_status import (
    classify,
    discover_scripts,
    enumerate_expected,
    get_max_round,
    parse_bash_script,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

def _write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    return path


@pytest.fixture()
def tmp_exp_dir(tmp_path):
    """Temporary experiments directory with a minimal script."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    return exp_dir


@pytest.fixture()
def tmp_logs_dir(tmp_path):
    """Temporary logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    return logs_dir


@pytest.fixture()
def minimal_script(tmp_exp_dir):
    """A minimal experiment script that produces 2 experiments (packing, 1 niid, 1 pct)."""
    p = tmp_exp_dir / "my_experiment.sh"
    _write(p, """\
        #!/bin/bash
        GPU_IDS=(0 1)
        CLIENTS_PER_GPU=2
        FOLDER="my_folder"
        NUM_ROUNDS=10
        NIIDS=(0.5)
        SEEDS=(42)
        TECHNIQUE="none"
        SOLUTIONS=('sweepfl')
        PERCENTAGES=(0.3)
        MODEL_DATASET_PAIRS=(
            "resnet20:CIFAR10"
        )
        CKKS_CONFIGS=("ckks 4096 60")
        KEY_TYPES=('None')
        SCALE_POWER=30
        PLAIN_MODULUS=-1
        N_LENGTH=-1
        source lib.sh
        run_theme
    """)
    return p


@pytest.fixture()
def commented_pairs_script(tmp_exp_dir):
    """Script where MODEL_DATASET_PAIRS has commented-out entries."""
    p = tmp_exp_dir / "commented.sh"
    _write(p, """\
        #!/bin/bash
        GPU_IDS=(5 6 7)
        CLIENTS_PER_GPU=3
        FOLDER="sparsification_percentage"
        NUM_ROUNDS=200
        NIIDS=(0.1 1)
        SEEDS=(42)
        TECHNIQUE="none"
        SOLUTIONS=('sweepfl')
        PERCENTAGES=(0.1 0.3 0.5 0.7 0.9)
        MODEL_DATASET_PAIRS=(
            # "dnn_tiny:FASHION_MNIST"
            "resnet20:CIFAR10"
            # "mobilenetv2:CIFAR10"
        )
        CKKS_CONFIGS=("ckks 8192 60")
        KEY_TYPES=('None')
        SCALE_POWER=40
        PLAIN_MODULUS=-1
        N_LENGTH=-1
        source lib.sh
        run_theme
    """)
    return p


# ── 1. parse_bash_script ───────────────────────────────────────────────────────

class TestParseBashScript:

    def test_scalars(self, minimal_script):
        cfg = parse_bash_script(str(minimal_script))
        assert cfg["NUM_ROUNDS"] == 10
        assert cfg["CLIENTS_PER_GPU"] == 2
        assert cfg["SCALE_POWER"] == 30
        assert cfg["PLAIN_MODULUS"] == -1
        assert cfg["N_LENGTH"] == -1
        assert cfg["FOLDER"] == "my_folder"
        assert cfg["TECHNIQUE"] == "none"

    def test_simple_arrays(self, minimal_script):
        cfg = parse_bash_script(str(minimal_script))
        assert cfg["GPU_IDS"] == ["0", "1"]
        assert cfg["SOLUTIONS"] == ["sweepfl"]
        assert cfg["PERCENTAGES"] == ["0.3"]
        assert cfg["NIIDS"] == ["0.5"]
        assert cfg["SEEDS"] == ["42"]
        assert cfg["KEY_TYPES"] == ["None"]

    def test_ckks_configs_with_spaces(self, minimal_script):
        """CKKS_CONFIGS=("ckks 4096 60") must parse as one token, not three."""
        cfg = parse_bash_script(str(minimal_script))
        assert cfg["CKKS_CONFIGS"] == ["ckks 4096 60"]

    def test_multiline_model_pairs(self, minimal_script):
        cfg = parse_bash_script(str(minimal_script))
        assert cfg["MODEL_DATASET_PAIRS"] == ["resnet20:CIFAR10"]

    def test_commented_entries_skipped(self, commented_pairs_script):
        """Lines starting with # inside MODEL_DATASET_PAIRS must be ignored."""
        cfg = parse_bash_script(str(commented_pairs_script))
        pairs = cfg["MODEL_DATASET_PAIRS"]
        assert "resnet20:CIFAR10" in pairs
        # Commented-out entries must NOT appear
        assert "dnn_tiny:FASHION_MNIST" not in pairs
        assert "mobilenetv2:CIFAR10" not in pairs
        assert len(pairs) == 1

    def test_multiple_niids(self, commented_pairs_script):
        cfg = parse_bash_script(str(commented_pairs_script))
        assert cfg["NIIDS"] == ["0.1", "1"]

    def test_multiple_percentages(self, commented_pairs_script):
        cfg = parse_bash_script(str(commented_pairs_script))
        assert cfg["PERCENTAGES"] == ["0.1", "0.3", "0.5", "0.7", "0.9"]

    def test_ckks_backend_parsed(self, tmp_exp_dir):
        """CKKS_BACKEND scalar is parsed from the script."""
        p = tmp_exp_dir / "backend_test.sh"
        _write(p, """\
            #!/bin/bash
            GPU_IDS=(0)
            CLIENTS_PER_GPU=2
            FOLDER="f"
            NUM_ROUNDS=5
            NIIDS=(0.1)
            SEEDS=(42)
            TECHNIQUE="none"
            SOLUTIONS=('sweepfl')
            PERCENTAGES=(0.1)
            MODEL_DATASET_PAIRS=("resnet20:CIFAR10")
            CKKS_CONFIGS=("ckks 8192 60")
            KEY_TYPES=('None')
            SCALE_POWER=40
            PLAIN_MODULUS=-1
            N_LENGTH=-1
            CKKS_BACKEND="ckks_python_simplify"
            source lib.sh
            run_theme
        """)
        cfg = parse_bash_script(str(p))
        assert cfg["CKKS_BACKEND"] == "ckks_python_simplify"

    def test_ckks_backend_default_tenseal(self, minimal_script):
        """When CKKS_BACKEND is absent, default is tenseal."""
        cfg = parse_bash_script(str(minimal_script))
        assert cfg["CKKS_BACKEND"] == "tenseal"

    def test_ckks_backend_propagated_to_exp(self, tmp_exp_dir):
        """enumerate_expected includes ckks_backend from the script."""
        p = tmp_exp_dir / "be.sh"
        _write(p, """\
            #!/bin/bash
            GPU_IDS=(0)
            CLIENTS_PER_GPU=2
            FOLDER="f"
            NUM_ROUNDS=5
            NIIDS=(0.1)
            SEEDS=(42)
            TECHNIQUE="none"
            SOLUTIONS=('sweepfl')
            PERCENTAGES=(0.1)
            MODEL_DATASET_PAIRS=("resnet20:CIFAR10")
            CKKS_CONFIGS=("ckks 8192 60")
            KEY_TYPES=('None')
            SCALE_POWER=40
            PLAIN_MODULUS=-1
            N_LENGTH=-1
            CKKS_BACKEND="ckks_python_simplify"
            source lib.sh
            run_theme
        """)
        cfg = parse_bash_script(str(p))
        exps = enumerate_expected(cfg)
        assert exps[0]["ckks_backend"] == "ckks_python_simplify"


# ── 2. enumerate_expected ──────────────────────────────────────────────────────

class TestEnumerateExpected:

    def test_packing_count(self, minimal_script):
        """sweepfl (packing) × 1 niid × 1 pct × 1 model × 1 seed = 1 experiment."""
        cfg = parse_bash_script(str(minimal_script))
        exps = enumerate_expected(cfg)
        assert len(exps) == 1

    def test_packing_multi_pct(self, commented_pairs_script):
        """sweepfl × 2 niids × 5 pcts × 1 model × 1 seed = 10 experiments."""
        cfg = parse_bash_script(str(commented_pairs_script))
        exps = enumerate_expected(cfg)
        assert len(exps) == 10

    def test_non_packing_uses_minus_one(self, tmp_exp_dir):
        """plaintext (non-packing) must use percentage=-1 regardless of PERCENTAGES."""
        p = tmp_exp_dir / "plain.sh"
        _write(p, """\
            #!/bin/bash
            GPU_IDS=(0)
            CLIENTS_PER_GPU=2
            FOLDER="test_folder"
            NUM_ROUNDS=5
            NIIDS=(1)
            SEEDS=(42)
            TECHNIQUE="none"
            SOLUTIONS=('plaintext')
            PERCENTAGES=(0.1 0.3 0.5)
            MODEL_DATASET_PAIRS=(
                "resnet20:CIFAR10"
            )
            CKKS_CONFIGS=("ckks 8192 60")
            KEY_TYPES=('None')
            SCALE_POWER=40
            PLAIN_MODULUS=-1
            N_LENGTH=-1
            source lib.sh
            run_theme
        """)
        cfg = parse_bash_script(str(p))
        exps = enumerate_expected(cfg)
        assert len(exps) == 1
        assert exps[0]["pct"] == "-1"

    def test_key_fields(self, minimal_script):
        """Verify that key logfolder fields are correctly computed."""
        cfg = parse_bash_script(str(minimal_script))
        exps = enumerate_expected(cfg)
        e = exps[0]
        assert e["model"] == "resnet20"
        assert e["dataset"] == "CIFAR10"
        assert e["dataset_folder"] == "CIFAR10"
        assert e["solution"] == "sweepfl"
        assert e["technique"] == "none"
        assert e["num_rounds"] == 10
        assert e["nclientes"] == 4          # 2 GPUs × 2 clients each
        assert e["seed"] == "42"
        assert e["niid"] == "0.5"
        assert e["pct"] == "0.3"
        assert e["poly"] == "4096"
        assert e["coeffs"] == "60"
        assert e["scale"] == str(2 ** 30)   # SCALE_POWER=30
        assert e["plain_modulus"] == "-1"
        assert e["n_length"] == "-1"
        assert e["key_type"] == "None"
        assert e["folder"] == "my_folder"

    def test_dataset_slash_replaced(self, tmp_exp_dir):
        """dataset 'tanganke/gtsrb' → dataset_folder 'tanganke_gtsrb'."""
        p = tmp_exp_dir / "gtsrb.sh"
        _write(p, """\
            #!/bin/bash
            GPU_IDS=(0)
            CLIENTS_PER_GPU=2
            FOLDER="vit_folder"
            NUM_ROUNDS=5
            NIIDS=(0.1)
            SEEDS=(42)
            TECHNIQUE="none"
            SOLUTIONS=('sweepfl')
            PERCENTAGES=(0.1)
            MODEL_DATASET_PAIRS=(
                "vit_tiny:tanganke/gtsrb"
            )
            CKKS_CONFIGS=("ckks 8192 60")
            KEY_TYPES=('None')
            SCALE_POWER=40
            PLAIN_MODULUS=-1
            N_LENGTH=-1
            source lib.sh
            run_theme
        """)
        cfg = parse_bash_script(str(p))
        exps = enumerate_expected(cfg)
        assert len(exps) == 1
        assert exps[0]["dataset"] == "tanganke/gtsrb"
        assert exps[0]["dataset_folder"] == "tanganke_gtsrb"


# ── 3. get_max_round ───────────────────────────────────────────────────────────

class TestGetMaxRound:

    def test_missing_file(self, tmp_path):
        assert get_max_round(str(tmp_path / "nonexistent")) == 0

    def test_empty_dir(self, tmp_path):
        log_dir = tmp_path / "exp"
        log_dir.mkdir()
        assert get_max_round(str(log_dir)) == 0

    def test_header_only(self, tmp_path):
        log_dir = tmp_path / "exp"
        log_dir.mkdir()
        (log_dir / "server_eval.csv").write_text(
            "experiment_name,round,n_clients,mean_loss,mean_acc\n"
        )
        assert get_max_round(str(log_dir)) == 0

    def test_normal_csv(self, tmp_path):
        log_dir = tmp_path / "exp"
        log_dir.mkdir()
        (log_dir / "server_eval.csv").write_text(
            "experiment_name,round,n_clients,mean_loss,mean_acc\n"
            "my_exp,1,9,2.3,0.45\n"
            "my_exp,2,9,1.8,0.55\n"
            "my_exp,200,9,0.9,0.82\n"
        )
        assert get_max_round(str(log_dir)) == 200

    def test_partial_csv(self, tmp_path):
        log_dir = tmp_path / "exp"
        log_dir.mkdir()
        (log_dir / "server_eval.csv").write_text(
            "experiment_name,round,n_clients,mean_loss,mean_acc\n"
            "my_exp,1,9,2.3,0.45\n"
            "my_exp,132,9,1.1,0.72\n"
        )
        assert get_max_round(str(log_dir)) == 132


# ── 4. classify ────────────────────────────────────────────────────────────────

class TestClassify:
    """Uses real logs dir structure with synthetic log folders."""

    def _make_exp(self, folder="my_folder", model="resnet20",
                  dataset_folder="CIFAR10", solution="sweepfl",
                  technique="none", num_rounds=10, nclientes=4,
                  seed="42", niid="0.5", pct="0.3", poly="4096",
                  plain_modulus="-1", n_length="-1", coeffs="60",
                  scale=None, key_type="None", ckks_backend="tenseal"):
        scale = scale or str(2 ** 30)
        return {
            "folder": folder,
            "model": model,
            "dataset": "CIFAR10",
            "dataset_folder": dataset_folder,
            "solution": solution,
            "technique": technique,
            "num_rounds": num_rounds,
            "nclientes": nclientes,
            "seed": seed,
            "niid": niid,
            "pct": pct,
            "poly": poly,
            "plain_modulus": plain_modulus,
            "n_length": n_length,
            "coeffs": coeffs,
            "scale": scale,
            "key_type": key_type,
            "ckks_backend": ckks_backend,
        }

    def _make_log_folder(self, logs_dir, exp, timestamp="2026-01-01_00-00-00",
                         max_round=None):
        """Create a fake log folder with optional server_eval.csv."""
        folder = exp["folder"]
        name = (
            f"{exp['model']}_{exp['dataset_folder']}_{exp['solution']}"
            f"_{exp['technique']}_{exp['num_rounds']}rounds"
            f"_{exp['nclientes']}clients_seed{exp['seed']}_niid{exp['niid']}"
            f"_percentage{exp['pct']}_{exp['poly']}"
            f"_{exp['plain_modulus']}_{exp['n_length']}"
            f"_{exp['coeffs']}_{exp['scale']}_{exp['key_type']}_{timestamp}"
        )
        log_dir = Path(logs_dir) / folder / name
        log_dir.mkdir(parents=True, exist_ok=True)

        if max_round is not None:
            rows = ["experiment_name,round,n_clients,mean_loss,mean_acc"]
            for r in range(1, max_round + 1):
                rows.append(f"exp,{r},4,1.0,0.5")
            (log_dir / "server_eval.csv").write_text("\n".join(rows) + "\n")
        return log_dir

    def test_missing(self, tmp_logs_dir):
        exp = self._make_exp()
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "missing"
        assert rnd == 0

    def test_done(self, tmp_logs_dir):
        exp = self._make_exp(num_rounds=10)
        self._make_log_folder(tmp_logs_dir, exp, max_round=10)
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"
        assert rnd == 10

    def test_partial(self, tmp_logs_dir):
        exp = self._make_exp(num_rounds=10)
        self._make_log_folder(tmp_logs_dir, exp, max_round=5)
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "partial"
        assert rnd == 5

    def test_folder_exists_no_csv(self, tmp_logs_dir):
        """Folder exists but server_eval.csv is absent → partial with 0 rounds."""
        exp = self._make_exp(num_rounds=10)
        self._make_log_folder(tmp_logs_dir, exp, max_round=None)
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "partial"
        assert rnd == 0

    def test_client_count_wildcard(self, tmp_logs_dir):
        """A log folder with different nclientes must still match (wildcard)."""
        exp = self._make_exp(num_rounds=10, nclientes=4)
        # Create log folder with 6 clients instead of 4
        different_nclientes_exp = dict(exp, nclientes=6)
        self._make_log_folder(tmp_logs_dir, different_nclientes_exp, max_round=10)
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"

    def test_best_of_duplicates(self, tmp_logs_dir):
        """Multiple log folders for same experiment → pick the one with highest round."""
        exp = self._make_exp(num_rounds=10)
        self._make_log_folder(tmp_logs_dir, exp, "2026-01-01_00-00-01", max_round=3)
        self._make_log_folder(tmp_logs_dir, exp, "2026-01-01_00-00-02", max_round=10)
        self._make_log_folder(tmp_logs_dir, exp, "2026-01-01_00-00-03", max_round=7)
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"
        assert rnd == 10

    # ── Backend verification tests ─────────────────────────────────────────────

    def test_wrong_backend_detected(self, tmp_logs_dir):
        """Log folder with mismatched CKKS_BACKEND → wrong_backend status."""
        exp = self._make_exp(num_rounds=10, ckks_backend="ckks_python_simplify")
        lf = self._make_log_folder(tmp_logs_dir, exp, max_round=10)
        (lf / "experiment_details.json").write_text(
            '{"CKKS_BACKEND": "tenseal"}'
        )
        status, rnd, _, found = classify(exp, str(tmp_logs_dir))
        assert status == "wrong_backend"
        assert found == "tenseal"

    def test_correct_backend_accepted(self, tmp_logs_dir):
        """Matching CKKS_BACKEND in experiment_details.json → done."""
        exp = self._make_exp(num_rounds=10, ckks_backend="ckks_python_simplify")
        lf = self._make_log_folder(tmp_logs_dir, exp, max_round=10)
        (lf / "experiment_details.json").write_text(
            '{"CKKS_BACKEND": "ckks_python_simplify"}'
        )
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"

    def test_no_details_file_accepts_match(self, tmp_logs_dir):
        """No experiment_details.json → backward compat, accept match regardless."""
        exp = self._make_exp(num_rounds=10, ckks_backend="ckks_python_simplify")
        self._make_log_folder(tmp_logs_dir, exp, max_round=10)  # no details file
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"

    def test_wrong_backend_valid_coexist(self, tmp_logs_dir):
        """If one folder has wrong backend and another has correct, pick the correct one."""
        exp = self._make_exp(num_rounds=10, ckks_backend="ckks_python_simplify")
        wrong_lf = self._make_log_folder(tmp_logs_dir, exp, "2026-01-01_00-00-01", max_round=10)
        (wrong_lf / "experiment_details.json").write_text('{"CKKS_BACKEND": "tenseal"}')
        right_lf = self._make_log_folder(tmp_logs_dir, exp, "2026-01-01_00-00-02", max_round=10)
        (right_lf / "experiment_details.json").write_text('{"CKKS_BACKEND": "ckks_python_simplify"}')
        status, rnd, _, _fb = classify(exp, str(tmp_logs_dir))
        assert status == "done"


# ── 5. discover_scripts ────────────────────────────────────────────────────────

class TestDiscoverScripts:

    def _make_script(self, directory: Path, name: str, has_run_theme: bool,
                     defines_run_theme: bool = False) -> Path:
        p = directory / name
        if defines_run_theme:
            body = "run_theme() {\n  echo hi\n}\nrun_theme\n"
        elif has_run_theme:
            body = "source lib.sh\nrun_theme\n"
        else:
            body = "echo hello\n"
        p.write_text(f"#!/bin/bash\n{body}")
        return p

    def test_finds_run_theme_callers(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "my_exp.sh", has_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=False, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "my_exp.sh" in names

    def test_skips_lib_files(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "lib.sh", has_run_theme=True)
        self._make_script(tmp_exp_dir, "lib_tmux.sh", has_run_theme=True, defines_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=False, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "lib.sh" not in names
        assert "lib_tmux.sh" not in names

    def test_skips_template(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "template.sh", has_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=False, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "template.sh" not in names

    def test_skips_test_by_default(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "test_foo.sh", has_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=False, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "test_foo.sh" not in names

    def test_includes_test_with_flag(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "test_foo.sh", has_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=True, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "test_foo.sh" in names

    def test_filter_scripts(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "exp_a.sh", has_run_theme=True)
        self._make_script(tmp_exp_dir, "exp_b.sh", has_run_theme=True)
        found = discover_scripts(str(tmp_exp_dir), include_test=False,
                                 filter_scripts=["exp_a.sh"])
        names = [os.path.basename(f) for f in found]
        assert "exp_a.sh" in names
        assert "exp_b.sh" not in names

    def test_skips_no_run_theme(self, tmp_exp_dir):
        self._make_script(tmp_exp_dir, "utils.sh", has_run_theme=False)
        found = discover_scripts(str(tmp_exp_dir), include_test=False, filter_scripts=[])
        names = [os.path.basename(f) for f in found]
        assert "utils.sh" not in names


# ── 6. CLI smoke tests ─────────────────────────────────────────────────────────

class TestCLI:
    """Run the script as a subprocess against the real repo."""

    SCRIPT = str(_ROOT / "experiments" / "check_status.py")

    def _run(self, *extra_args):
        cmd = [sys.executable, self.SCRIPT] + list(extra_args)
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    def test_exits_zero(self):
        r = self._run()
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_summary_only(self):
        r = self._run("--summary-only")
        assert r.returncode == 0
        assert "SUMMARY" in r.stdout

    def test_no_color(self):
        r = self._run("--no-color")
        assert r.returncode == 0
        # No ANSI escape codes in output
        assert "\033[" not in r.stdout

    def test_single_script_filter(self):
        r = self._run("--scripts", "benchmark_resources.sh", "--all")
        assert r.returncode == 0
        assert "benchmark_resources.sh" in r.stdout
        # Other scripts must not appear
        assert "sparsification_percentage_sweepfl" not in r.stdout

    def test_output_contains_done_partial_missing(self):
        r = self._run("--summary-only")
        assert r.returncode == 0
        assert "Done" in r.stdout
        assert "Partial" in r.stdout or "Missing" in r.stdout

    def test_all_flag_shows_done_rows(self):
        """--all should show ✅ done rows for benchmark_resources (all done)."""
        r = self._run("--scripts", "benchmark_resources.sh", "--all")
        assert r.returncode == 0
        assert "done" in r.stdout

    def test_missing_only_default_hides_done(self):
        """Default mode must not show ✅ done rows for benchmark_resources."""
        r = self._run("--scripts", "benchmark_resources.sh")
        assert r.returncode == 0
        # With no missing/partial, the table should NOT appear — just "All done"
        assert "All" in r.stdout and "done" in r.stdout
        # No table rows with checkmark
        assert "✅ done" not in r.stdout
