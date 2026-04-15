#!/usr/bin/env python3
"""
Перебор комбинаций моделей (оптический поток × опционально MDE × опционально SS)
с реальными прогонами run_patch_attack.py. Результаты пишутся в CSV; рейтинг
строится по числам из блока «Final AEE Metrics» (без чтения MLflow).

Какие три (или K) моделей «самые уязвимые» **в терминах этого скрипта**
-----------------------------------------------------------------------
Скрипт **не угадывает** архитектуры: ты задаёшь кандидатов в ``--flow-models``, один
и тот же патч и одну и ту же базовую команду после ``--``. После всех прогонов
уязвимость = **наименьшая** итоговая метрика среди успешных run:

  1) сначала по ``multitask_target_ratio`` (строка «Multi-task target ratio …»),
     если она попала в вывод;
  2) иначе по ``AEE (attacked vs target)``;
  3) при равенстве по (1) сравнивается второе число (aee_target).

**Топ-K самых уязвимых** — это первые K строк итоговой таблицы (см. ``--top-k``).
Чтобы получить именно «три модели», передай хотя бы трёх кандидатов в ``--flow-models``
и смотри первые 3 строки (или ``--top-k 3``).

Критерии (меньше = сильнее атака к цели):
  - multitask_ratio — строка «Multi-task target ratio ...»
  - aee_target — «AEE (attacked vs target)»

Пример (только разные OF-модели, один патч):
  python scripts/select_best_attack_models.py \\
    --flow-models raft,gma,skflow \\
    --results-csv experiment_data/combo_results.csv \\
    -- python run_patch_attack.py --dataset Kitti15 --small_run --trained_patch patch.png

С MDE и сегментацией (декартово произведение):
  python scripts/select_best_attack_models.py \\
    --attack-mde --mde-models depth-anything-v2,marigold \\
    --attack-ss --ss-models pspnet_cityscapes,deeplabv3 \\
    --flow-models raft,gma \\
    --results-csv experiment_data/combo_results.csv \\
    -- python run_patch_attack.py --dataset Kitti15 --small_run --trained_patch patch.png \\
        --attack_mde --attack_ss
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

RE_AEE_TARGET = re.compile(
    r"AEE \(attacked vs target\):\s*(?P<v>[\d.]+|N/A)", re.MULTILINE
)
RE_MTS = re.compile(
    r"lower = stronger attack\):\s*(?P<v>[\d.]+)", re.MULTILINE
)


def parse_metrics_from_output(text: str) -> dict[str, Any]:
    """Достаёт финальные метрики из stdout/stderr прогона."""
    out: dict[str, Any] = {}
    m = RE_AEE_TARGET.search(text)
    if m and m.group("v") != "N/A":
        out["aee_target_attack"] = float(m.group("v"))
    m2 = RE_MTS.search(text)
    if m2:
        out["multitask_target_ratio"] = float(m2.group("v"))
    return out


def load_flow_model_names() -> list[str]:
    try:
        from ptlflow.utils.utils import get_list_of_available_models_list

        return list(get_list_of_available_models_list())
    except Exception as e:
        print("ptlflow недоступен для списка моделей:", e, file=sys.stderr)
        return []


def split_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_combos(
    flow_models: list[str],
    mde_models: list[str] | None,
    ss_models: list[str] | None,
    attack_mde: bool,
    attack_ss: bool,
) -> list[tuple[str, str | None, str | None]]:
    mde_axis: list[str | None]
    ss_axis: list[str | None]
    if attack_mde:
        mde_axis = mde_models or []
        if not mde_axis:
            raise ValueError("Задан --attack-mde, но список --mde-models пуст.")
    else:
        mde_axis = [None]
    if attack_ss:
        ss_axis = ss_models or []
        if not ss_axis:
            raise ValueError("Задан --attack-ss, но список --ss-models пуст.")
    else:
        ss_axis = [None]

    combos: list[tuple[str, str | None, str | None]] = []
    for f, m, s in product(flow_models, mde_axis, ss_axis):
        combos.append((f, m, s))
    return combos


def append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_combo(
    base_cmd: list[str],
    flow: str,
    mde: str | None,
    ss: str | None,
    cwd: Path,
    env: dict[str, str],
) -> tuple[int, str]:
    cmd = list(base_cmd)
    cmd.extend(["--model_name", flow])
    if mde is not None:
        cmd.extend(["--mde_model", mde])
    if ss is not None:
        cmd.extend(["--ss_model", ss])
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode, out


def _fnum(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[float, float]:
        fm = _fnum(r.get("multitask_target_ratio"))
        fa = _fnum(r.get("aee_target_attack"))
        if fm is not None:
            return (fm, fa if fa is not None else float("inf"))
        if fa is not None:
            return (float("inf"), fa)
        return (float("inf"), float("inf"))

    def valid(r: dict[str, Any]) -> bool:
        return int(r.get("returncode", -1)) == 0

    ok = [r for r in rows if valid(r)]
    ok.sort(key=key)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Перебор комбинаций моделей с прогонами run_patch_attack.py и CSV-отчётом."
    )
    p.add_argument(
        "--flow-models",
        default="raft",
        help="Модели оптического потока через запятую или all (все из ptlflow)",
    )
    p.add_argument(
        "--attack-mde",
        action="store_true",
        help="Перебирать --mde_model (нужен --mde-models и в базовой команде обычно --attack_mde)",
    )
    p.add_argument(
        "--mde-models",
        default="depth-anything-v2,marigold",
        help="Список MDE через запятую (при --attack-mde)",
    )
    p.add_argument(
        "--attack-ss",
        action="store_true",
        help="Перебирать --ss_model (нужен --ss-models и в базовой команде обычно --attack_ss)",
    )
    p.add_argument(
        "--ss-models",
        default="pspnet_cityscapes,deeplabv3,segformer_cityscapes,mask2former_cityscapes",
        help="Список SS через запятую (при --attack-ss)",
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=REPO_ROOT / "experiment_data" / "model_combo_sweep.csv",
        help="Куда дописывать строки после каждого прогона",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Только напечатать комбинации и команды, не запускать",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Продолжать после неуспешного return code",
    )
    p.add_argument(
        "--max-combos",
        type=int,
        default=500,
        help="Предохранитель: не стартовать, если комбинаций больше этого числа",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Сколько первых строк рейтинга печатать как «самые уязвимые» (остальные до 20 — кратко)",
    )
    p.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="После -- : команда, обычно: python run_patch_attack.py ...",
    )

    args = p.parse_args()
    rem = list(args.remainder or [])
    if rem and rem[0] == "--":
        rem = rem[1:]
    if not rem:
        sys.exit(
            "Укажите команду после --, например:\n"
            "  -- python run_patch_attack.py --dataset Kitti15 --small_run --trained_patch patch.png"
        )

    if args.flow_models.strip().lower() == "all":
        flow_models = load_flow_model_names()
        if not flow_models:
            sys.exit("Не удалось получить список OF-моделей.")
    else:
        flow_models = split_list(args.flow_models)
    if not flow_models:
        sys.exit("Список --flow-models пуст.")

    mde_list = split_list(args.mde_models) if args.attack_mde else None
    ss_list = split_list(args.ss_models) if args.attack_ss else None

    try:
        combos = build_combos(
            flow_models,
            mde_list,
            ss_list,
            args.attack_mde,
            args.attack_ss,
        )
    except ValueError as e:
        sys.exit(str(e))

    if len(combos) > args.max_combos:
        sys.exit(
            f"Слишком много комбинаций ({len(combos)} > {args.max_combos}). "
            "Сузьте списки или увеличьте --max-combos."
        )

    fieldnames = [
        "flow_model",
        "mde_model",
        "ss_model",
        "returncode",
        "seconds",
        "aee_target_attack",
        "multitask_target_ratio",
        "command",
    ]

    env = os.environ.copy()
    rows_accum: list[dict[str, Any]] = []

    print(f"Всего комбинаций: {len(combos)}")
    for i, (flow, mde, ss) in enumerate(combos, 1):
        label = f"[{i}/{len(combos)}] flow={flow!r}"
        if mde:
            label += f" mde={mde!r}"
        if ss:
            label += f" ss={ss!r}"
        print(label)

        if args.dry_run:
            cmd = list(rem) + ["--model_name", flow]
            if mde:
                cmd.extend(["--mde_model", mde])
            if ss:
                cmd.extend(["--ss_model", ss])
            print("  ", " ".join(cmd))
            continue

        t0 = time.perf_counter()
        code, text = run_combo(rem, flow, mde, ss, REPO_ROOT, env)
        elapsed = time.perf_counter() - t0
        metrics = parse_metrics_from_output(text)

        row = {
            "flow_model": flow,
            "mde_model": mde if mde else "",
            "ss_model": ss if ss else "",
            "returncode": code,
            "seconds": f"{elapsed:.2f}",
            "aee_target_attack": metrics.get("aee_target_attack", ""),
            "multitask_target_ratio": metrics.get("multitask_target_ratio", ""),
            "command": " ".join(
                rem
                + ["--model_name", flow]
                + (["--mde_model", mde] if mde else [])
                + (["--ss_model", ss] if ss else [])
            ),
        }
        append_csv_row(args.results_csv, row, fieldnames)
        rows_accum.append(row)

        if code != 0:
            print(f"  exit {code}", file=sys.stderr)
            if not args.continue_on_error:
                sys.exit(code)
        else:
            print(
                f"  aee_target={row['aee_target_attack']!s} "
                f"mts={row['multitask_target_ratio']!s} ({elapsed:.1f}s)"
            )

    if args.dry_run:
        return

    ranked = rank_rows(rows_accum)
    top_k = max(1, args.top_k)
    print("\n--- Топ уязвимых (меньше multitask / aee_target ⇒ сильнее атака к цели) ---\n")
    for j, r in enumerate(ranked[:top_k], 1):
        print(
            f"{j:2}. flow={r['flow_model']!r} mde={r['mde_model']!r} ss={r['ss_model']!r} | "
            f"mts={r['multitask_target_ratio']!s} aee_tgt={r['aee_target_attack']!s}"
        )
    rest = 20
    if len(ranked) > top_k:
        print(f"\n--- До {rest} места (продолжение) ---\n")
        for j, r in enumerate(ranked[top_k:rest], top_k + 1):
            print(
                f"{j:2}. flow={r['flow_model']!r} mde={r['mde_model']!r} ss={r['ss_model']!r} | "
                f"mts={r['multitask_target_ratio']!s} aee_tgt={r['aee_target_attack']!s}"
            )
    if len(ranked) > rest:
        print(f"... и ещё {len(ranked) - rest} строк (полный лог в {args.results_csv})")
    elif ranked:
        print(f"\nПолные строки: {args.results_csv}")
    else:
        print("Нет успешных прогонов для рейтинга.", file=sys.stderr)


if __name__ == "__main__":
    main()
