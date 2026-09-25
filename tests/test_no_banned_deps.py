"""方法程式不得與 QPMIL（或其舊命名）有任何相依。

指導教授已否決 QPMIL：selector/ data/ scripts/ configs/ sota/ 底下不得再出現該方法的
任何識別字（`sota/` 於 DR-048 加入 —— PI 紅線「sota/ 內亦不得引入 QPMIL 程式碼，
只能讀其切分／manifest 檔」在此之前沒有被任何測試強制），舊的 ZeroNav 命名也一併淘汰（navcil NC-2 起 router 解禁，見 AGENTS.md 命名原則）。

third_party/ 也一起掃：那裡是 CONCH text tower 的原樣複製，必須確認它沒有夾帶
任何舊方法的東西（實測四個 vendored 檔案含 BPE 詞表都零命中）。"conch" 是模型
名稱，不在禁用清單裡 —— 見 ALLOWED。

reference/ 是唯讀存檔（有自己的 SHA256SUMS.txt），outputs/ 是實驗產物，都不掃。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = ("selector", "data", "scripts", "configs", "third_party", "sota")
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".sh", ".toml", ".cfg", ".md"}

# 舊方法（QPMIL）的識別字 + 已淘汰的舊命名。大小寫不敏感。
BANNED = {
    "qpmil": "QPMIL 方法本體",
    "prototype_pool": "QPMIL 的 Prototype Pool",
    "prompt_learner": "QPMIL 的 PromptLearner",
    "tunable_v": "QPMIL 的 Tunable Vector",
    "zeronav": "舊命名，改用 EvidenceSelector / SelectorBank",
    "aggregate_and_predict": "QPMIL backbone 前向；改用 conch_classify",
    "class_text_features": "QPMIL 的 class-feature enhancement；改用 selector.text_encoder",
}
BANNED_RE = re.compile("|".join(re.escape(w) for w in BANNED), re.IGNORECASE)

# 逐檔的窄例外。只放「必須指名舊方法才能把話講清楚」的文件/診斷腳本，且逐字列出
# 允許哪幾個字 —— 方法程式本體（selector/、data/）一個例外都沒有。
# navcil：只保留本 repo 實際存在的檔案（pathselect 的 audit/repro/sota 三條未複製）。
EXEMPT: dict[str, dict[str, str]] = {
    "third_party/conch/PROVENANCE.md": {
        "qpmil": "來源查核紀錄：必須指名才能陳述『QPMIL 沒有改過 CONCH 原始碼』",
    },
    "scripts/v9_reference.py": {
        "qpmil": "唯讀存檔讀取器：說明 v9 側的對照條件",
        "zeronav": "reference/v9 存檔 JSON 的既有 key 名，讀它就得指名",
        "aggregate_and_predict": "陳述 v9 用的是哪個分類器（本檔不呼叫它）",
    },
}

#: scripts/ 的例外只准開給唯讀腳本。這些字樣代表「會寫檔」，出現即視為寫入腳本。
WRITE_MARKERS = ("write_text(", "write_bytes(", "writelines(", ".write(",
                 "torch.save(", "json.dump(", "mkdir(", "makedirs(",
                 '"w"', "'w'", '"a"', "'a'", '"wb"', "'wb'")

# 明確允許：這些字看起來相關但完全合法，不得被誤判為禁用項。
ALLOWED = {
    "conch": "CONCH 是基礎模型名稱（Mahmood Lab），不是被否決的方法",
}
assert not (BANNED.keys() & ALLOWED.keys())

# 匯入 selector 後不該出現在 sys.modules 裡的模組名片段
# navcil NC-2：router 已解禁（分派到 expert 的模組即 selector/router.py），zeronav 維持禁用。
BANNED_MODULE_RE = re.compile(r"qpmil|zeronav", re.IGNORECASE)


def _scanned_files() -> list[Path]:
    files = []
    for d in SCANNED_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in SCANNED_SUFFIXES and "__pycache__" not in p.parts:
                files.append(p)
    return files


def test_scanned_dirs_are_present():
    """該掃的目錄都要在，否則這個測試會空掃而假綠。"""
    for d in ("selector", "data", "configs", "third_party"):
        assert (REPO_ROOT / d).is_dir(), f"{d}/ 不存在，掃描範圍不完整"
    assert _scanned_files(), "no files scanned — 掃描設定壞了"


def test_allowed_words_are_not_banned():
    """"conch" 這類合法字不能被禁用規則掃到。"""
    for word in ALLOWED:
        assert not BANNED_RE.search(word), f"{word!r} 被誤列為禁用字"


@pytest.mark.parametrize("path", _scanned_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_file_has_no_banned_token(path: Path):
    rel = str(path.relative_to(REPO_ROOT))
    exempt = EXEMPT.get(rel, {})
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in BANNED_RE.finditer(line):
            word = m.group(0).lower()
            if word in exempt:
                continue
            hits.append(f"  {rel}:{lineno}: {m.group(0)!r} — {BANNED[word]}"
                        f"\n      {line.strip()}")
    assert not hits, "禁用字殘留：\n" + "\n".join(hits)


def test_method_code_has_no_exemptions():
    """例外只准開在文件與唯讀腳本上；selector/ 與 data/ 必須零例外。"""
    bad = [p for p in EXEMPT if p.startswith(("selector/", "data/", "configs/"))]
    assert not bad, f"方法程式不得有例外：{bad}"
    missing = [p for p in EXEMPT if not (REPO_ROOT / p).exists()]
    assert not missing, f"例外指向不存在的檔案，該清掉：{missing}"
    unknown = {w for words in EXEMPT.values() for w in words} - set(BANNED)
    assert not unknown, f"例外列了不在禁用清單裡的字：{unknown}"


def test_every_exemption_carries_a_reason():
    """拍板 1：每條例外都必須帶理由字串，不准留空或塞佔位符。"""
    for path, words in EXEMPT.items():
        assert words, f"{path} 的例外是空的，該直接刪掉整條"
        for word, reason in words.items():
            assert isinstance(reason, str), f"{path}:{word} 的理由不是字串"
            assert len(reason.strip()) >= 8, f"{path}:{word} 的理由太短：{reason!r}"
            assert reason.strip() not in {"TODO", "FIXME", "-", "N/A"}, \
                f"{path}:{word} 用了佔位符當理由"


def test_only_read_only_scripts_may_be_exempt():
    """拍板 1：scripts/ 與 sota/ 底下只有唯讀模組可申請例外。

    `sota/` 於 DR-048 併入 —— 它同樣是產出結果的 pipeline，同一個理由適用：
    會寫 outputs/ 的腳本一律不得例外。

    任何會寫入 outputs/ 的訓練或評估腳本一律不得例外 —— 產出結果的 pipeline
    必須完全乾淨。這裡用靜態檢查：例外腳本不得出現寫檔字樣，也不得提到 outputs。
    """
    offenders = []
    for path in EXEMPT:
        if not path.startswith(("scripts/", "sota/")):
            continue
        src = (REPO_ROOT / path).read_text(encoding="utf-8")
        hits = [m for m in WRITE_MARKERS if m in src]
        if "outputs" in src:
            hits.append("outputs")
        if hits:
            offenders.append(f"{path}: {hits}")
    assert not offenders, (
        "會寫檔的腳本不得申請禁用字例外，請把唯讀部分拆成獨立模組：\n"
        + "\n".join(offenders))


def test_importing_selector_pulls_no_banned_module():
    """selector 套件 import 後，不得把任何舊方法模組帶進來。"""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import selector.flat_selector, selector.multiround,"
        " selector.classifier, selector.text_encoder;"
        "print('\\n'.join(sys.modules))" % str(REPO_ROOT)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, f"import selector 失敗：\n{proc.stderr}"
    bad = [m for m in proc.stdout.split() if BANNED_MODULE_RE.search(m)]
    assert not bad, f"import 後出現舊方法模組：{bad}"


def test_vendored_conch_matches_upstream():
    """third_party/conch 只准有 PROVENANCE.md 記錄過的那一處改動。

    對照 UPSTREAM_SHA256SUMS.txt（官方 CONCH commit 141cc09c 的逐檔 sha256）。
    custom_tokenizer.py 的 batch_encode_plus → tokenizer() 相容性修改是已知例外。
    """
    import hashlib

    root = REPO_ROOT / "third_party" / "conch"
    known_patched = {"custom_tokenizer.py"}
    expected = {}
    for line in (root / "UPSTREAM_SHA256SUMS.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest

    drifted = []
    for name, digest in expected.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != digest and name not in known_patched:
            drifted.append(f"{name}: {actual} != upstream {digest}")
    assert not drifted, ("vendored CONCH 與官方不符，且未記錄在 PROVENANCE.md：\n"
                         + "\n".join(drifted))
    provenance = (root / "PROVENANCE.md").read_text()
    for name in known_patched:
        assert name in provenance, f"{name} 的改動必須寫進 PROVENANCE.md"
