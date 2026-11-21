"""基序库：从 JASPAR 拉取并缓存 PWM，或加载内置的离线基序集。

设计要点
--------
1. **离线优先**：仓库内 `resources/jaspar_core.json` 保存了一小组常用基序，
   保证在无网络环境下 `motifscope demo` 依然可以完整运行。
2. **磁盘缓存**：联网拉取到的基序写入用户缓存目录，重复调用不重复请求。
3. **失败降级**：网络异常时自动回落到内置基序集，并在日志中说明。
"""

from __future__ import annotations

import json
import logging
import os
from importlib import resources
from pathlib import Path

import requests

from .pwm import PWM

log = logging.getLogger(__name__)

JASPAR_MATRIX_URL = "https://jaspar.genereg.net/api/v1/matrix/{matrix_id}/"
JASPAR_LIST_URL = "https://jaspar.genereg.net/api/v1/matrix/"

#: 一组在调控基因组学中常用、信息量较高的脊椎动物基序。
DEFAULT_MATRIX_IDS = (
    "MA0007.1",  # Ar
    "MA0058.1",  # CREB1
    "MA0098.1",  # ETS1
    "MA0139.1",  # CTCF
    "MA0144.1",  # Stat3
    "MA0079.1",  # SP1
    "MA0106.1",  # TP53
    "MA0476.1",  # FOS
)


def default_cache_dir() -> Path:
    """用户级缓存目录，跨项目复用拉取到的基序。"""
    env = os.environ.get("MOTIFSCOPE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "motifscope"
    return Path.home() / ".cache" / "motifscope"


def _bundled_motifs() -> list[dict]:
    """读取随包分发的基序集合。"""
    try:
        ref = resources.files("motifscope.resources").joinpath("jaspar_core.json")
        with ref.open("r", encoding="utf-8") as fh:
            return json.load(fh)["matrices"]
    except (FileNotFoundError, ModuleNotFoundError, KeyError):
        log.warning("内置基序集不可用，返回空列表")
        return []


class MotifDatabase:
    """基序集合的容器，提供按编号/名称检索。"""

    def __init__(self, pwms: list[PWM] | None = None) -> None:
        self._by_id: dict[str, PWM] = {}
        for p in pwms or []:
            self._by_id[p.matrix_id] = p

    # ---------------------------------------------------------------- 构造
    @classmethod
    def load_default(cls, cache_dir: Path | None = None, allow_network: bool = True) -> "MotifDatabase":
        """加载默认基序集：优先磁盘缓存，其次联网，最后内置离线集。"""
        cache_dir = default_cache_dir() if cache_dir is None else Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "jaspar_core.json"

        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                pwms = [PWM.from_dict(d) for d in data["matrices"]]
                if pwms:
                    log.info("从缓存加载 %d 条基序: %s", len(pwms), cache_file)
                    return cls(pwms)
            except Exception as exc:  # noqa: BLE001 - 缓存损坏时继续降级
                log.warning("缓存读取失败(%s)，将重新获取", exc)

        if allow_network:
            try:
                pwms = fetch_jaspar(DEFAULT_MATRIX_IDS, timeout=30)
                if pwms:
                    payload = {"matrices": [p.to_dict() for p in pwms]}
                    cache_file.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
                    log.info("从 JASPAR 拉取并缓存 %d 条基序", len(pwms))
                    return cls(pwms)
            except Exception as exc:  # noqa: BLE001 - 网络失败属预期降级路径
                log.warning("JASPAR 拉取失败(%s)，回落到内置基序集", exc)

        pwms = [PWM.from_dict(d) for d in _bundled_motifs()]
        log.info("使用内置离线基序集，共 %d 条", len(pwms))
        return cls(pwms)

    @classmethod
    def from_json(cls, path: str | Path) -> "MotifDatabase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data["matrices"] if isinstance(data, dict) else data
        return cls([PWM.from_dict(d) for d in items])

    # ---------------------------------------------------------------- 访问
    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):
        return iter(self._by_id.values())

    def __getitem__(self, key: str) -> PWM:
        return self._by_id[key]

    def get(self, key: str) -> PWM | None:
        return self._by_id.get(key)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    def filter_by_length(self, min_len: int = 5, max_len: int = 30) -> "MotifDatabase":
        return MotifDatabase([p for p in self if min_len <= p.length <= max_len])

    def filter_by_information(self, min_bits: float) -> "MotifDatabase":
        return MotifDatabase([p for p in self if p.information_content() >= min_bits])

    def select(self, n: int, seed: int = 0) -> list[PWM]:
        """确定性地挑选 n 条基序（按编号排序后等距抽样）。"""
        ordered = sorted(self._by_id.values(), key=lambda p: p.matrix_id)
        if n >= len(ordered):
            return ordered
        step = len(ordered) / n
        return [ordered[int(i * step)] for i in range(n)]

    def to_json(self, path: str | Path) -> None:
        payload = {"matrices": [p.to_dict() for p in self]}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def fetch_jaspar(
    matrix_ids: tuple[str, ...] | list[str] | None = None,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[PWM]:
    """从 JASPAR REST API 拉取指定编号的基序。

    单条失败不会中断整体流程，只记录告警。
    """
    ids = list(matrix_ids or DEFAULT_MATRIX_IDS)
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "motifscope/0.1 (+https://github.com/moon-star-art)")
    out: list[PWM] = []
    for mid in ids:
        try:
            resp = sess.get(JASPAR_MATRIX_URL.format(matrix_id=mid), timeout=timeout)
            resp.raise_for_status()
            out.append(PWM.from_jaspar_pfm(resp.json()))
        except Exception as exc:  # noqa: BLE001 - 逐条降级
            log.warning("拉取基序 %s 失败: %s", mid, exc)
    return out


def fetch_jaspar_list(
    collection: str = "CORE",
    tax_group: str = "vertebrates",
    page_size: int = 100,
    timeout: float = 30.0,
) -> list[dict]:
    """拉取 JASPAR 基序列表（仅元数据），用于按家族筛选。"""
    sess = requests.Session()
    resp = sess.get(
        JASPAR_LIST_URL,
        params={"collection": collection, "tax_group": tax_group, "page_size": page_size},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])
