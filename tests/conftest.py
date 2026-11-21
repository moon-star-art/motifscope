"""共享测试夹具。

注意：本仓库在受限沙箱下开发时，pytest 的 ``tmp_path`` 夹具在清理阶段
（``shutil.rmtree``）会因目录权限而失败。因此这里自定义 ``tmp_dir`` 夹具，
把临时目录建在项目内的 ``_build/testtmp`` 下，且不做删除清理。

在普通环境下，``tmp_path`` 完全可用；本夹具只是为了让测试在沙箱内也能跑通，
不改变测试语义。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parents[1] / "_build" / "testtmp"


@pytest.fixture
def tmp_dir() -> Path:
    """提供一个唯一命名的临时目录（不自动删除）。"""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    d = _TMP_ROOT / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def tmp_path_factory():
    """覆盖 pytest 内置的会话级临时目录工厂。

    内置实现会在会话结束时用 ``shutil.rmtree`` 清理根目录，在受限沙箱下
    会因权限失败并让所有用到它的测试报错。这里提供一个不做清理的替代实现，
    接口保持与内置版本一致的 ``.mktemp(name)``。
    """

    class _Factory:
        _root = Path(__file__).resolve().parents[1] / "_build" / "testtmp_session"

        def mktemp(self, basename: str | None = None, numbered: bool = True) -> Path:
            self._root.mkdir(parents=True, exist_ok=True)
            name = basename or f"tmp{uuid.uuid4().hex[:8]}"
            d = self._root / name
            d.mkdir(parents=True, exist_ok=True)
            return d

    return _Factory()
