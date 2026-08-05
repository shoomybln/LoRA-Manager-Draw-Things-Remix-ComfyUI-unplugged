import pytest

from py.utils import lora_metadata


def make_sqlite_ckpt(path, names):
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE tensors (name TEXT, data BLOB)")
    for name in names:
        conn.execute("INSERT INTO tensors (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()


class DummySafeOpen:
    def __init__(self, metadata):
        self._metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def metadata(self):
        return self._metadata


@pytest.mark.asyncio
async def test_extract_lora_metadata_returns_base_model(monkeypatch, tmp_path):
    file_path = tmp_path / "model.safetensors"
    file_path.write_bytes(b"")

    monkeypatch.setattr(
        lora_metadata,
        "safe_open",
        lambda *args, **kwargs: DummySafeOpen({"ss_base_model_version": "sdxl"}),
    )

    metadata = await lora_metadata.extract_lora_metadata(str(file_path))

    assert metadata == {"base_model": "SDXL 1.0"}


@pytest.mark.asyncio
async def test_extract_lora_metadata_handles_errors(monkeypatch):
    def raising_safe_open(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(lora_metadata, "safe_open", raising_safe_open)

    metadata = await lora_metadata.extract_lora_metadata("missing.safetensors")

    assert metadata == {"base_model": "Unknown"}


def test_detect_base_model_sqlite_ckpt_sdxl_with_te2(tmp_path):
    path = tmp_path / "sdxl.ckpt"
    make_sqlite_ckpt(
        path,
        [
            "__te2__text_model__[t-0-0]__down__",
            "__text_model__[t-0-0]__down__",
            "__unet__[t-100-0]__down__",
        ],
    )

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "SDXL"


def test_detect_base_model_sqlite_ckpt_sd15_single_text_model(tmp_path):
    path = tmp_path / "sd15.ckpt"
    make_sqlite_ckpt(
        path,
        ["__text_model__[t-11-0]__down__", "__unet__[t-5-0]__down__"],
    )

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "SD1.5"


def test_detect_base_model_sqlite_ckpt_sdxl_unet_only(tmp_path):
    path = tmp_path / "sdxl_unet.ckpt"
    make_sqlite_ckpt(
        path,
        [f"__unet__[t-{i}-0]__down__" for i in range(700, 980)],
    )

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "SDXL"


def test_detect_base_model_sqlite_ckpt_sd15_unet_only(tmp_path):
    path = tmp_path / "sd15_unet.ckpt"
    make_sqlite_ckpt(
        path,
        [f"__unet__[t-{i}-0]__down__" for i in range(0, 400)],
    )

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "SD1.5"


def test_detect_base_model_sqlite_ckpt_dit(tmp_path):
    path = tmp_path / "flux.ckpt"
    make_sqlite_ckpt(path, ["__dit__[t-c_down_proj-0-0]__down__"])

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "Flux.1 D"


def test_detect_base_model_sqlite_ckpt_empty_stub(tmp_path):
    path = tmp_path / "stub.ckpt"
    make_sqlite_ckpt(path, [])

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "Unknown"


def test_detect_base_model_sqlite_ckpt_no_tensors_table(tmp_path):
    import sqlite3

    path = tmp_path / "nostub.ckpt"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE other (name TEXT)")
    conn.commit()
    conn.close()

    assert lora_metadata._detect_base_model_from_sqlite_ckpt(str(path)) == "Unknown"
