"""User configuration directory: ~/.darktable-vlm-tagger/

Created lazily on the first CLI invocation, mirroring darktable's own
~/.config/darktable convention. Holds an editable copy of the prompt and
vocabulary plus config.toml for the settings a user is most likely to want
to change without touching the installed package. Precedence is always
CLI flag > config.toml > built-in default.
"""

import tomllib
from dataclasses import dataclass, replace
from importlib import resources
from pathlib import Path

DEFAULT_CONFIG_DIR = Path.home() / ".darktable-vlm-tagger"

CONFIG_TEMPLATE = """\
# darktable-vlm-tagger configuration.
# Any value here can be overridden per run with the matching --flag.

[ollama]
host = "http://localhost:11434"
model = "qwen3-vl:4b-instruct"
num_ctx = 8192
num_predict = 2000
timeout = 600
retries = 1

[image]
# Long edge, in pixels, of the image actually sent to the model. This
# matches the resolution the model and prompt were evaluated at - raising
# it changes runtime and (untested) tagging quality.
target_long_edge = 1024
# darktable mipmap cache level to source from before downscaling (0-8).
mip_level = 4

[darktable]
# Leave commented out to use ~/.config/darktable
# library = "/path/to/alternate/config/dir"

[output]
# print | json | sidecar
mode = "print"
"""


@dataclass(frozen=True)
class Config:
    host: str
    model: str
    num_ctx: int
    num_predict: int
    timeout: int
    retries: int
    target_long_edge: int
    mip_level: int
    library: Path | None
    mode: str
    config_dir: Path

    @property
    def prompt_path(self) -> Path:
        return self.config_dir / "prompt.txt"

    @property
    def vocab_path(self) -> Path:
        return self.config_dir / "vocab.json"


def _package_data(name: str) -> Path:
    return resources.files("darktable_vlm_tagger.data").joinpath(name)


def ensure_config_dir(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_toml = config_dir / "config.toml"
    if not config_toml.exists():
        config_toml.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    for name in ("prompt.txt", "vocab.json"):
        target = config_dir / name
        if not target.exists():
            target.write_bytes(_package_data(name).read_bytes())


def load_config(config_dir: Path = DEFAULT_CONFIG_DIR, **overrides) -> Config:
    """Load config.toml, seeding the config dir first on a first run.

    overrides are CLI-supplied values that take precedence over config.toml;
    a None override means "not set on the CLI" and is ignored.
    """
    config_dir = Path(config_dir).expanduser()
    ensure_config_dir(config_dir)
    data = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
    ollama = data.get("ollama", {})
    image = data.get("image", {})
    darktable = data.get("darktable", {})
    output = data.get("output", {})

    library = darktable.get("library")
    cfg = Config(
        host=ollama.get("host", "http://localhost:11434"),
        model=ollama.get("model", "qwen3-vl:4b-instruct"),
        num_ctx=ollama.get("num_ctx", 8192),
        num_predict=ollama.get("num_predict", 2000),
        timeout=ollama.get("timeout", 600),
        retries=ollama.get("retries", 1),
        target_long_edge=image.get("target_long_edge", 1024),
        mip_level=image.get("mip_level", 4),
        library=Path(library).expanduser() if library else None,
        mode=output.get("mode", "print"),
        config_dir=config_dir,
    )

    clean_overrides = {k: v for k, v in overrides.items() if v is not None}
    if clean_overrides:
        cfg = replace(cfg, **clean_overrides)
    return cfg
