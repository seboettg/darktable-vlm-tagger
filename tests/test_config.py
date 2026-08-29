from darktable_vlm_tagger.config import load_config


def test_pair_render_source_defaults_to_raw(tmp_path):
    cfg = load_config(config_dir=tmp_path / "cfg")
    assert cfg.pair_render_source == "raw"


def test_pair_render_source_read_from_config_toml(tmp_path):
    config_dir = tmp_path / "cfg"
    load_config(config_dir=config_dir)  # seed the dir
    toml = config_dir / "config.toml"
    toml.write_text(toml.read_text().replace('render_source = "raw"',
                                              'render_source = "jpeg"'),
                     encoding="utf-8")
    assert load_config(config_dir=config_dir).pair_render_source == "jpeg"
