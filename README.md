# darktable-vlm-tagger

Tag, title and describe your darktable photographs with a local
vision-language model — a drop-in replacement for auto-keywording 
(STAG), whose tags (`st|building`, `st|sky`, ...) are too generic 
to be useful for archive search.

Everything runs locally through [Ollama](https://ollama.com/): no cloud
service, no per-image cost, no data leaving your machine.

## How it works

For each photograph the model returns a JSON object built from a
closed-vocabulary schema, covering:

- `category` (exactly one, e.g. `landscape`, `portrait`, `street photography`)
- `color`, `tone`, `light`, `composition`, `technique`, `mood` — closed
  vocabularies, each free to stay empty when nothing applies
- `subject` — free vocabulary, what is actually visible in the frame
- `title` — a short, specific caption
- `description` — one factual sentence

Tags are written as hierarchical darktable tags (`category|landscape`,
`subject|tram shelter`, ...) directly into the image's XMP sidecar, next to
`title` and `description`. The full vocabulary and the reasoning behind the
schema live in `src/darktable_vlm_tagger/data/vocab.json` and `prompt.txt`.

To keep a huge image batch case fast, images are sourced from darktable's
own mipmap thumbnail cache wherever possible instead of being re-rendered,
falling back to `darktable-generate-cache` or `darktable-cli` only when
needed. See `image_source.py` for the exact priority.

## Installation & setup

Five things, in order: darktable itself, Python, Ollama plus a model,
ImageMagick, then this tool. Commands are given for Linux and macOS; adjust
the package manager line for your distro if you're not on Arch/Debian.

### 1. darktable

Install it from [darktable.org](https://www.darktable.org/install/) or your
distro's package manager (`pacman -S darktable`, `brew install darktable`,
...). Official builds ship with Lua scripting enabled by default, which the
[in-app integration](#darktable-in-app-integration-lua) below needs —
nothing extra to configure for that part.

This tool also shells out to darktable's command-line helpers
(`darktable-cli`, `darktable-generate-cache`) to render RAW/edited images
the mipmap cache doesn't already cover. On Linux those are on your `$PATH`.
On **macOS** the app bundle keeps them to itself — the tool looks inside
`/Applications/darktable.app/Contents/MacOS` (and the same path under
`~/Applications`) automatically. If you installed darktable somewhere else,
point `DARKTABLE_BIN_DIR` at the directory holding those binaries, e.g.
`export DARKTABLE_BIN_DIR=/Applications/darktable.app/Contents/MacOS`.

### 2. Python 3.11+

- **Linux**: usually already installed (`python3 --version`). If not:
  `sudo pacman -S python` / `sudo apt install python3` / your distro's
  equivalent.
- **macOS**: `brew install python@3.12`, or the installer from
  [python.org](https://www.python.org/downloads/macos/).

### 3. Ollama, and a vision model

- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **macOS**: `brew install ollama` (CLI only), or the app from
  [ollama.com/download](https://ollama.com/download) (adds a menu-bar app
  alongside the CLI). Either way, make sure the Ollama service is running
  before using this tool (`ollama serve`, or just open the app on macOS).

Then pull the model:

```bash
ollama pull qwen3-vl:4b-instruct
```

Use the `-instruct` tag specifically. The bare tag (`qwen3-vl:4b`) resolves
to the *thinking* variant, whose reasoning trace adds no value here but
costs a 3-4x slowdown.

If your GPU has less than ~12 GB VRAM and you ever run more than one model,
stop Ollama from silently offloading layers to the CPU by setting
`OLLAMA_MAX_LOADED_MODELS=1`:

- **Linux (systemd)**: `sudo systemctl edit ollama`, then add:

  ```ini
  [Service]
  Environment="OLLAMA_MAX_LOADED_MODELS=1"
  ```

- **macOS**: add `export OLLAMA_MAX_LOADED_MODELS=1` to your shell's rc
  file (`~/.zshrc` on a default install), then restart Ollama.

### 4. ImageMagick

Used to downscale images to the resolution the model was evaluated at
before sending them to Ollama.

- **Linux**: `sudo pacman -S imagemagick` / `sudo apt install imagemagick`
- **macOS**: `brew install imagemagick`

### 5. darktable-vlm-tagger itself

Installed with [pipx](https://pipx.pypa.io/), which keeps it in its own
isolated environment rather than touching your system Python.

```bash
# if you don't have pipx yet:
#   Linux:  sudo pacman -S python-pipx   (or: python3 -m pip install --user pipx)
#   macOS:  brew install pipx
pipx ensurepath   # once, then open a new shell

git clone https://github.com/seboettg/darktable-vlm-tagger.git
cd darktable-vlm-tagger
pipx install -e .
```

This installs the `dt-vlm-tag` command onto your `PATH`, normally at
`~/.local/bin/dt-vlm-tag` — the same location on both Linux and macOS. If
you're not sure where it landed, `pipx list` or (after `pipx ensurepath`
and a fresh shell) `which dt-vlm-tag` will show it.

## Configuration

On first run, `dt-vlm-tag` creates `~/.darktable-vlm-tagger/` (mirroring
darktable's own `~/.config/darktable`) containing:

- `config.toml` — model, Ollama host, target image resolution, mipmap level,
  darktable library path, default output mode. Any value can be overridden
  per run with the matching `--flag`.
- `prompt.txt` — an editable copy of the prompt sent to the model.
- `vocab.json` — an editable copy of the tag vocabulary.

Edit these directly; the installed package's copies are only used to seed a
fresh config directory and are never read again afterwards.

## Usage

```bash
# Preview only, nothing written anywhere
dt-vlm-tag --file ~/Pictures/darktable/roll/DSCF1234.RAF --mode print

# Write results to a JSON file for review before touching any sidecar
dt-vlm-tag --folder ~/Pictures/darktable/roll --mode json --out results.json

# Write directly into the XMP sidecars
dt-vlm-tag --folder ~/Pictures/darktable/roll --mode sidecar
```

`--folder` processes every image darktable has imported from that exact
folder (a film roll); `--file` processes a single image. A file that was
never imported into the darktable library is still handled — it just always
falls back to a direct render instead of using the thumbnail cache.

**`--mode sidecar` refuses to run while darktable is open.** Close it first;
otherwise a running darktable can both lock out the fallback render and race
the sidecar write against darktable's own database.

### Resuming and re-running

Every tagged image gets a marker tag (`darktable-vlm-tagger|tagged`) written
into its sidecar. Re-running over a folder skips anything that already
carries it, so an interrupted batch run can simply be started again. Pass
`--force` to retag anyway.

A JSON-lines log is written on every run to `<config-dir>/run.log`
(configurable with `--log-file`), one entry per image, in every mode.

### One-time darktable setup

The `category` tag is designed to be flagged as a **category** in
darktable's tag manager (right-click → "as category"), the same idiom
darktable itself documents for e.g. a `places` category. This is a one-off,
manual, whole-tag-tree setting — not something this tool can or should set
on your behalf per image. Without it, `category` still works correctly as a
regular hierarchical tag; it just additionally shows up once as a flat
keyword alongside the real category value.

## Darktable in-app integration (Lua)

`lua/vlm_tagger.lua` adds a "tag with VLM" button and a keyboard shortcut
inside a **running** darktable, operating on the current lighttable
selection. Unlike `--mode sidecar`, it writes through darktable's own Lua
tag/metadata API, so tags, title and description appear **immediately** in
the UI — no restart needed. The image itself is rendered by darktable's own
exporter before being handed to `dt-vlm-tag` (`--source-file`), not sourced
from the mipmap cache — no external process can safely re-render an image
while darktable holds the library open, so darktable has to do it itself.

Install (run from inside your clone of this repo, e.g.
`cd darktable-vlm-tagger` from [step 5](#5-darktable-vlm-tagger-itself)):

```bash
mkdir -p ~/.config/darktable/lua
ln -s "$(pwd)/lua/vlm_tagger.lua" ~/.config/darktable/lua/vlm_tagger.lua
ln -s "$(pwd)/lua/json.lua" ~/.config/darktable/lua/json.lua
echo 'require "vlm_tagger"' >> ~/.config/darktable/luarc
```

Restart darktable, then in the lighttable's right panel find the **VLM
tagger** module and set the `dt-vlm-tag` executable path — typically
`~/.local/bin/dt-vlm-tag` (see [step 5](#5-darktable-vlm-tagger-itself)
above), but use `which dt-vlm-tag` if you installed it differently.
darktable's own process doesn't inherit your shell's `PATH`, so this has to
be the full path, not just `dt-vlm-tag`. Select one or more images and
press **tag with VLM**, or assign a shortcut under *Preferences →
shortcuts → lua → vlm tag selection*.

Same skip/force semantics as the CLI: an image already carrying the
`darktable-vlm-tagger|tagged` marker is skipped unless the "re-tag
already-tagged images" checkbox is on. Errors on individual images (e.g.
Ollama unreachable) don't abort the rest of the selection — check the
summary message and `-d lua` console output (or your system journal, e.g.
`journalctl --user -f`, if darktable was launched via its desktop entry
rather than a terminal).

## Not in this version

- No synonym normalisation across tagging runs.

## License

MIT — see [LICENSE](LICENSE).
