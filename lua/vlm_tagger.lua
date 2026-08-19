--[[
  vlm_tagger.lua - darktable plugin for darktable-vlm-tagger

  Copyright (c) 2026 Sebastian Boettger

  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  darktable is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with darktable.  If not, see <http://www.gnu.org/licenses/>.
--]]

--[[
  Tags the current lighttable selection with darktable-vlm-tagger, a local
  vision-language-model auto-tagger. Unlike the CLI's `--mode sidecar`, this
  writes through darktable's own Lua tag/metadata API, so results (tags,
  title, description) appear immediately in the running UI - no restart.

  ADDITIONAL SOFTWARE NEEDED FOR THIS SCRIPT
  * dt-vlm-tag - https://github.com/seboettg/darktable-vlm-tagger
    (installed separately; point the executable widget below at its path,
    e.g. ~/programming/darktable-vlm-tagger/.venv/bin/dt-vlm-tag)

  USAGE
  * require this script from your main lua file:
    require "vlm_tagger"
  * configure the dt-vlm-tag executable path (widget in the panel below)
  * select one or more images in the lighttable
  * press "tag with VLM", or use the bound keyboard shortcut
    (Preferences -> shortcuts -> lua -> vlm tag selection)
--]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dtsys = require "lib/dtutils.system"
local json = require "json"

local MODULE = "vlm_tagger"
local EXECUTABLE = "dt-vlm-tag"
local MARKER_TAG = "darktable-vlm-tagger|tagged"
-- Must stay in sync by hand with vocab.FIELDS in the Python package
-- (src/darktable_vlm_tagger/vocab.py) - Lua can't import Python.
local FIELDS = {"category", "color", "tone", "light", "composition",
                "technique", "mood", "subject"}

du.check_min_api_version("7.0.0", MODULE)

local gettext = dt.gettext.gettext
local function _(msgid)
  return gettext(msgid)
end

local script_data = {}
script_data.metadata = {
  name = _("VLM tagger"),
  purpose = _("tag the selection with a local vision-language model"),
  author = "Sebastian Boettger",
  help = "https://github.com/seboettg/darktable-vlm-tagger",
}
script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

local state = {
  module_installed = false,
  event_registered = false,
  in_progress = false,
}

local function already_tagged(image)
  for _, tag in ipairs(image:get_tags()) do
    if tag.name == MARKER_TAG then
      return true
    end
  end
  return false
end

-- Exports the image via darktable's own live pixelpipe (dt.new_format +
-- write_image) to a fresh temp JPEG. This is deliberately not sourced from
-- the mipmap cache or a re-render by dt-vlm-tag itself: no external process
-- can safely render an image while darktable holds the library open (see
-- image_source.py's _render_with_darktable_cli guard - film-simulation
-- LUTs can silently mis-apply), so darktable itself has to do it. Capped at
-- a generous 2048px long edge; dt-vlm-tag downscales further to its own
-- configured target regardless.
local function export_image(image)
  local base = os.tmpname()
  os.remove(base)
  local export_file = base .. ".jpg"

  local exporter = dt.new_format("jpeg")
  exporter.quality = 90
  exporter.max_width = 2048
  exporter.max_height = 2048
  exporter:write_image(image, export_file, false)

  if not df.check_if_file_exists(export_file) then
    return nil
  end
  return export_file
end

-- Runs dt-vlm-tag for one image, parses its --mode json output, and
-- attaches the result via darktable's own tag/metadata API (which writes
-- both the live DB and the XMP sidecar, and refreshes the UI - unlike
-- sidecar.py, which this script deliberately does not go through).
-- Returns "tagged" | "skipped" | "error", and a reason string on error.
local function process_one(image, force)
  if not force and already_tagged(image) then
    return "skipped"
  end

  local bin = df.get_executable_path_preference(EXECUTABLE)
  if not bin or bin == "" or not df.check_if_bin_exists(bin) then
    return "error", _("dt-vlm-tag executable not configured (see panel below)")
  end

  local export_file = export_image(image)
  if not export_file then
    return "error", _("darktable could not export this image")
  end

  local out_file = df.create_tmp_file()
  if not out_file then
    os.remove(export_file)
    return "error", _("could not create a temporary output file")
  end

  local command = string.format(
    "%s --image-id %d --source-file %s --mode json --out %s",
    df.sanitize_filename(bin), image.id,
    df.sanitize_filename(export_file), df.sanitize_filename(out_file))

  local ret = dtsys.external_command(command)
  os.remove(export_file)
  if ret ~= 0 then
    os.remove(out_file)
    return "error", string.format(_("dt-vlm-tag exited with code %d"), ret)
  end

  local f = io.open(out_file, "r")
  if not f then
    return "error", _("dt-vlm-tag produced no output file")
  end
  local content = f:read("*a")
  f:close()
  os.remove(out_file)

  local ok, results = pcall(json.decode, content)
  if not ok or type(results) ~= "table" then
    return "error", _("could not parse dt-vlm-tag's JSON output")
  end

  -- results is {sidecar_path = data}; --image-id always processes exactly
  -- one image, so there is exactly one entry.
  local data = nil
  for _, v in pairs(results) do
    data = v
    break
  end
  if not data then
    return "error", _("empty result from dt-vlm-tag")
  end

  for _, field in ipairs(FIELDS) do
    for _, value in ipairs(data[field] or {}) do
      dt.tags.attach(dt.tags.create(field .. "|" .. value), image)
    end
  end
  dt.tags.attach(dt.tags.create(MARKER_TAG), image)
  image.title = data.title or ""
  image.description = data.description or ""

  return "tagged"
end

local function run(force)
  if state.in_progress then
    dt.print(_("VLM tagger is already running"))
    return
  end

  local images = dt.gui.action_images
  if #images == 0 then
    dt.print(_("no image selected"))
    return
  end

  local bin = df.get_executable_path_preference(EXECUTABLE)
  if not bin or bin == "" or not df.check_if_bin_exists(bin) then
    dt.print(_("VLM tagger: dt-vlm-tag executable not configured - "
                .. "set its path in the VLM tagger panel below"))
    return
  end

  state.in_progress = true
  local job = dt.gui.create_job(_("VLM tagging"), true,
                                 function(j) j.valid = false end)

  local tagged, skipped, errors = 0, 0, 0
  for i, image in ipairs(images) do
    if not job.valid then
      break
    end
    local outcome, reason = process_one(image, force)
    if outcome == "tagged" then
      tagged = tagged + 1
    elseif outcome == "skipped" then
      skipped = skipped + 1
    else
      errors = errors + 1
      local message = string.format("vlm_tagger: %s (image id %d): %s",
                                     tostring(image.filename), image.id,
                                     tostring(reason))
      dt.print_log(message)
      dt.print(message)
    end
    job.percent = i / #images
  end

  job.valid = false

  -- image.title/image.description (unlike dt.tags.attach) raise no GUI
  -- refresh signal on their own - confirmed in darktable's source
  -- (src/lua/image.c's metadata_member only calls dt_metadata_set() +
  -- dt_image_synch_xmp(), nothing else), so the metadata editor panel
  -- keeps showing stale (empty) title/description until something else
  -- triggers DT_SIGNAL_SELECTION_CHANGED, which it does listen to.
  -- Re-selecting the same images forces exactly that signal.
  if tagged > 0 then
    dt.gui.selection(images)
  end

  state.in_progress = false
  dt.print(string.format(_("VLM tagger: %d tagged, %d skipped, %d errors"),
                          tagged, skipped, errors))
end

-- GUI: a lighttable panel with the executable path, a re-tag checkbox and
-- the "tag with VLM" button.
local retag_checkbox = dt.new_widget("check_button") {
  label = _("re-tag already-tagged images"),
  value = false,
}

local tag_button = dt.new_widget("button") {
  label = _("tag with VLM"),
  clicked_callback = function() run(retag_checkbox.value) end,
}

local container = dt.new_widget("box") {
  orientation = "vertical",
  df.executable_path_widget({EXECUTABLE}),
  retag_checkbox,
  tag_button,
}

local function install_module()
  if not state.module_installed then
    dt.register_lib(MODULE, _("VLM tagger"), true, true,
      {[dt.gui.views.lighttable] = {"DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100}},
      container, nil, nil)
    state.module_installed = true
  end
end

local function destroy()
  dt.gui.libs[MODULE].visible = false
  dt.destroy_event(MODULE, "shortcut")
end

local function restart()
  dt.gui.libs[MODULE].visible = true
end

script_data.destroy = destroy
script_data.restart = restart
script_data.destroy_method = "hide"
script_data.show = restart

if dt.gui.current_view().id == "lighttable" then
  install_module()
elseif not state.event_registered then
  dt.register_event(MODULE, "view-changed", function(_event, _old_view, new_view)
    if new_view.name == "lighttable" then
      install_module()
    end
  end)
  state.event_registered = true
end

dt.register_event(MODULE, "shortcut",
  function(_event, _shortcut) run(false) end,
  _("VLM tag selection"))

return script_data
