"""Regression tests for ComfyUI-Reverb-TTS without downloading large models."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_fake_folder_paths = types.ModuleType("folder_paths")
_fake_folder_paths.models_dir = ""
_fake_folder_paths.get_output_directory = lambda: ""
sys.modules["folder_paths"] = _fake_folder_paths

spec = importlib.util.spec_from_file_location("reverb_nodes_under_test", ROOT / "nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


def write_model(root, variant):
    paths = [
        root / "Reverb-TTS" / "model" / "safetensors" / variant / "model.safetensors",
        root / "Reverb-TTS" / "model" / "safetensors" / variant / "config.json",
        root / "Reverb-TTS" / "model" / "tokenizer.json",
        root / "higgs-audio-v2-tokenizer" / "model.safetensors",
        root / "higgs-audio-v2-tokenizer" / "config.json",
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()


class LoaderTests(unittest.TestCase):
    def test_loader_fields_no_redundant_variant_or_directory(self):
        fields = nodes.ReverbLoadModel.INPUT_TYPES()
        self.assertEqual(set(fields["required"]), {"model", "device"})
        self.assertNotIn("variant", fields["required"])
        self.assertNotIn("models_directory", fields.get("optional", {}))
        self.assertIn("force_fp32_compute", fields["optional"])
        self.assertFalse(fields["optional"]["force_fp32_compute"][1]["default"])

    def test_detect_one_installed_model_no_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            write_model(root, "fp32")
            with mock.patch.object(nodes, "PREFERRED_MODELS_DIR", root), mock.patch.object(
                _fake_folder_paths, "models_dir", str(Path(tmp) / "comfy")
            ), mock.patch.dict("os.environ", {"REVERB_TTS_MODELS_DIR": ""}):
                found = nodes._installed_model_choices()
                self.assertEqual(list(found), ["Auto / fp32"])
                self.assertEqual(found["Auto / fp32"], (str(root.resolve()), "fp32"))

    def test_detect_multiple_precisions_without_inventing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            write_model(root, "fp32")
            write_model(root, "int8")
            with mock.patch.object(nodes, "PREFERRED_MODELS_DIR", root), mock.patch.object(
                _fake_folder_paths, "models_dir", str(Path(tmp) / "comfy")
            ), mock.patch.dict("os.environ", {"REVERB_TTS_MODELS_DIR": ""}):
                found = nodes._installed_model_choices()
                self.assertEqual(set(found), {"Auto / fp32", "Auto / int8"})

    def test_loader_routes_selected_model_to_backend(self):
        with mock.patch.object(nodes, "_installed_model_choices", return_value={"Auto / fp32": ("Z:/models", "fp32")}), mock.patch.object(
            nodes, "_get_model", return_value=object()
        ) as get_model:
            (out,) = nodes.ReverbLoadModel().run("Auto / fp32", "auto")
            self.assertIsNotNone(out)
            get_model.assert_called_once_with("fp32", "auto", False, "Z:/models")

    def test_loader_rejects_missing_model(self):
        with mock.patch.object(nodes, "_installed_model_choices", return_value={}):
            with self.assertRaises(FileNotFoundError):
                nodes.ReverbLoadModel().run("Auto / fp32", "auto")


class VoiceSelectionTests(unittest.TestCase):
    def test_preset_dropdown_discovers_local_json_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            voices = root / "Reverb-TTS" / "voices"
            voices.mkdir(parents=True)
            (voices / "pt_male_standard.json").write_text("{}", encoding="utf-8")
            (voices / "en_female_test.json").write_text("{}", encoding="utf-8")
            (voices / "not_a_voice.txt").write_text("dummy", encoding="utf-8")
            with mock.patch.object(nodes, "_installed_model_choices", return_value={
                "Auto / fp32": (str(root), "fp32")
            }):
                self.assertEqual(nodes._preset_names(), ["en_female_test", "pt_male_standard"])
                preset_type = nodes.ReverbPresetVoice.INPUT_TYPES()["required"]["preset_name"][0]
                self.assertEqual(preset_type, ["en_female_test", "pt_male_standard"])
                self.assertIsInstance(preset_type, list)

    def test_no_preset_dropdown_uses_descriptive_placeholder(self):
        with mock.patch.object(nodes, "_installed_model_choices", return_value={}):
            preset = nodes.ReverbPresetVoice.INPUT_TYPES()["required"]["preset_name"]
            self.assertEqual(preset[0], ["(nenhuma voz predefinida encontrada)"])
            with self.assertRaises(FileNotFoundError):
                class Dummy:
                    presets = Path("/")
                with mock.patch.object(Dummy, "preset_voices", return_value=[] , create=True):
                    nodes.ReverbPresetVoice().run(Dummy(), "(nenhuma voz predefinida encontrada)")

    def test_npz_library_dropdown_lists_actual_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp)
            (profiles / "voz_a.npz").touch()
            (profiles / "voz_b.npz").touch()
            (profiles / "skip.json").touch()
            with mock.patch.object(nodes, "_voices_dir", return_value=profiles):
                choices = nodes.ReverbLibraryVoice.INPUT_TYPES()["required"]["voice_name"][0]
                self.assertEqual(choices, ["voz_a", "voz_b"])
                self.assertIsInstance(choices, list)

    def test_empty_npz_library_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(nodes, "_voices_dir", return_value=Path(tmp)):
                self.assertEqual(nodes.ReverbLibraryVoice.INPUT_TYPES()["required"]["voice_name"][0], ["(nenhuma voz salva)"])

    def test_preset_dropdown_selection_reaches_model(self):
        class FakeModel:
            presets = Path("/")
            def preset_voices(self):
                return ["pt_male_standard", "en_female_test"]
            def preset_voice(self, name):
                return {"chosen": name}
        output = nodes.ReverbPresetVoice().run(FakeModel(), "pt_male_standard")
        self.assertEqual(output[0]["chosen"], "pt_male_standard")
        with self.assertRaises(ValueError):
            nodes.ReverbPresetVoice().run(FakeModel(), "not_installed")

    def test_npz_library_selection_loads_selected_profile(self):
        with mock.patch.object(nodes.ReverbLoadVoice, "run", return_value=("voice_object",)) as load:
            output = nodes.ReverbLibraryVoice().run("voz_b")
            self.assertEqual(output, ("voice_object",))
            load.assert_called_once_with("voz_b")

    def test_generate_speech_forwards_speed_one(self):
        class FakeModel:
            sample_rate = 24000
            def generate(self, *args, **kwargs):
                self.options = kwargs
                return [0.0, 0.0]
        model = FakeModel()
        with mock.patch.object(nodes, "_to_audio", return_value={"waveform": "ok", "sample_rate": 24000}):
            result = nodes.ReverbGenerate().run(model, "voice", "Olá", "pt", 42, 1.0, 8)
        self.assertEqual(result[0]["sample_rate"], 24000)
        self.assertEqual(model.options["speed"], 1.0)
        self.assertEqual(model.options["steps"], 8)
        with self.assertRaises(ValueError):
            nodes.ReverbGenerate().run(model, "voice", "Olá", "pt", 42, 8.0, 8)

    def test_generate_speed_default_one_and_valid_range(self):
        speed = nodes.ReverbGenerate.INPUT_TYPES()["required"]["speed"]
        self.assertEqual(speed[1]["default"], 1.0)
        self.assertEqual(speed[1]["min"], 0.5)
        self.assertEqual(speed[1]["max"], 2.0)


class SaveWavTests(unittest.TestCase):
    def test_audio_ui_not_required_input(self):
        fields = nodes.ReverbSaveWav.INPUT_TYPES()
        self.assertIn("audio", fields["required"])
        self.assertNotIn("audioUI", fields["required"])
        self.assertEqual(fields["optional"]["audioUI"][0], "AUDIO_UI")

    def test_save_produces_valid_wav_and_preview_metadata(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            _fake_folder_paths.get_output_directory = lambda: tmp
            data = np.zeros(2400, dtype=np.float32)
            with mock.patch.object(nodes, "_comfy_audio_to_numpy", return_value=(data, 24000)):
                result = nodes.ReverbSaveWav().run(object(), "Reverb/narracao", "PCM_24")
            self.assertEqual(len(result["result"]), 1)
            path = Path(result["result"][0])
            self.assertTrue(path.is_file())
            audio, sr = sf.read(path)
            self.assertEqual(len(audio), 2400)
            self.assertEqual(sr, 24000)
            self.assertEqual(result["ui"]["audio"], [{
                "filename": path.name, "subfolder": "Reverb", "type": "output"
            }])

    def test_save_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_folder_paths.get_output_directory = lambda: tmp
            with mock.patch.object(nodes, "_comfy_audio_to_numpy", return_value=([0.0], 24000)):
                with self.assertRaises(ValueError):
                    nodes.ReverbSaveWav().run(object(), "../danger", "PCM_24")


class WorkflowTests(unittest.TestCase):
    def test_all_public_workflows_have_valid_loader_and_links(self):
        files = list((ROOT / "workflows").glob("*.json"))
        self.assertEqual(len(files), 6)
        for p in files:
            with self.subTest(p=p.name):
                workflow = json.loads(p.read_text(encoding="utf-8"))
                by_id = {n["id"]: n for n in workflow["nodes"]}
                self.assertEqual(len(by_id), len(workflow["nodes"]))
                loads = [n for n in workflow["nodes"] if n["type"] == "ReverbLoadModel"]
                self.assertEqual(len(loads), 1)
                self.assertEqual(loads[0]["widgets_values"], ["Auto / fp32", "auto", False])
                for generate in (n for n in workflow["nodes"] if n["type"] == "ReverbGenerate"):
                    values = generate["widgets_values"]
                    self.assertEqual(len(values), 6)
                    self.assertEqual(values[3], "fixed")
                    self.assertEqual(values[4], 1.0)
                    self.assertEqual(values[5], 8)
                    self.assertEqual(generate["widgets_values_named"]["speed"], 1.0)
                    self.assertEqual(generate["widgets_values_named"]["control_after_generate"], "fixed")
                for preset in (n for n in workflow["nodes"] if n["type"] == "ReverbPresetVoice"):
                    self.assertEqual(preset["widgets_values"], ["pt_male_standard"])
                for link in workflow["links"]:
                    _, src, srcslot, dst, dstslot, datatype = link
                    self.assertIn(src, by_id)
                    self.assertIn(dst, by_id)
                    self.assertEqual(by_id[src]["outputs"][srcslot]["type"], datatype)
                    self.assertEqual(by_id[dst]["inputs"][dstslot]["type"], datatype)


if __name__ == "__main__":
    unittest.main()
