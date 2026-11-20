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
                for link in workflow["links"]:
                    _, src, srcslot, dst, dstslot, datatype = link
                    self.assertIn(src, by_id)
                    self.assertIn(dst, by_id)
                    self.assertEqual(by_id[src]["outputs"][srcslot]["type"], datatype)
                    self.assertEqual(by_id[dst]["inputs"][dstslot]["type"], datatype)


if __name__ == "__main__":
    unittest.main()
