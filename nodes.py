"""ComfyUI integration for Software Mansion Reverb-TTS (experimental v0.3.3 — local weights, voice library and audio preview)."""
import os
import re
import gc
from pathlib import Path
import threading

import numpy as np
import soundfile as sf

_MODEL_CACHE = {}
_LOCK = threading.RLock()
LANGUAGES = ("pt", "en", "es", "fr", "de", "it", "pl")
VARIANTS = ("fp32", "fp16", "int8", "int4")


def _lib():
    try:
        from reverb_tts import TTS, Voice
    except ImportError as exc:
        raise RuntimeError(
            "Reverb-TTS nao instalado no Python do ComfyUI. Instale com: "
            "C:\\AI\\ComfyUI\\python_embeded\\python.exe -m pip install reverb-tts"
        ) from exc
    return TTS, Voice


# Pasta escolhida pelo usuario para nao misturar TTS com diffusion_models.
# Compatibilidade: fallback para modelos em ComfyUI/models/reverb_tts.
PREFERRED_MODELS_DIR = Path(r"C:\AI\ComfyUI-Data\models\reverb_tts")
MODELS_DIR_ENV = "REVERB_TTS_MODELS_DIR"


def _models_root(models_directory=""):
    """Resolve models without downloading weights.

    Priority: explicit loader path > environment variable > existing legacy
    Windows directory > ComfyUI's configured models directory.
    """
    import folder_paths
    raw = str(models_directory or "").strip() or os.environ.get(MODELS_DIR_ENV, "").strip()
    if raw:
        selected = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
        if not selected.is_dir():
            raise FileNotFoundError(
                f"Reverb-TTS models directory does not exist: {selected}. "
                "Create it and install the weights manually."
            )
        return selected
    if PREFERRED_MODELS_DIR.is_dir():
        return PREFERRED_MODELS_DIR.resolve()
    return (Path(folder_paths.models_dir) / "reverb_tts").resolve()


def _validate_local_files(root, variant):
    weights = root / 'Reverb-TTS' / 'model' / 'safetensors' / variant
    tokenizer = root / 'Reverb-TTS' / 'model'
    codec = root / 'higgs-audio-v2-tokenizer'
    missing = [p for p in (
        weights / 'config.json', weights / 'model.safetensors',
        tokenizer / 'tokenizer.json',
        codec / 'config.json', codec / 'model.safetensors',
    ) if not p.is_file()]
    if missing:
        lines = '\n'.join('  - ' + str(p) for p in missing)
        raise FileNotFoundError(
            'Reverb-TTS: faltam arquivos locais. Download automatico DESATIVADO.\n'
            f'Arquivos ausentes:\n{lines}\n'
            'Consulte README.md, secao Download manual.'
        )
    return root / 'Reverb-TTS', codec


def _get_model(variant, device, true_fp32, models_directory=""):
    import torch
    # Valida ANTES de importar o pacote ou chamar de pretrained, sem qualquer download.
    root = _models_root(models_directory)
    model_path, codec_path = _validate_local_files(root, variant)
    TTS, _ = _lib()
    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    if dev == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selecionada, mas torch.cuda.is_available() == False.')
    dtype = torch.float32 if true_fp32 else None
    key = (str(model_path.resolve()), str(codec_path.resolve()), variant, dev, true_fp32)
    with _LOCK:
        if key not in _MODEL_CACHE:
            # A API oficial aceita pastas locais para AMBOS os modelos.
            # Nao usamos o nome de um repositorio Hugging Face em nenhum argumento.
            kwargs = dict(variant=variant, device=dev, codec=str(codec_path))
            if dtype is not None:
                kwargs['dtype'] = dtype
            _MODEL_CACHE[key] = TTS.from_pretrained(str(model_path), **kwargs)
        return _MODEL_CACHE[key]


def _comfy_audio_to_numpy(audio):
    import torch
    if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
        raise ValueError('Entrada AUDIO invalida: esperado dict com waveform e sample_rate')
    x = audio['waveform']
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError('Apenas um arquivo por vez. Batch AUDIO com mais de um item nao suportado.')
        x = x[0]  # [channels, samples]
    if x.ndim == 2:
        if x.shape[0] > 8:
            raise ValueError('Formato AUDIO inesperado: esperado [canais, amostras].')
        x = x.mean(dim=0)
    elif x.ndim != 1:
        raise ValueError('Waveform invalida. Esperado [batch, canais, amostras].')
    samples = x.detach().float().cpu().numpy().astype(np.float32, copy=False)
    return samples, int(audio['sample_rate'])


def _to_audio(data, sample_rate):
    import torch
    samples = np.asarray(data, dtype=np.float32).reshape(-1).copy()
    if not np.isfinite(samples).all():
        raise ValueError('Modelo produziu amostras nao finitas.')
    return {'waveform': torch.from_numpy(samples).reshape(1, 1, -1), 'sample_rate': int(sample_rate)}


def _voices_dir():
    """Pasta persistente dos perfis locais, ao lado dos modelos, nunca em output/."""
    root = _models_root() / 'voices_custom'
    root.mkdir(parents=True, exist_ok=True)
    return root


def _library_names():
    """Lista segura de nomes disponíveis; o menu é reconstruído ao atualizar nodes."""
    return sorted(p.stem for p in _voices_dir().glob('*.npz') if p.is_file() and not p.is_symlink())


def _voice_path(name):
    stem = _safe_stem(name)
    path = _voices_dir() / (stem + '.npz')
    if path.is_symlink():
        raise ValueError('Perfis simbolicos nao sao aceitos.')
    return path


def _safe_stem(name):
    name = re.sub(r'[^\w.-]+', '_', name.strip(), flags=re.UNICODE).strip('._')[:96]
    if not name:
        raise ValueError('Nome vazio ou invalido.')
    return name


class ReverbLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'variant': (list(VARIANTS), {'default': 'fp32'}),
            'device': (['auto', 'cuda', 'cpu'], {'default': 'auto'}),
            'force_fp32_compute': ('BOOLEAN', {'default': True}),
        }, 'optional': {
            'models_directory': ('STRING', {'default': '', 'multiline': False,
                'tooltip': 'Optional local folder containing Reverb-TTS and higgs-audio-v2-tokenizer. Blank uses REVERB_TTS_MODELS_DIR or standard folders.'}),
        }}
    RETURN_TYPES = ('REVERB_TTS_MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    def run(self, variant, device, force_fp32_compute, models_directory=""):
        return (_get_model(variant, device, force_fp32_compute, models_directory),)


class ReverbVoiceFromAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('REVERB_TTS_MODEL',),
            'reference_audio': ('AUDIO',),
            'reference_transcript': ('STRING', {'default': '', 'multiline': True}),
            'reference_language': (list(LANGUAGES), {'default': 'pt'}),
        }}
    RETURN_TYPES = ('REVERB_VOICE',)
    RETURN_NAMES = ('voice',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    def run(self, model, reference_audio, reference_transcript, reference_language):
        if not reference_transcript.strip():
            raise ValueError('Informe a transcricao EXATA do audio de referencia.')
        samples, sr = _comfy_audio_to_numpy(reference_audio)
        if not (2.0 <= len(samples) / sr <= 25.0):
            raise ValueError('Referencia deve ter 2 a 25 segundos; recomendado 3 a 10 segundos.')
        # API oficial aceita samples e sample_rate; evita WAV temporario.
        voice = model.voice(samples, reference_transcript.strip(), language=reference_language, sample_rate=sr)
        return (voice,)


class ReverbPresetVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('REVERB_TTS_MODEL',),
            'preset_name': ('STRING', {'default': 'pt_male_standard'}),
        }}
    RETURN_TYPES = ('REVERB_VOICE',)
    RETURN_NAMES = ('voice',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    def run(self, model, preset_name):
        # Impede o fallback da biblioteca que consulta Hugging Face quando presets nao estao locais.
        if model.presets is None or not model.presets.is_dir():
            raise FileNotFoundError(
                'Preset local ausente. Baixe os JSONs para Reverb-TTS/voices/. '
                'Download automatico desativado.'
            )
        available = model.preset_voices()
        name = preset_name.strip()
        if name not in available:
            raise ValueError(f'Preset {name!r} inexistente. Disponiveis: {available}')
        return (model.preset_voice(name),)


class ReverbSaveVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'voice': ('REVERB_VOICE',),
            'voice_name': ('STRING', {'default': 'minha_voz'}),
            'overwrite': ('BOOLEAN', {'default': False}),
        }}
    RETURN_TYPES = ('REVERB_VOICE', 'STRING')
    RETURN_NAMES = ('voice', 'saved_path')
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'
    OUTPUT_NODE = True

    def run(self, voice, voice_name, overwrite):
        path = _voice_path(voice_name)
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} ja existe. Marque overwrite para substituir.')
        # Grava em arquivo temporario e depois substitui o destino de forma atomica.
        import uuid
        temp = path.with_name(f'.{path.stem}-{uuid.uuid4().hex}.npz')
        try:
            voice.save(str(temp))
            if not temp.is_file():
                raise RuntimeError(f'A biblioteca nao criou o arquivo: {temp}')
            if not overwrite and path.exists():
                raise FileExistsError(f'{path} ja existe. Marque overwrite para substituir.')
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return (voice, str(path))


class ReverbLoadVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'voice_name': ('STRING', {'default': 'minha_voz'})}}
    RETURN_TYPES = ('REVERB_VOICE',)
    RETURN_NAMES = ('voice',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    def run(self, voice_name):
        _, Voice = _lib()
        path = _voice_path(voice_name)
        if not path.is_file():
            raise FileNotFoundError(f'Voz nao encontrada: {path}. Crie com Reverb Save Voice.')
        return (Voice.load(str(path)),)


class ReverbLibraryVoice:
    """Seleciona perfis NPZ por nome na biblioteca, sem digitar caminho."""
    @classmethod
    def INPUT_TYPES(cls):
        names = _library_names()
        return {'required': {'voice_name': (names or ['(nenhuma voz salva)'],)}}
    RETURN_TYPES = ('REVERB_VOICE',)
    RETURN_NAMES = ('voice',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    @classmethod
    def IS_CHANGED(cls, voice_name):
        path = _voice_path(voice_name) if voice_name != '(nenhuma voz salva)' else None
        if path is None or not path.exists():
            return float('nan')
        stat = path.stat()
        return (str(path), stat.st_size, stat.st_mtime_ns)

    def run(self, voice_name):
        if voice_name == '(nenhuma voz salva)':
            raise FileNotFoundError('Biblioteca vazia. Use Reverb | Save Voice para criar um perfil NPZ e atualize a pagina do ComfyUI.')
        return ReverbLoadVoice().run(voice_name)


class ReverbGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('REVERB_TTS_MODEL',),
            'voice': ('REVERB_VOICE',),
            'text': ('STRING', {'default': 'Olá! Esta é uma voz gerada localmente.', 'multiline': True}),
            'language': (list(LANGUAGES), {'default': 'pt'}),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 2147483647}),
            'speed': ('FLOAT', {'default': 1.0, 'min': 0.5, 'max': 2.0, 'step': 0.05, 'round': 0.01}),
            'steps': ('INT', {'default': 8, 'min': 1, 'max': 64}),
        }}
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('audio',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'

    def run(self, model, voice, text, language, seed, speed, steps):
        if not text.strip():
            raise ValueError('Texto de saida vazio.')
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f'speed deve estar entre 0.5 e 2.0, recebido: {speed}')
        generated = model.generate(text.strip(), voice, language=language, seed=seed, speed=speed, steps=steps)
        return (_to_audio(generated, model.sample_rate),)


class ReverbSaveWav:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio': ('AUDIO',),
            'filename_prefix': ('STRING', {'default': 'Reverb/narracao'}),
            'subtype': (['PCM_24', 'PCM_16', 'FLOAT'], {'default': 'PCM_24'}),
            # The ComfyUI frontend's built-in AudioWidget extension recognizes
            # AUDIO_UI and creates its native inline HTML audio player. The
            # ui.audio payload below then populates it after execution.
            'audioUI': ('AUDIO_UI', {}),
        }}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('saved_path',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'
    OUTPUT_NODE = True

    def run(self, audio, filename_prefix, subtype):
        import folder_paths
        samples, sr = _comfy_audio_to_numpy(audio)
        base = Path(folder_paths.get_output_directory()).resolve()
        prefix = filename_prefix.strip().replace('\\', '/')
        parts = [p for p in prefix.split('/') if p and p not in ('.', '..')]
        if len(parts) != len([p for p in prefix.split('/') if p]):
            raise ValueError('Prefixo invalido: nao use . ou ..')
        if not parts:
            raise ValueError('Prefixo de arquivo vazio.')
        dest_dir = base.joinpath(*parts[:-1]).resolve()
        if not dest_dir.is_relative_to(base):
            raise ValueError('Caminho fora da pasta output.')
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(parts[-1])
        idx = 1
        while (dest_dir / f'{stem}_{idx:05d}.wav').exists():
            idx += 1
        path = dest_dir / f'{stem}_{idx:05d}.wav'
        sf.write(str(path), samples, sr, format='WAV', subtype=subtype)
        # ComfyUI's built-in audio widget expects UI metadata in this exact shape.
        # Keep the previous STRING output for backward compatibility with workflows.
        # The saved WAV is served from /view?filename=...&subfolder=...&type=output.
        subfolder = dest_dir.relative_to(base).as_posix()
        if subfolder == '.':
            subfolder = ''
        return {
            'ui': {'audio': [{
                'filename': path.name,
                'subfolder': subfolder,
                'type': 'output',
            }]},
            'result': (str(path),),
        }


class ReverbUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'unload': ('BOOLEAN', {'default': True})}}
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'run'
    CATEGORY = 'audio/Reverb TTS'
    OUTPUT_NODE = True

    def run(self, unload):
        if unload:
            with _LOCK:
                _MODEL_CACHE.clear()
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return ('Model cache cleared' if unload else 'No action',)


NODE_CLASS_MAPPINGS = {
    'ReverbLoadModel': ReverbLoadModel,
    'ReverbVoiceFromAudio': ReverbVoiceFromAudio,
    'ReverbPresetVoice': ReverbPresetVoice,
    'ReverbSaveVoice': ReverbSaveVoice,
    'ReverbLoadVoice': ReverbLoadVoice,
    'ReverbLibraryVoice': ReverbLibraryVoice,
    'ReverbGenerate': ReverbGenerate,
    'ReverbSaveWav': ReverbSaveWav,
    'ReverbUnload': ReverbUnload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'ReverbLoadModel': 'Reverb | Load Model (Local)',
    'ReverbVoiceFromAudio': 'Reverb | Clone Voice from Audio',
    'ReverbPresetVoice': 'Reverb | Preset Voice',
    'ReverbSaveVoice': 'Reverb | Save Voice (.npz)',
    'ReverbLoadVoice': 'Reverb | Load Voice by Name (.npz)',
    'ReverbLibraryVoice': 'Reverb | Voice Library (NPZ Selector)',
    'ReverbGenerate': 'Reverb | Generate Speech',
    'ReverbSaveWav': 'Reverb | Save WAV',
    'ReverbUnload': 'Reverb | Clear Model Cache',
}
