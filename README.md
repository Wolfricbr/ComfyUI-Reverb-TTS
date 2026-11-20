# ComfyUI-Reverb-TTS

**Local speech synthesis, voice cloning, reusable voice profiles, and WAV export for ComfyUI.**

[🇧🇷 Português](#português) · [Original Reverb-TTS model](https://huggingface.co/software-mansion/Reverb-TTS) · [Workflows](./workflows)

> **Experimental integration · v0.3.6.** This is a community ComfyUI integration, not an official Software Mansion release. Models are **not downloaded automatically**. Install the weights separately and read the license section before using generated audio.

## Features

- Local loading of Reverb-TTS **FP32, FP16, INT8, and INT4** weight variants.
- Zero-shot voice cloning from a short audio clip **and its exact transcript**.
- Official preset voices using locally installed JSON files.
- Save/load reusable custom voice profiles in **`.npz`** format.
- Select saved profiles from a voice-library dropdown.
- Text-to-speech generation with language, seed, speed and steps controls.
- Lossless WAV export (PCM 24-bit, PCM 16-bit, or float), with an audio player in the ComfyUI node.
- Clear the cached TTS model from memory.

**Important:** `.npz` voice profiles are specific to the Reverb-TTS `Voice` serialization. They are not interchangeable with Fish Audio Drama 3 `.safetensors` voice files or other TTS models.

## Requirements

- ComfyUI with Python compatible with the official `reverb-tts==1.0.0` package.
- A supported PyTorch environment; CUDA GPU recommended.
- Manually downloaded Reverb-TTS weights and **Higgs Audio V2 tokenizer/codec** weights.
- A reference recording and matching transcript for cloning.

> **Dependency warning:** `reverb-tts` can change shared Python dependencies. In one Windows ComfyUI environment, installing it introduced a `huggingface-hub 2.x` versus `diffusers 0.37.1` version conflict. Check planned package changes before installing. An import succeeding does not guarantee every other custom node will remain compatible.

## Installation

Clone into `ComfyUI/custom_nodes`:

```bat
cd /d C:\AI\ComfyUI\ComfyUI\custom_nodes
git clone https://github.com/Wolfricbr/ComfyUI-Reverb-TTS.git
```

For a **Windows portable ComfyUI** installation, preview the dependency changes first:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --dry-run -r C:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Reverb-TTS\requirements.txt
```

If compatible with your environment, install and verify:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install -r C:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Reverb-TTS\requirements.txt
C:\AI\ComfyUI\python_embeded\python.exe -c "from reverb_tts import TTS, Voice; print('Reverb-TTS import OK')"
```

These paths are **examples** for a portable Windows installation. Use your actual ComfyUI Python executable if installed elsewhere. Restart ComfyUI after installing.

## Download the models manually

**The node does not download weights automatically.** By default, this version prefers the Windows path:

```text
C:\AI\ComfyUI-Data\models\reverb_tts\
├── Reverb-TTS\
│   ├── model\
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── chat_template.jinja
│   │   └── safetensors\
│   │       ├── fp32\
│   │       │   ├── config.json
│   │       │   └── model.safetensors
│   │       ├── fp16\          (optional)
│   │       ├── int8\          (optional)
│   │       └── int4\          (optional)
│   └── voices\               (optional for voice cloning)
│       └── pt_male_standard.json
├── higgs-audio-v2-tokenizer\
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── voices_custom\            (created automatically for custom .npz voices)
```

The loader searches the existing Windows directory, `ComfyUI/models/reverb_tts`, and the directory specified by the `REVERB_TTS_MODELS_DIR` environment variable. Select the model from its dropdown. The root must contain both `Reverb-TTS/` and `higgs-audio-v2-tokenizer/`. This project does not use `diffusion_models`.

Download these components:

1. [Reverb-TTS FP32 weights](https://huggingface.co/software-mansion/Reverb-TTS/tree/main/model/safetensors/fp32): `config.json` and `model.safetensors`.
2. [Reverb tokenizer files](https://huggingface.co/software-mansion/Reverb-TTS/tree/main/model): `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`.
3. [Higgs Audio V2 tokenizer/codec](https://huggingface.co/eustlb/higgs-audio-v2-tokenizer/tree/main): `config.json`, `model.safetensors`, `preprocessor_config.json`.
4. Optional [preset voice JSONs](https://huggingface.co/software-mansion/Reverb-TTS/tree/main/voices).
5. Optional FP16, INT8 or INT4 weights from their respective `model/safetensors/` folders, with each variant's `config.json`.

For example, to download the FP32 model, common tokenizer files, and preset voices **to your local folders** using Hugging Face Hub:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='software-mansion/Reverb-TTS', local_dir=r'C:\AI\ComfyUI-Data\models\reverb_tts\Reverb-TTS', allow_patterns=['model/tokenizer.json', 'model/tokenizer_config.json', 'model/chat_template.jinja', 'model/safetensors/fp32/*', 'voices/*.json'])"
```

The separate codec must also be downloaded:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='eustlb/higgs-audio-v2-tokenizer', local_dir=r'C:\AI\ComfyUI-Data\models\reverb_tts\higgs-audio-v2-tokenizer', allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json'])"
```

These commands are **one-time manual setup steps**, not automatic runtime downloads.

## Available nodes

| Node | Function |
|---|---|
| **Reverb · Load Model (Local)** | Load locally installed FP32/FP16/INT8/INT4 weights and select CPU/CUDA. |
| **Reverb · Clone Voice from Audio** | Build a voice from an audio reference and matching transcript. |
| **Reverb · Preset Voice** | Select a locally installed official voice preset by name. |
| **Reverb · Save Voice (.npz)** | Save a reusable custom voice profile. |
| **Reverb · Load Voice by Name (.npz)** | Load an existing custom voice profile by file stem. |
| **Reverb · Voice Library (NPZ Selector)** | Pick a saved voice from a dropdown. |
| **Reverb · Generate Speech** | Generate AUDIO from text and a voice. |
| **Reverb · Save WAV** | Save a WAV and display an audio preview/player after execution. |
| **Reverb · Clear Model Cache** | Release cached model references and request CUDA cache cleanup. |

### Generation settings

- `language`: `pt`, `en`, `es`, `fr`, `de`, `it`, `pl`
- `speed`: **0.5–2.0**, step 0.05
- `steps`: default 8 (separate from speed)
- `seed`: default 42
- `variant`: `fp32`, `fp16`, `int8`, `int4`
- `force_fp32_compute`: changes compute dtype; **does not** change the weight variant

## Example workflows

Open a JSON from [`workflows/`](./workflows) in ComfyUI.

| Workflow | What it does |
|---|---|
| [01 — Clone and Generate](./workflows/01-Clonar-e-Gerar-FP32.json) | Reference audio + transcript → cloned speech WAV. |
| [02 — Preset Voice](./workflows/02-Voz-Predefinida-FP32.json) | Official local preset + text → WAV. |
| [03 — Create and Save Voice](./workflows/03-Criar-e-Salvar-Voz.json) | Reference audio + transcript → reusable `.npz` voice. |
| [04 — Load Voice and Generate](./workflows/04-Carregar-Voz-Salva-e-Gerar-FP32.json) | Existing custom voice → WAV. |
| [05 — Clone, Save and Generate](./workflows/05-Clonar-Salvar-e-Gerar-FP32.json) | One execution produces both a voice profile and speech WAV. |
| [07 — Voice Library Selector](./workflows/07-Biblioteca-de-Vozes-Seletor.json) | Select a `.npz` from the library and synthesize speech. |

**Suggested first test:** Workflow 02 with `pt_male_standard` and a short sentence. Then try Workflow 03 followed by 07. To refresh the saved voice list, reload the ComfyUI page or recreate the selector node.

### Voice library

Custom profiles are stored in:

```text
C:\AI\ComfyUI-Data\models\reverb_tts\voices_custom\
```

Use `Reverb | Save Voice (.npz)` to create them. Existing files are not overwritten unless `overwrite` is enabled. The optional `migrar_vozes_antigas.bat` copies profiles from an older output location without deleting the originals.

## Troubleshooting

**`No module named 'reverb_tts'`** — install the Python package in the **same interpreter** used by ComfyUI.

**Missing model files** — check both the Reverb variant folder **and** the Higgs codec folder. A `.safetensors` file by itself is not sufficient.

**`speed` greater than 2** — use the current workflows; older JSONs may have shifted widget values.

**No voice in the selector** — save a `.npz` in `voices_custom` and refresh the frontend.

**Dependency conflicts** — run `python -m pip check` using ComfyUI's interpreter. Resolve incompatible dependency requirements carefully; do not indiscriminately upgrade or downgrade packages.

## License and responsible use

This repository contains community integration code and example workflows; **no model weights are distributed here**. The [upstream Reverb-TTS project](https://huggingface.co/software-mansion/Reverb-TTS) has its own licenses. At the time of writing, the **Reverb-TTS model weights are licensed CC BY-NC 4.0 (non-commercial)**, while upstream code licensing is separate. The present repository **declares Apache-2.0 for its own integration code** (see `LICENSE`); the license for model weights remains separate.

Use reference voices only with appropriate consent or authorization, and review the upstream licenses before commercial use.

---

<a id="português"></a>
## 🇧🇷 Português

**ComfyUI-Reverb-TTS** é uma integração experimental do Reverb-TTS ao ComfyUI, com geração de voz local, clonagem por referência, perfis de voz reutilizáveis em `.npz` e exportação WAV com player integrado.

### Recursos principais

- Carregamento **local**, sem download automático, das variantes FP32, FP16, INT8 e INT4.
- Clonagem de voz a partir de áudio e **transcrição exata**.
- Vozes oficiais predefinidas armazenadas em JSON.
- Biblioteca de vozes clonadas `.npz` com seletor.
- Geração de fala em português (`pt`) e outros idiomas.
- WAV PCM 24-bit, PCM 16-bit ou float, com prévia dentro do node.

### Instalação rápida

1. Clone o repositório dentro de `ComfyUI/custom_nodes/`.
2. Instale `requirements.txt` **no Python que executa seu ComfyUI**, após conferir conflitos de dependências com `--dry-run`.
3. Baixe manualmente os pesos do Reverb, o tokenizer e o codec Higgs para a estrutura explicada acima.
4. Reinicie o ComfyUI e abra um dos sete workflows.

**Pasta principal configurada nesta versão para Windows:**

```text
C:\AI\ComfyUI-Data\models\reverb_tts\
```

Na v0.3.2, o node **Reverb | Load Model (Local)** aceita o campo opcional `models_directory` para informar outra pasta de modelos. Também é possível usar a variável de ambiente `REVERB_TTS_MODELS_DIR`. Se não houver caminho configurado, o sistema preserva o caminho Windows acima (quando existe) e, caso contrário, procura `ComfyUI/models/reverb_tts`.

### Para que serve cada workflow?

| Nº | Workflow | Resultado |
|---|---|---|
| 01 | Clonar e gerar | Áudio WAV com voz clonada |
| 02 | Voz predefinida | Narração com voz oficial |
| 03 | Criar e salvar voz | Perfil de voz `.npz` |
| 04 | Carregar voz e gerar | WAV a partir de perfil salvo |
| 05 | Clonar, salvar e gerar | Perfil `.npz` + WAV |
| 07 | Biblioteca de vozes | Escolha a voz no menu e gere WAV |

**Biblioteca de vozes:** `C:\AI\ComfyUI-Data\models\reverb_tts\voices_custom\`. Novas vozes podem exigir atualizar a página (F5) para aparecer no seletor.

**Atenção:** a integração ainda é experimental. Os testes relatados anteriormente foram de estrutura e código; a geração real e a compatibilidade com todos os ambientes precisam ser validadas. Os pesos Reverb-TTS são **CC BY-NC 4.0**, com restrições a uso comercial. Use vozes de outras pessoas somente com autorização.


## v0.3.2 — Custom model path / Caminho personalizado

**English:** The model loader now supports an optional `models_directory` input. Leave it blank to use `REVERB_TTS_MODELS_DIR` if set, otherwise the existing legacy Windows directory (`C:\\AI\\ComfyUI-Data\\models\\reverb_tts`) when present, otherwise `ComfyUI/models/reverb_tts`. No models are downloaded automatically. The directory must contain both `Reverb-TTS` and `higgs-audio-v2-tokenizer` with their expected contents.

**Important:** This setting applies to **model loading**. In the current version, the `.npz` voice library still uses the default/environment-resolved root, so set `REVERB_TTS_MODELS_DIR` as well if you move your entire library, or keep `voices_custom` in its existing location. Existing workflows remain compatible.

**Português:** O carregador aceita o campo opcional `models_directory`. Se ficar vazio, usa a variável `REVERB_TTS_MODELS_DIR` quando definida; senão, mantém a pasta Windows já utilizada (quando existir) e, por último, `ComfyUI/models/reverb_tts`. Não há download automático. **Atenção:** a biblioteca de vozes `.npz` ainda utiliza o diretório padrão/definido pela variável de ambiente; o campo do node não muda a pasta dos perfis.

### Dependency compatibility / Compatibilidade de dependências

The package `reverb-tts==1.0.0` can introduce conflicting dependency requirements in an existing ComfyUI installation. Run `python -m pip install --dry-run -r requirements.txt` before installation, and `python -m pip check` afterwards. Do not assume all custom nodes will remain compatible. No dependency upgrades are required solely to pull this v0.3.2 code change.

O pacote `reverb-tts==1.0.0` pode produzir conflitos em instalações com outros custom nodes. Verifique o `--dry-run` antes de instalar e execute `pip check` depois. Não é necessário reinstalar dependências apenas para atualizar o código desta versão.


## v0.3.3 — Native inline WAV player / Player WAV integrado

**Fix:** The `Reverb | Save WAV` node now explicitly declares ComfyUI's `AUDIO_UI` widget so the frontend creates the **native inline audio player**. The output still returns `ui.audio` metadata, retains the `saved_path` output and saves lossless WAV in the chosen subtype. A frontend widget was missing before, so returning `ui.audio` alone did not produce a preview.

**After updating:** completely restart ComfyUI, refresh the browser, **remove the old Save WAV node and add a fresh `Reverb | Save WAV`** if the player does not immediately appear, reconnect its audio input, then run the workflow. The player is populated **after a new execution**; previously generated audio does not automatically reappear. No extra Python dependencies or model downloads are needed. This change is based on ComfyUI's built-in `AUDIO_UI` widget interface; actual frontend behavior should still be confirmed on the user's installation.

**Português:** O node `Reverb | Save WAV` passa a declarar `AUDIO_UI`, necessário para renderizar o player nativo no frontend. Reinicie o ComfyUI, atualize a página e, se necessário, **apague apenas o node antigo de salvamento, adicione o novo e reconecte a entrada `audio`**. Execute novamente para preencher o player. Não é necessário baixar pesos nem reinstalar bibliotecas. A interface precisa ser verificada no ComfyUI real.


## v0.3.4 — Installed-model dropdown / Lista de modelos instalados

**English:** `Reverb | Load Model (Local)` now includes the optional `installed_model` dropdown. It detects locally installed Reverb variants (FP32, FP16, INT8, INT4) from the configured environment directory, the legacy Windows folder and the ComfyUI models folder. Only variants with both weights/config and the required codec/tokenizer are shown. No downloads occur. Select `Use variant / custom path` to retain the previous `variant` and `models_directory` behavior. The new field is appended after existing widgets for compatibility. Restart ComfyUI and recreate the loader node if its dropdown does not appear; refresh the page after adding model files.

**Português:** O node `Reverb | Load Model (Local)` agora tem o menu `installed_model`, que mostra as variantes instaladas localmente com os arquivos necessários. Exemplo: `Windows / fp32`. Se escolher `Use variant / custom path`, ele continua usando os controles antigos `variant` e `models_directory`. Não baixa pesos automaticamente. Reinicie o ComfyUI e recrie apenas o node de carregamento se o menu não aparecer. A interface ainda precisa ser validada em uma instalação real.


## v0.3.5 — Simplified Load Model / Carregador simplificado

**English:** The model loader now uses a single `model` dropdown listing only complete local model variants found in supported directories (for example `Auto / fp32`, `Windows / int8`). The redundant `variant` field and the per-node `models_directory` field were removed. `device` remains available, while `force_fp32_compute` is an **optional advanced** checkbox defaulting to `false`. An alternative models directory can still be configured globally using the `REVERB_TTS_MODELS_DIR` environment variable before launching ComfyUI. Model weights are never downloaded automatically.

The six public workflow JSON files (01–05 and 07) were migrated to `Auto / fp32` and will work when an FP32 model is installed in one of the scanned folders. If only another precision is installed, select it from the dropdown. Existing saved workflows using the old loader widget layout should be reloaded using the updated examples or have the loader node replaced. **Restart ComfyUI** after updating and refresh to rescan model folders.

**Português:** O node **Reverb | Load Model (Local)** tem agora um único menu `model`, com os modelos instalados detectados automaticamente. Os campos `variant` e `models_directory` foram retirados da interface. Permanecem `device` e a opção avançada `force_fp32_compute`, desativada por padrão. Para usar uma pasta personalizada, defina `REVERB_TTS_MODELS_DIR` antes de abrir o ComfyUI. Os seis workflows públicos (01–05 e 07) foram adaptados para `Auto / fp32`. Atualize os workflows e, se necessário, recrie o node carregador antigo. Nenhum modelo é baixado automaticamente.


## v0.3.6 — Deduplicated model selector / Seletor sem duplicatas

The `model` dropdown lists one `Auto / <variant>` entry for each detected precision. A location-specific entry is displayed **only if that variant is also installed in another distinct models directory**. The existing `Auto / fp32` workflow selections remain compatible. No weights or dependencies change.

O menu de modelos não mostra mais `Auto / fp32` e `Windows / fp32` quando ambos apontam para os mesmos pesos. Se houver outra instalação realmente distinta da mesma variante, essa segunda pasta poderá ser selecionada separadamente. Reinicie o ComfyUI e atualize o node após fazer o `git pull`.
