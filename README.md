# ComfyUI-Reverb-TTS v0.3.0 — somente arquivos locais

Integração experimental com o [Reverb-TTS](https://github.com/software-mansion-labs/Reverb-TTS), com carregamento manual dos modelos, clonagem de voz, vozes predefinidas, perfis `.npz` e WAV PCM.

**NÃO baixa modelos automaticamente.** O carregador valida os arquivos antes de tentar inferência. A API oficial suporta `TTS.from_pretrained(pasta_local, codec=pasta_local)`.

## 1. Pastas dos modelos (NÃO é diffusion_models)

Crie `C:\AI\ComfyUI-Data\models\reverb_tts\` e mantenha a estrutura:

```text
C:\AI\ComfyUI-Data\models\reverb_tts\
  Reverb-TTS\
    model\
      tokenizer.json
      tokenizer_config.json         (recomendado)
      chat_template.jinja           (recomendado)
      safetensors\
        fp32\
          config.json
          model.safetensors
        fp16\                 (opcional)
          config.json
          model.safetensors
        int8\                 (opcional)
          config.json
          model.safetensors
        int4\                 (opcional)
          config.json
          model.safetensors
    voices\                  (opcional; obrigatoria para vozes predefinidas offline)
      pt_male_standard.json
      ...
  higgs-audio-v2-tokenizer\
    config.json
    model.safetensors
    preprocessor_config.json        (recomendado)
```

Os caminhos são validados. **Não basta copiar o `.safetensors` sozinho**: cada variante também precisa do seu `config.json`, e o `tokenizer.json` fica em `Reverb-TTS/model/`. O codec é um segundo modelo, distinto do TTS.

### Downloads manuais

1. Escolha FP32 em https://huggingface.co/software-mansion/Reverb-TTS/tree/main/model/safetensors/fp32 ; baixe **`config.json` e `model.safetensors`** para `Reverb-TTS/model/safetensors/fp32/`.
2. Baixe `tokenizer.json`, `tokenizer_config.json` e `chat_template.jinja` de https://huggingface.co/software-mansion/Reverb-TTS/tree/main/model e coloque em `Reverb-TTS/model/`.
3. Baixe `config.json`, `model.safetensors`, `preprocessor_config.json` de https://huggingface.co/eustlb/higgs-audio-v2-tokenizer/tree/main para `higgs-audio-v2-tokenizer/`.
4. Se for usar presets, baixe os `.json` desejados de https://huggingface.co/software-mansion/Reverb-TTS/tree/main/voices para `Reverb-TTS/voices/`.
5. Para FP16 e INT8, baixe ambos os arquivos da variante respectiva em `model/safetensors/fp16/` ou `/int8/`. Troque `variant` no loader; não precisa refazer o workflow.

As pastas são deliberadamente fora de `diffusion_models` (Reverb é TTS, não modelo de difusão de imagem). Se a pasta personalizada acima ainda não existir, há fallback para `ComfyUI/models/reverb_tts`; **preferência sempre pelo caminho `C:\AI\ComfyUI-Data\models\reverb_tts`**.

**Atenção ao modo totalmente offline:** instalar o pacote Python (`reverb-tts`) é uma operação separada, e o carregamento pode requerer módulos Python já presentes; para modelos, ambos os caminhos passados ao carregador são locais e presets requerem JSONs locais. Nenhum peso é incluído neste pacote.

## 2. Erro identificado no seu log

O erro atual é `ModuleNotFoundError: No module named 'reverb_tts'`. Instale as dependências **no Python embarcado do ComfyUI**. Antes, confira alterações de dependências, pois você já tem vários custom nodes:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install --dry-run -r C:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Reverb-TTS\requirements.txt
```

Se aceitáveis:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -m pip install -r C:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Reverb-TTS\requirements.txt
```

Verifique:

```bat
C:\AI\ComfyUI\python_embeded\python.exe -c "from reverb_tts import TTS, Voice; print('Reverb instalado')"
```

Instalar os pesos sem instalar a biblioteca Python **não elimina** esse erro.

## 3. Instalar o node e workflows

Extraia/atualize `ComfyUI-Reverb-TTS` em `C:\AI\ComfyUI\ComfyUI\custom_nodes\ComfyUI-Reverb-TTS\` (substitua `nodes.py`, `README.md` etc). Reinicie o ComfyUI. Arraste os `.json` para o editor ou copie-os à sua pasta normal de workflows.

- `01-Clonar-e-Gerar-FP32.json`: gravar e narrar com a voz clonada.
- `02-Voz-Predefinida-FP32.json`: usar JSON local de preset.
- `03-Criar-e-Salvar-Voz.json`: `.npz` reutilizável de voz, sem gerar narração.
- `04-Carregar-Voz-Salva-e-Gerar-FP32.json`: `.npz` salvo → WAV.
- `05-Clonar-Salvar-e-Gerar-FP32.json`: clonagem + salva `.npz` + narração.
- `06-Seu-Workflow-Corrigido.json`: seu workflow anexado, com `speed` e `steps` corrigidos no JSON.

## 4. Velocidade e precisão

`speed` vai de **0.5 a 2.0** (passo 0.05). O parâmetro `steps` é independente (padrão 8). O erro `Value 8.0 bigger than max of 2.0: speed` no seu log surgiu de um valor `8` deslocado no JSON antigo: `widgets_values` tinha um item extra de controle da seed. Os JSONs deste pacote foram corrigidos; não use o workflow antigo sem migrá-lo. Para comparar precisões, selecione `fp32`, `fp16`, `int8` ou `int4` no loader depois de instalar as respectivas pastas. `force_fp32_compute` controla o dtype de execução, não a escolha dos pesos.

## 5. Limitações

Testado: compilação Python, inspeção estática do carregamento local e integridade dos workflows. **Ainda não testado com os pesos reais na sua RTX 4060 Ti**. O usuário deve instalar as dependências separadamente. Pesos do modelo: **CC BY-NC 4.0 (uso não comercial com atribuição)**. Código oficial Apache 2.0. Use vozes somente com autorização do locutor.


## 6. Biblioteca de vozes NPZ (v0.3.0)

Os novos perfis sao salvos diretamente em:

```text
C:\AI\ComfyUI-Data\models\reverb_tts\voices_custom\
  minha_voz.npz
  narrador_01.npz
```

- `Reverb | Save Voice (.npz)` cria um perfil usando a voz obtida do audio de referencia. `overwrite=false` evita sobrescrever um perfil existente.
- **`Reverb | Voice Library (NPZ Selector)`** mostra a lista de arquivos `.npz` da biblioteca e entrega uma saída `REVERB_VOICE` para o gerador. **Depois de salvar uma nova voz, atualize a pagina do ComfyUI (F5) ou recrie o node seletor para atualizar a lista**. A busca nao precisa de um carregamento de modelo TTS.
- `Reverb | Load Voice by Name (.npz)` continua disponivel para compatibilidade com workflows antigos; digite apenas o nome, sem extensao.
- Para trazer perfis da pasta antiga `ComfyUI/output/Reverb/voices/`, execute opcionalmente `migrar_vozes_antigas.bat`. Os arquivos antigos nao sao apagados; arquivos existentes no destino nao sao substituidos.
- As vozes `.npz` sao especificas da API `reverb_tts.Voice`; arquivos de outros TTS nao sao compativeis automaticamente. O formato de serializacao e carregamento precisa ser validado com a versao instalada da biblioteca.

### Workflows

`03-Criar-e-Salvar-Voz.json` e `05-Clonar-Salvar-e-Gerar-FP32.json` agora salvam na biblioteca central sem mudancas nos seus links. `04-Carregar-Voz-Salva-e-Gerar-FP32.json` foi readaptado para usar o menu de selecao. Novo `07-Biblioteca-de-Vozes-Seletor.json` e um modelo minimo com seletor + geracao + WAV. Os workflows 01, 02 e 06 continuam operando independentemente da biblioteca.

**Limites de validacao:** houve verificacao sintatica e estrutural local; ainda nao foi executada inferencia real nem o carregamento de um perfil NPZ real nesta maquina. Instale a biblioteca `reverb-tts` no Python usado pelo ComfyUI para testar.


## v0.3.1 — Preview integrado no Reverb | Save WAV

O node `Reverb | Save WAV` agora mostra um player de áudio diretamente no canvas após executar, usando o padrão de UI de áudio do ComfyUI. Mantém WAV `PCM_24`, `PCM_16` ou `FLOAT` e a saída `saved_path` (STRING) para preservar conexões existentes. O player aparece depois de uma nova execução; não é necessário modificar nenhum workflow existente.
