# ComfyUI-OracleMotion V2

Este proyecto implementa una arquitectura avanzada de "Serial Batch Processing" para la generación de historias animadas dentro de ComfyUI. Está diseñado para operar eficientemente en entornos con recursos limitados (como una RTX 3090) mediante la gestión estricta de VRAM y el uso de modelos de última generación.

El proyecto original y la inspiración provienen de [ToonComposer de TencentARC](https://github.com/TencentARC/ToonComposer), adaptado y evolucionado aquí para flujos de trabajo locales y modulares.

## 🚀 Filosofía del Proyecto (V2)

- **Serial Batch Processing:** En lugar de cargar todos los modelos a la vez, el sistema procesa cada etapa secuencialmente y limpia agresivamente la VRAM entre pasos.
- **Dependencias Limpias:** Se evita el "infierno de dependencias" utilizando el ecosistema nativo de HuggingFace (`diffusers`, `transformers`) y evitando reinstalaciones innecesarias de PyTorch.
- **Modularidad:** Cuatro nodos especializados (Brain, Visualizer, Engine, Editor) que pueden usarse juntos o por separado.

## 📦 Instalación

1.  Clona este repositorio en tu carpeta `custom_nodes` de ComfyUI:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/tu-usuario/ComfyUI-OracleMotion.git
    ```

2.  Instala las dependencias. El sistema intentará instalarlas automáticamente al iniciar, pero puedes hacerlo manualmente:
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: No se instalan `torch` ni `torchvision` para no romper tu entorno existente de ComfyUI.*

## 🛠️ Los Nodos

### 1. OracleBrain (El Guionista)
Actúa como el director creativo.
-   **Entrada:** Narrativa (texto) y opcionalmente Audio (voz).
-   **Proceso:**
    -   Si hay audio, usa `faster-whisper` para transcribirlo.
    -   Usa un LLM (vía OpenAI API, compatible con **Ollama** o **LMStudio** local) para convertir la idea en un Storyboard estructurado en JSON.
-   **Salida:** JSON con la descripción visual, acción y referencias de cada escena.

### 2. OracleVisualizer (El Director de Arte)
Genera los "Keyframes" (imágenes estáticas) para cada escena.
-   **Tecnología:** SDXL + IP-Adapter.
-   **Características:**
    -   Permite inyectar referencias visuales específicas (objetos, personajes) usando `IP-Adapter Plus` para SDXL.
    -   Soporta un `global_style_prompt` para mantener la coherencia artística.
    -   Gestiona la carga y descarga del modelo para ahorrar memoria.

### 3. OracleEngine (El Animador)
Da vida a las imágenes estáticas.
-   **Tecnología:** Wan 2.1 (Image-to-Video).
-   **Características:**
    -   Carga el modelo `Wan-AI/Wan2.1-I2V-1.3B-Diffusers` (con `trust_remote_code=True`).
    -   Genera clips de video de alta calidad a partir de los keyframes.
    -   Limpia la VRAM después de cada video generado.

### 4. OracleEditor (Post-Producción)
Ensambla el resultado final.
-   **Tecnología:** MoviePy.
-   **Proceso:** Une todos los clips de video generados en un único archivo MP4 final.

## 📋 Requisitos del Sistema
-   Python 3.10+ (Probado en 3.12)
-   ComfyUI actualizado.
-   GPU con al menos 24GB VRAM recomendada (RTX 3090/4090) para el flujo completo con Wan 2.1, aunque el procesamiento serial permite flexibilidad.

## 🔗 Créditos y Referencias
-   Basado en ideas de [TencentARC/ToonComposer](https://github.com/TencentARC/ToonComposer).
-   Utiliza [Diffusers](https://github.com/huggingface/diffusers) de HuggingFace.
-   Modelo de Video: [Wan 2.1](https://github.com/Wan-AI/Wan2.1).
