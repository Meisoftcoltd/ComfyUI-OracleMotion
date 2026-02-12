from .nodes import OracleBrainAPI, OracleBrainLocal, OracleDirector, OracleVisualizer, OracleEngine, OraclePostProduction, OracleVoiceKokoro, OracleVoiceInjector

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "OracleBrainAPI": OracleBrainAPI,
    "OracleBrainLocal": OracleBrainLocal,
    "OracleVoiceKokoro": OracleVoiceKokoro,
    "OracleVoiceInjector": OracleVoiceInjector,
    "OracleDirector": OracleDirector,
    "OracleVisualizer": OracleVisualizer,
    "OracleEngine": OracleEngine,
    "OraclePostProduction": OraclePostProduction
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OracleBrainLocal": "🧠 Oracle Brain (Local Director)",
    "OracleBrainAPI": "🧠 Oracle Brain (Cloud API)",
    "OracleVoiceKokoro": "🎙️ Oracle Voice (Kokoro Local)",
    "OracleVoiceInjector": "🎙️ Oracle Voice (External/Qwen3 Bridge)",
    "OracleDirector": "🪬 Oracle Director (Timeline Studio)",
    "OracleVisualizer": "🎨 Oracle Visualizer (Art Gen)",
    "OracleEngine": "🎬 Oracle Engine (Agnostic Animator)",
    "OraclePostProduction": "✂️ Oracle Post-Production (Viral Editor)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
