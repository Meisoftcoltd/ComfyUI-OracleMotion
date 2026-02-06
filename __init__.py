from .nodes import OracleBrainAPI, OracleBrainLocal, OracleDirector, OracleVisualizer, OracleEngine, OracleEditor

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "OracleBrainAPI": OracleBrainAPI,
    "OracleBrainLocal": OracleBrainLocal,
    "OracleDirector": OracleDirector,
    "OracleVisualizer": OracleVisualizer,
    "OracleEngine": OracleEngine,
    "OracleEditor": OracleEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OracleBrainAPI": "🧠 Oracle Brain (Cloud/Ollama API)",
    "OracleBrainLocal": "🧠 Oracle Brain (Local Native GGUF)",
    "OracleDirector": "🎬 Oracle Director (The UI Controller)",
    "OracleVisualizer": "🟠 Oracle Visualizer (Art Director)",
    "OracleEngine": "🔴 Oracle Engine (The Agnostic Animator)",
    "OracleEditor": "✂️ Oracle Editor (Post-Production)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
