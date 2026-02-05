from .nodes import OracleBrain, OracleVisualizer, OracleEngine, OracleEditor

NODE_CLASS_MAPPINGS = {
    "OracleBrain": OracleBrain,
    "OracleVisualizer": OracleVisualizer,
    "OracleEngine": OracleEngine,
    "OracleEditor": OracleEditor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OracleBrain": "Oracle Brain (Scriptwriter)",
    "OracleVisualizer": "Oracle Visualizer (Art Director)",
    "OracleEngine": "Oracle Engine (Wan 2.1 Animator)",
    "OracleEditor": "Oracle Editor (Post-Production)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
