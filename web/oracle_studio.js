import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI.OracleMotion",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OracleDirector") {
            // Enhance the OracleDirector node
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Simple visual feedback for V1
                // (Advanced timeline editing happens via the JSON input in this version)
                const w = this.addWidget("text", "Status", "Studio Ready", (v) => {}, { disabled: true });

                this.onResize = function(size) {
                    size[0] = Math.max(size[0], 300);
                    size[1] = Math.max(size[1], 100);
                }

                return r;
            };
        }
    }
});
