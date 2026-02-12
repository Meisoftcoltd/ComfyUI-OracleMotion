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

                // Add a visual hint or status widget
                const w = this.addWidget("text", "Timeline Info", "Listening to Brain...", (v) => {}, { disabled: true });

                // We hook into the execution to update the display
                // Real-time timeline editing requires a custom DOM overlay which is complex for V1.
                // For V1: This ensures the node registers correctly and shows status.

                this.onResize = function(size) {
                    size[0] = Math.max(size[0], 300);
                    size[1] = Math.max(size[1], 100);
                }

                return r;
            };
        }
    }
});
