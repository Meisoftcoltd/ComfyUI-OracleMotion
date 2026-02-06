import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.OracleMotion.Studio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OracleDirector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the hidden 'user_edits' widget or create it if missing
                let userEditsWidget = this.widgets.find(w => w.name === "user_edits");
                if (!userEditsWidget) {
                    userEditsWidget = this.addWidget("string", "user_edits", "[]", (v) => {}, { hidden: true });
                }

                // Create the Visual Timeline Widget
                const timelineWidget = {
                    name: "OracleTimeline",
                    type: "ORACLE_TIMELINE",
                    value: [],
                    draw(ctx, node, widget_width, y, widget_height) {
                        // Custom drawing placeholder
                    },
                    computeSize(width) {
                        return [width, 400]; // Increased height for more columns
                    }
                };

                this.addCustomWidget(timelineWidget);

                // Create DOM element for the Timeline
                const div = document.createElement("div");
                div.style.background = "#1e1e1e";
                div.style.color = "#ddd";
                div.style.padding = "10px";
                div.style.fontFamily = "sans-serif";
                div.style.overflowY = "auto";
                div.style.height = "380px";
                div.style.border = "1px solid #444";
                div.style.borderRadius = "4px";

                // Table Header
                const header = document.createElement("div");
                header.innerHTML = `
                    <div style="display: grid; grid-template-columns: 40px 1fr 1fr 80px 80px; gap: 5px; font-weight: bold; border-bottom: 1px solid #555; padding-bottom: 5px; margin-bottom: 5px; font-size: 12px;">
                        <span>#</span>
                        <span>Prompt</span>
                        <span>Dialogue</span>
                        <span>Emotion</span>
                        <span>Voice</span>
                    </div>
                `;
                div.appendChild(header);

                // Container for rows
                const rowsContainer = document.createElement("div");
                div.appendChild(rowsContainer);

                // Add Row Button
                const addBtn = document.createElement("button");
                addBtn.textContent = "+ Add Scene";
                addBtn.style.marginTop = "10px";
                addBtn.style.background = "#333";
                addBtn.style.color = "white";
                addBtn.style.border = "none";
                addBtn.style.padding = "5px 10px";
                addBtn.style.cursor = "pointer";
                addBtn.onclick = () => {
                    this.state.push({
                        scene_id: this.state.length,
                        visual_prompt: "New Scene",
                        dialogue: "",
                        audio_emotion: "Neutral",
                        voice_name: "",
                        reference_path: ""
                    });
                    renderRows();
                    updateWidgetValue();
                };
                div.appendChild(addBtn);

                // Store state
                this.state = [];
                try {
                    if (userEditsWidget.value) {
                         this.state = JSON.parse(userEditsWidget.value);
                    }
                } catch (e) {
                    this.state = [];
                }

                const updateWidgetValue = () => {
                    userEditsWidget.value = JSON.stringify(this.state);
                };

                const renderRows = () => {
                    rowsContainer.innerHTML = "";
                    this.state.forEach((kf, index) => {
                        const row = document.createElement("div");
                        row.style.display = "grid";
                        row.style.gridTemplateColumns = "40px 1fr 1fr 80px 80px";
                        row.style.gap = "5px";
                        row.style.alignItems = "center";
                        row.style.marginBottom = "5px";
                        row.style.padding = "5px";
                        row.style.background = "#2a2a2a";
                        row.style.borderRadius = "3px";
                        row.style.fontSize = "11px";

                        // Scene ID
                        const idSpan = document.createElement("span");
                        idSpan.textContent = kf.scene_id !== undefined ? kf.scene_id : index;
                        row.appendChild(idSpan);

                        // Visual Prompt Input
                        const promptInp = document.createElement("input");
                        promptInp.type = "text";
                        promptInp.value = kf.visual_prompt || kf.prompt || "";
                        promptInp.placeholder = "Visuals...";
                        promptInp.style.width = "100%";
                        promptInp.style.background = "#111";
                        promptInp.style.color = "#ddd";
                        promptInp.style.border = "1px solid #444";
                        promptInp.onchange = (e) => {
                            kf.visual_prompt = e.target.value;
                            updateWidgetValue();
                        };
                        row.appendChild(promptInp);

                        // Dialogue Input
                        const diagInp = document.createElement("textarea");
                        diagInp.value = kf.dialogue || "";
                        diagInp.placeholder = "Dialogue...";
                        diagInp.style.width = "100%";
                        diagInp.style.height = "30px";
                        diagInp.style.background = "#111";
                        diagInp.style.color = "#ddd";
                        diagInp.style.border = "1px solid #444";
                        diagInp.style.resize = "none";
                        diagInp.onchange = (e) => {
                            kf.dialogue = e.target.value;
                            updateWidgetValue();
                        };
                        row.appendChild(diagInp);

                        // Emotion Input
                        const emoInp = document.createElement("input");
                        emoInp.type = "text";
                        emoInp.value = kf.audio_emotion || "";
                        emoInp.placeholder = "Emotion";
                        emoInp.style.width = "100%";
                        emoInp.style.background = "#111";
                        emoInp.style.color = "#ddd";
                        emoInp.style.border = "1px solid #444";
                        emoInp.onchange = (e) => {
                            kf.audio_emotion = e.target.value;
                            updateWidgetValue();
                        };
                        row.appendChild(emoInp);

                        // Voice Input & Ref Image combined for space?
                        // Or just Voice Input. Let's add Ref Image as a small icon or drag area.
                        const voiceContainer = document.createElement("div");
                        voiceContainer.style.display = "flex";
                        voiceContainer.style.flexDirection = "column";

                        const voiceInp = document.createElement("input");
                        voiceInp.type = "text";
                        voiceInp.value = kf.voice_name || "";
                        voiceInp.placeholder = "Voice";
                        voiceInp.style.width = "100%";
                        voiceInp.style.marginBottom = "2px";
                        voiceInp.style.background = "#111";
                        voiceInp.style.color = "#ddd";
                        voiceInp.style.border = "1px solid #444";
                        voiceInp.onchange = (e) => {
                            kf.voice_name = e.target.value;
                            updateWidgetValue();
                        };
                        voiceContainer.appendChild(voiceInp);

                        // Ref Image Area (Small)
                        const refArea = document.createElement("div");
                        refArea.style.height = "20px";
                        refArea.style.background = (kf.reference_path || kf.path) ? "#4a4" : "#333";
                        refArea.style.border = "1px dashed #666";
                        refArea.textContent = (kf.reference_path || kf.path) ? "IMG OK" : "Drop Img";
                        refArea.style.textAlign = "center";
                        refArea.style.cursor = "default";
                        refArea.title = kf.reference_path || kf.path || "Drag image here";

                        refArea.ondragover = (e) => { e.preventDefault(); refArea.style.borderColor = "#fff"; };
                        refArea.ondragleave = (e) => { e.preventDefault(); refArea.style.borderColor = "#666"; };
                        refArea.ondrop = (e) => {
                            e.preventDefault();
                            const items = e.dataTransfer.items;
                            for (let i = 0; i < items.length; i++) {
                                if (items[i].kind === 'string') {
                                     items[i].getAsString((s) => {
                                         if (s.match(/\.(png|jpg|jpeg|webp)$/i)) {
                                             kf.reference_path = s;
                                             refArea.style.background = "#4a4";
                                             refArea.textContent = "IMG OK";
                                             updateWidgetValue();
                                         }
                                     });
                                }
                            }
                        };
                        voiceContainer.appendChild(refArea);

                        row.appendChild(voiceContainer);

                        rowsContainer.appendChild(row);
                    });
                };

                // Initial Render
                renderRows();
                timelineWidget.element = div;

                return r;
            };
        }
    }
});
