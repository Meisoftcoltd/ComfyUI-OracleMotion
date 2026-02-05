import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.OracleMotion.Studio",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OracleDirector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Find the hidden 'user_edits' widget or create it if missing (it should be defined in python)
                let userEditsWidget = this.widgets.find(w => w.name === "user_edits");
                if (!userEditsWidget) {
                    // Fallback if not found, though python side should define it
                    userEditsWidget = this.addWidget("string", "user_edits", "[]", (v) => {}, { hidden: true });
                }

                // Create the Visual Timeline Widget
                const timelineWidget = {
                    name: "OracleTimeline",
                    type: "ORACLE_TIMELINE",
                    value: [],
                    draw(ctx, node, widget_width, y, widget_height) {
                        // Custom drawing could go here, but we will use DOM overlay mostly
                        // This is just a placeholder to reserve space if needed in canvas
                    },
                    computeSize(width) {
                        return [width, 300]; // Height for the timeline area
                    }
                };

                // Add the widget
                this.addCustomWidget(timelineWidget);

                // Create DOM element for the Timeline
                const div = document.createElement("div");
                div.style.background = "#1e1e1e";
                div.style.color = "#ddd";
                div.style.padding = "10px";
                div.style.fontFamily = "sans-serif";
                div.style.overflowY = "auto";
                div.style.height = "280px";
                div.style.border = "1px solid #444";
                div.style.borderRadius = "4px";

                // Table Header
                const header = document.createElement("div");
                header.innerHTML = `
                    <div style="display: grid; grid-template-columns: 50px 1fr 100px; gap: 10px; font-weight: bold; border-bottom: 1px solid #555; padding-bottom: 5px; margin-bottom: 5px;">
                        <span>Frame</span>
                        <span>Prompt</span>
                        <span>Reference</span>
                    </div>
                `;
                div.appendChild(header);

                // Container for rows
                const rowsContainer = document.createElement("div");
                div.appendChild(rowsContainer);

                // Add Row Button
                const addBtn = document.createElement("button");
                addBtn.textContent = "+ Add Keyframe";
                addBtn.style.marginTop = "10px";
                addBtn.style.background = "#333";
                addBtn.style.color = "white";
                addBtn.style.border = "none";
                addBtn.style.padding = "5px 10px";
                addBtn.style.cursor = "pointer";
                addBtn.onclick = () => {
                    this.state.push({ frame: (this.state.length * 20), prompt: "New Scene", path: "" });
                    renderRows();
                    updateWidgetValue();
                };
                div.appendChild(addBtn);

                // Store state on the node instance
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
                        row.style.gridTemplateColumns = "50px 1fr 100px";
                        row.style.gap = "10px";
                        row.style.alignItems = "center";
                        row.style.marginBottom = "5px";
                        row.style.padding = "5px";
                        row.style.background = "#2a2a2a";
                        row.style.borderRadius = "3px";

                        // Frame Input
                        const frameInp = document.createElement("input");
                        frameInp.type = "number";
                        frameInp.value = kf.frame;
                        frameInp.style.width = "100%";
                        frameInp.style.background = "#111";
                        frameInp.style.color = "#ddd";
                        frameInp.style.border = "1px solid #444";
                        frameInp.onchange = (e) => {
                            kf.frame = parseInt(e.target.value);
                            updateWidgetValue();
                        };
                        row.appendChild(frameInp);

                        // Prompt Input
                        const promptInp = document.createElement("input");
                        promptInp.type = "text";
                        promptInp.value = kf.prompt;
                        promptInp.style.width = "100%";
                        promptInp.style.background = "#111";
                        promptInp.style.color = "#ddd";
                        promptInp.style.border = "1px solid #444";
                        promptInp.onchange = (e) => {
                            kf.prompt = e.target.value;
                            updateWidgetValue();
                        };
                        row.appendChild(promptInp);

                        // Reference Image Area (Drop Target)
                        const refArea = document.createElement("div");
                        refArea.style.width = "100px";
                        refArea.style.height = "50px";
                        refArea.style.background = kf.path ? `url("/view?filename=${encodeURIComponent(kf.path.split(/[/\\]/).pop())}") center/cover` : "#333";
                        refArea.style.border = "1px dashed #666";
                        refArea.style.fontSize = "10px";
                        refArea.style.display = "flex";
                        refArea.style.alignItems = "center";
                        refArea.style.justifyContent = "center";
                        refArea.style.overflow = "hidden";
                        refArea.textContent = kf.path ? "" : "Drop Img";
                        refArea.title = kf.path || "Drag image here";

                        // Drag & Drop Handlers
                        refArea.ondragover = (e) => {
                             e.preventDefault();
                             refArea.style.borderColor = "#fff";
                        };
                        refArea.ondragleave = (e) => {
                             e.preventDefault();
                             refArea.style.borderColor = "#666";
                        };
                        refArea.ondrop = (e) => {
                            e.preventDefault();
                            refArea.style.borderColor = "#666";

                            // ComfyUI usually puts the image info in dataTransfer
                            // We look for typical ComfyUI image drag data
                            // The gallery images are usually dragged as URLs or standard text

                            // Trying to handle various drop types from Comfy Gallery
                            // Usually drag data contains 'text/plain' with filename or similar

                            // Simplified handler: assume dropping from Comfy Gallery or OS
                            // If dropping from Comfy Gallery, it often passes an element we can inspect,
                            // or we can parse the src.

                            // This is a bit tricky without exact API, but we'll try to get the filename.

                            // Hack: Inspect the drag data
                            const items = e.dataTransfer.items;
                            for (let i = 0; i < items.length; i++) {
                                if (items[i].kind === 'string') {
                                     items[i].getAsString((s) => {
                                         // If it looks like a filename
                                         if (s.match(/\.(png|jpg|jpeg|webp)$/i)) {
                                             kf.path = s;
                                             // Update UI
                                             refArea.style.background = `url("/view?filename=${encodeURIComponent(kf.path)}") center/cover`;
                                             refArea.textContent = "";
                                             updateWidgetValue();
                                         }
                                     });
                                }
                            }

                            // Also check for files dropped from OS
                            if (e.dataTransfer.files.length > 0) {
                                // Uploading logic would be needed here, but let's assume usage of existing internal images for now
                                // or just show alert that only Gallery images are supported for path referencing in this simple version
                                alert("Please drop images from the ComfyUI History/Gallery.");
                            }
                        };

                        row.appendChild(refArea);

                        rowsContainer.appendChild(row);
                    });
                };

                // Initial Render
                renderRows();

                // Add to DOM
                // We need to wait for the node element to be created in the DOM
                // ComfyUI v1 extension API:
                // widget.element is often created by app, but here we want to inject our div

                // Override the onResize or similar to append our div?
                // Better approach: use `element` property of the widget if supported or append to node.content

                // Using standard approach for custom DOM widgets in Comfy
                timelineWidget.element = div;

                return r;
            };
        }
    }
});
