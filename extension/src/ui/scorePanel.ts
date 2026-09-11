import * as vscode from "vscode";

let currentPanel: vscode.WebviewPanel | undefined = undefined;

export function showScorePanel(
  context: vscode.ExtensionContext,
  prompt: string,
  score: number,
  suggestions: string[] = []
) {
  if (currentPanel) {
    // Already open — reveal it without stealing focus from the editor.
    currentPanel.reveal(vscode.ViewColumn.Beside, true);
  } else {
    currentPanel = vscode.window.createWebviewPanel(
      "codelensScore",
      "CodeLens",
      vscode.ViewColumn.Beside,     // always split to the right
      { enableScripts: true, retainContextWhenHidden: true }
    );

    currentPanel.onDidDispose(
      () => {
        currentPanel = undefined;
      },
      null,
      context.subscriptions
    );

    currentPanel.webview.onDidReceiveMessage(
      (msg) => {
        if (msg.command === "applySuggestions") {
          vscode.window.showInformationMessage("Suggestion applied (demo).");
        }
      },
      null,
      context.subscriptions
    );
  }

  currentPanel.webview.html = getHtml(prompt, score, suggestions);
}

function getHtml(prompt: string, score: number, suggestions: string[]): string {
  // ACQP is on a 0-10 scale. Display it as X.X / 10.
  // The gauge arc is still drawn using pct (0-100) for the visual fill.
  const clamped = Math.max(0, Math.min(10, score));
  const pct = clamped * 10;
  const label = clamped >= 6.5 ? "Good" : clamped >= 4.0 ? "Fair" : "Weak";
  const color = clamped >= 6.5 ? "#4caf50" : clamped >= 4.0 ? "#ff9800" : "#f44336";
  const dash = (pct / 100) * 251;  // circumference of r=40 circle ≈ 251

  const suggestionsHtml = suggestions.length
    ? `<ul>${suggestions.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`
    : `<p><em>No specific suggestions for this prompt.</em></p>`;

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground);
         background: var(--vscode-sideBar-background); padding: 20px; }
  h2 { font-size: 14px; font-weight: 600; margin-bottom: 18px; }
  .gauge-wrap { display: flex; flex-direction: column; align-items: center; margin: 20px 0 24px; }
  svg { transform: rotate(-90deg); }
  .pct { font-size: 40px; font-weight: 700; margin-top: -90px; }
  .pct-total { font-size: 14px; color: var(--vscode-descriptionForeground); }
  .label { margin-top: 30px; font-size: 16px; font-weight: 600; color: ${color}; }
  h3 { font-size: 13px; margin-top: 24px; margin-bottom: 8px; }
  ul { padding-left: 18px; line-height: 1.6; }
  button { margin-top: 16px; padding: 6px 14px; font-size: 13px;
           background: var(--vscode-button-background); color: var(--vscode-button-foreground);
           border: none; border-radius: 3px; cursor: pointer; }
  button:hover { background: var(--vscode-button-hoverBackground); }
  .prompt-preview { font-family: monospace; font-size: 12px;
                    color: var(--vscode-descriptionForeground);
                    border-left: 2px solid var(--vscode-panel-border);
                    padding-left: 8px; margin-bottom: 16px;
                    white-space: pre-wrap; }
</style>
</head>
<body>
  <h2>Prompt Quality</h2>
  <div class="prompt-preview">${escapeHtml(prompt.slice(0, 300))}</div>

  <div class="gauge-wrap">
    <svg width="100" height="100" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="40" fill="none"
              stroke="var(--vscode-panel-border)" stroke-width="8"/>
      <circle cx="50" cy="50" r="40" fill="none"
              stroke="${color}" stroke-width="8" stroke-linecap="round"
              stroke-dasharray="${dash} 251"/>
    </svg>
    <div class="pct">${clamped.toFixed(1)}<span class="pct-total"> /10</span></div>
    <div class="label">${label}</div>
  </div>

  <h3>Suggestions</h3>
  ${suggestionsHtml}
  <button onclick="applySuggestions()">Apply Suggestions</button>

  <script>
    const vscode = acquireVsCodeApi();
    function applySuggestions() {
      vscode.postMessage({ command: "applySuggestions" });
    }
  </script>
</body>
</html>`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!)
  );
}