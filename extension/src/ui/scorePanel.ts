import * as vscode from "vscode";

let currentPanel: vscode.WebviewPanel | undefined = undefined;

export function showScorePanel(
  context: vscode.ExtensionContext,
  prompt: string,
  score: number,
  suggestions: string[] = []
) {
  const columnToShowIn = vscode.window.activeTextEditor
    ? vscode.window.activeTextEditor.viewColumn
    : undefined;

  if (currentPanel) {
    currentPanel.reveal(columnToShowIn);
  } else {
    currentPanel = vscode.window.createWebviewPanel(
      "codelensScore",
      "CodeLens",
      columnToShowIn ?? vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );

    currentPanel.onDidDispose(
      () => {
        currentPanel = undefined;
      },
      null,
      context.subscriptions
    );
  }

  currentPanel.webview.html = getHtml(prompt, score, suggestions);
}

function getHtml(prompt: string, score: number, suggestions: string[]): string {
  // Map the raw ACQP score (0-10) to a 0-100 display value.
  const pct = Math.max(0, Math.min(100, score * 10));
  const label = pct >= 70 ? "Good" : pct >= 40 ? "Fair" : "Weak";
  const color = pct >= 70 ? "#4caf50" : pct >= 40 ? "#ff9800" : "#f44336";
  const dash = (pct / 100) * 251;  // circumference of r=40 circle ≈ 251

  const suggestionsHtml = suggestions.length
    ? `<ul>${suggestions.map((s) => `<li>${s}</li>`).join("")}</ul>`
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
    <div class="pct">${Math.round(pct)}<span class="pct-total"> /100</span></div>
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