import * as vscode from "vscode";

let statusBarItem: vscode.StatusBarItem;

export function initStatusBar(context: vscode.ExtensionContext): vscode.StatusBarItem {
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "codelens.scorePrompt";
  statusBarItem.text = "$(sparkle) CodeLens";
  statusBarItem.tooltip = "Select a prompt and click to estimate AI code-quality";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);
  return statusBarItem;
}

/**
 * Score bands are placeholders until the real ACQP distribution on the
 * mined+labeled corpus tells us what "good"/"fair"/"weak" actually mean
 * for this dataset. Update these three thresholds once results/metrics/
 * has real percentiles -- don't present these as validated cutoffs in
 * the report as-is.
 */
const SCORE_BANDS: [number, string, string][] = [
  [0.7, "$(check)", "Strong prompt"],
  [0.4, "$(warning)", "Could be more specific"],
  [0, "$(error)", "Likely to produce weak code"],
];

export function showScore(score: number) {
  const [, icon, label] = SCORE_BANDS.find(([threshold]) => score >= threshold)!;
  statusBarItem.text = `${icon} CodeLens: ${score.toFixed(2)}`;
  statusBarItem.tooltip = `${label} (estimated quality: ${score.toFixed(2)})`;
}

export function showError(message: string) {
  statusBarItem.text = "$(alert) CodeLens: error";
  statusBarItem.tooltip = message;
}

export function showLoading() {
  statusBarItem.text = "$(sync~spin) CodeLens: scoring...";
  statusBarItem.tooltip = "Contacting the scoring server";
}

export function resetStatusBar() {
  statusBarItem.text = "$(sparkle) CodeLens";
  statusBarItem.tooltip = "Select a prompt and click to estimate AI code-quality";
}
