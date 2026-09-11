import * as vscode from "vscode";
import { scorePrompt, SCORE_SERVER_URL } from "./scoring/scorer";
import { initStatusBar, showScore, showError, showLoading, resetStatusBar } from "./ui/statusBar";
import { showScorePanel } from "./ui/scorePanel";

export function activate(context: vscode.ExtensionContext) {
  console.log("CodeLens extension activated");
  initStatusBar(context);

  const disposable = vscode.commands.registerCommand("codelens.scorePrompt", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage("CodeLens: open an editor and select a prompt to score.");
      return;
    }

    const selection = editor.selection;
    const prompt = editor.document.getText(selection).trim();
    if (!prompt) {
      vscode.window.showWarningMessage("CodeLens: select some text (your coding prompt) first.");
      return;
    }

    await scoreAndShow(context, prompt);
  });

  context.subscriptions.push(disposable);
}

async function scoreAndShow(context: vscode.ExtensionContext, prompt: string) {
  showLoading();
  try {
    const score = await scorePrompt(prompt);
    showScore(score);
    showScorePanel(context, prompt, score, defaultSuggestions(score, prompt));
  } catch (err) {
    const message =
      `CodeLens: could not reach scoring server at ${SCORE_SERVER_URL}. ` +
      `Is it running? (uvicorn src.models.student.serve:app --port 8000)`;
    showError(message);
    vscode.window.showErrorMessage(message);
    console.error(err);
  }
}

/**
 * Prompt-content heuristics. Each suggestion fires ONLY if the prompt is
 * actually missing the corresponding signal -- not based on the score.
 * The score is the model's output; these are independent textual checks.
 */
function defaultSuggestions(score: number, prompt: string): string[] {
  const out: string[] = [];
  const p = prompt.toLowerCase();

  // 1. Input/output types specified?
  const hasTypes =
    /(:|->|→|int\b|str\b|float\b|bool\b|list\b|dict\b|tuple\b|returns?\b|takes\b|input\b|output\b)/.test(p);
  if (!hasTypes) {
    out.push("Consider specifying expected input and output types.");
  }

  // 2. Edge cases mentioned?
  const hasEdgeCases =
    /\b(edge|invalid|empty|null|none|error|exception|handle|negative|zero|punctuation|case|whitespace|overflow|boundary)\b/.test(p);
  if (!hasEdgeCases) {
    out.push("Mention edge cases the generated code should handle.");
  }

  // 3. Explicit error behavior?
  const hasErrorBehavior =
    /\b(raise|error|exception|invalid|fail|return none|return -1|raise valueerror|raise typeerror|throw)\b/.test(p);
  if (!hasErrorBehavior) {
    out.push("Say how the code should behave on invalid or empty input.");
  }

  // 4. Example provided?
  const hasExample = /\b(example|e\.g\.|for instance|such as|sample)\b/.test(p);
  if (!hasExample) {
    out.push("Include an example input/output to anchor the expected behavior.");
  }

  // 5. Prompt too short?
  if (prompt.length < 60) {
    out.push("Add more context about the intended use case.");
  }

  // Cap at 4 to keep the panel readable
  return out.length ? out.slice(0, 4) : ["Prompt looks well-specified."];
}

export function deactivate() {}