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

function defaultSuggestions(score: number, prompt: string): string[] {
  const out: string[] = [];
  if (score < 5) {
    out.push("Consider specifying expected input and output types.");
    out.push("Mention edge cases the generated code should handle.");
  }
  if (prompt.length < 60) {
    out.push("Try adding more context about the intended use case.");
  }
  if (!/\b(test|example|input|output|return)\b/i.test(prompt)) {
    out.push("State what the function should return and give an example input.");
  }
  if (!/\b(error|invalid|edge|empty|null|negative)\b/i.test(prompt)) {
    out.push("Say how the code should behave on invalid or empty input.");
  }
  return out.length ? out.slice(0, 4) : ["Prompt looks reasonably specified."];
}

export function deactivate() {}