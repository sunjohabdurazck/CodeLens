import * as vscode from "vscode";
import axios from "axios";

const SCORE_SERVER_URL = "http://localhost:8000/score";

export function activate(context: vscode.ExtensionContext) {
  console.log("CodeLens extension activated");

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

    await scoreAndShow(prompt);
  });

  context.subscriptions.push(disposable);
}

async function scoreAndShow(prompt: string) {
  try {
    const response = await axios.post(SCORE_SERVER_URL, { prompt }, { timeout: 5000 });
    const score = response.data.score as number;
    vscode.window.showInformationMessage(`CodeLens quality estimate: ${score.toFixed(2)}`);
  } catch (err) {
    vscode.window.showErrorMessage(
      `CodeLens: could not reach scoring server at ${SCORE_SERVER_URL}. ` +
        `Is it running? (uvicorn src.models.student.serve:app --port 8000)`
    );
    console.error(err);
  }
}

export function deactivate() {}
