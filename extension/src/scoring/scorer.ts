import axios from "axios";

const SCORE_SERVER_URL = "http://localhost:8000/score";

export interface ScoreResult {
  score: number;
}

export async function scorePrompt(prompt: string, timeoutMs = 5000): Promise<number> {
  const response = await axios.post<ScoreResult>(
    SCORE_SERVER_URL,
    { prompt },
    { timeout: timeoutMs }
  );
  return response.data.score;
}

export { SCORE_SERVER_URL };
