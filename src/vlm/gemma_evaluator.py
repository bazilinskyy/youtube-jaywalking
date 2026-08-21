import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.vlm.alpamayo_detector import AlpamayoFullVideoDetector

GEMMA_PROMPT_TEMPLATE = """You are the final decision maker for a jaywalking detection system.

Alpamayo has already analyzed the video and produced the structured visual evidence and reasoning below.

Your task is to critically evaluate that evidence and reach the final classification.

Do not blindly copy Alpamayo's preliminary verdict.
Determine whether the evidence supports jaywalking or compliant behavior.

Consider:
- whether the pedestrian actually crosses the roadway
- pedestrian trajectory and movement
- presence or absence of a designated crossing
- traffic signals and right-of-way
- vehicle interaction
- whether Alpamayo made unsupported assumptions

Return ONLY valid JSON:

{{
  "verdict": "JAYWALKING" or "COMPLIANT",
  "reasoning": "brief explanation"
}}

ALPAMAYO ANALYSIS:
{ALPAMAYO_OUTPUT}"""


class AlpamayoGemmaEvaluator:
    """Alpamayo -> Gemma conclusion pipeline.
    Alpamayo performs all visual/temporal video understanding.
    Gemma critically evaluates Alpamayo's extracted text evidence to issue the final verdict.
    """

    def __init__(
        self,
        alpamayo_model: str = "qwen2.5vl:7b",
        gemma_model: str = "gemma:2b",
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.alpamayo_model = alpamayo_model
        self.gemma_model = gemma_model
        self.alpamayo_detector = AlpamayoFullVideoDetector(
            model_name=alpamayo_model,
            temperature=temperature,
            seed=seed,
        )

    def run_gemma_inference(self, prompt: str) -> str:
        """Executes Gemma inference via Ollama subprocess CLI."""
        res = subprocess.run(
            ["ollama", "run", self.gemma_model, prompt],
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    def parse_gemma_json(self, raw_text: str, fallback_pred: str) -> Dict[str, Any]:
        """Parses Gemma's JSON output safely."""
        text = raw_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
            verdict = str(parsed.get("verdict", "")).strip().lower()
            if "jaywalking" in verdict:
                verdict = "jaywalking"
            elif "compliant" in verdict:
                verdict = "compliant"
            else:
                verdict = fallback_pred
            reasoning = str(parsed.get("reasoning", text)).strip()
            return {"verdict": verdict, "reasoning": reasoning}
        except Exception:
            upper = text.upper()
            if "JAYWALKING" in upper and "COMPLIANT" not in upper:
                verdict = "jaywalking"
            elif "COMPLIANT" in upper and "JAYWALKING" not in upper:
                verdict = "compliant"
            else:
                verdict = fallback_pred
            return {"verdict": verdict, "reasoning": text}

    def predict(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        t0 = time.time()

        # Step 1: Run Alpamayo visual & temporal video analysis
        t_alp_0 = time.time()
        alpamayo_res = self.alpamayo_detector.predict(video_path)
        alpamayo_elapsed = round(time.time() - t_alp_0, 3)

        alpamayo_pred = alpamayo_res["prediction"]
        alpamayo_coc = alpamayo_res.get("chain_of_causation", "")

        # Construct full Alpamayo output block for Gemma
        alpamayo_output_block = "Preliminary Verdict: " + alpamayo_pred.upper() + "\nReasoning:\n" + alpamayo_coc

        # Step 2: Build exact Gemma input prompt
        gemma_prompt = GEMMA_PROMPT_TEMPLATE.format(ALPAMAYO_OUTPUT=alpamayo_output_block)

        # Step 3: Run Gemma inference
        t_gem_0 = time.time()
        gemma_raw = self.run_gemma_inference(gemma_prompt)
        gemma_elapsed = round(time.time() - t_gem_0, 3)

        # Step 4: Parse Gemma verdict
        gemma_parsed = self.parse_gemma_json(gemma_raw, fallback_pred=alpamayo_pred)
        gemma_pred = gemma_parsed["verdict"]

        # Step 5: Deterministic Arbitration — Gemma verdict is final
        final_pred = gemma_pred
        total_elapsed = round(time.time() - t0, 3)

        return {
            "prediction": final_pred,
            "confidence": "high",
            "reason": "Alpamayo->Gemma conclusion (" + gemma_pred + ")",
            "alpamayo_model": self.alpamayo_model,
            "gemma_model": self.gemma_model,
            "alpamayo_prediction": alpamayo_pred,
            "alpamayo_reasoning": alpamayo_coc,
            "gemma_prompt": gemma_prompt,
            "gemma_raw_response": gemma_raw,
            "gemma_prediction": gemma_pred,
            "gemma_reasoning": gemma_parsed["reasoning"],
            "alpamayo_elapsed_seconds": alpamayo_elapsed,
            "gemma_elapsed_seconds": gemma_elapsed,
            "elapsed_seconds": total_elapsed,
        }
