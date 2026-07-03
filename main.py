"""
SkinCoach Web — простая и мощная версия.
Использует OpenRouter vision + reasoner модели.
"""
import base64
import json
import logging
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger("skincoach.web")

# ─── Config ────────────────────────────────────────────────────────────────
OR_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
VISION_M = os.getenv("VISION_MODEL", "openai/gpt-4o-mini").strip()
REASON_M = os.getenv("REASON_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip()
REASONER_A_M = os.getenv("REASONER_A_MODEL", "meta-llama/llama-3.3-70b-instruct:free").strip()
REASONER_B_M = os.getenv("REASONER_B_MODEL", "qwen/qwen3-next-80b-a3b-instruct:free").strip()
JUDGE_M = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b:free").strip()
VIS_FB = [m.strip() for m in os.getenv("VISION_FALLBACKS", "google/gemma-4-31b-it:free").split(",") if m.strip()]
TXT_FB = [m.strip() for m in os.getenv("TEXT_FALLBACKS", "qwen/qwen3-next-80b-a3b-instruct:free,openai/gpt-oss-120b:free").split(",") if m.strip()]
TEMP = float(os.getenv("TEMPERATURE", "0.3"))
TOUT = int(os.getenv("TIMEOUT", "180"))

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

app = FastAPI(title="SkinCoach Web", debug=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ───────────────────────────────────────────────────────────────
def rp(filename: str, default: str = "") -> str:
    p = PROMPTS_DIR / filename
    if p.exists():
        return p.read_text("utf-8").strip()
    return default


def hdr() -> dict:
    return {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://skincoach.ru",
        "X-Title": "SkinCoach",
    }


def xj(text: str):
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except Exception:
            pass
    raise ValueError(f"No JSON: {text[:300]}")


def cm(text: str) -> str:
    text = text.replace("**", "").replace("__", "").replace("```", "").replace("`", "")
    return "\n".join(line.lstrip("#").strip() if line.lstrip().startswith("#") else line for line in text.split("\n"))


async def call_raw(msgs, model, fallbacks, max_tokens=800):
    last_e = None
    async with httpx.AsyncClient(timeout=TOUT) as c:
        for m in [model] + fallbacks:
            try:
                log.info(f"-> {m}")
                r = await c.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=hdr(),
                    json={"model": m, "messages": msgs, "temperature": TEMP, "max_tokens": max_tokens},
                )
                if r.status_code == 200:
                    d = r.json()
                    if "choices" in d and d["choices"]:
                        content = d["choices"][0]["message"].get("content") or ""
                        if isinstance(content, list):
                            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                        if not content.strip():
                            log.warning(f"{m}: empty")
                            continue
                        if content.strip().upper().startswith("ERROR"):
                            log.warning(f"{m}: error msg: {content[:100]}")
                            last_e = f"{m}:{content[:100]}"
                            continue
                        log.info(f"OK: {m}")
                        return content
                log.warning(f"{m}: {r.status_code}")
                last_e = f"{m}:{r.status_code}"
            except httpx.TimeoutException:
                log.warning(f"{m}: timeout")
                last_e = f"{m}:timeout"
            except Exception as e:
                log.warning(f"{m}: {e}")
                last_e = str(e)
    raise Exception(f"All down. {last_e}")


async def cj(msgs, model, fallbacks, max_tokens=800):
    return xj(await call_raw(msgs, model, fallbacks, max_tokens))


async def ct(msgs, model, fallbacks, max_tokens=800):
    return cm(await call_raw(msgs, model, fallbacks, max_tokens))


# ─── ML fallback ───────────────────────────────────────────────────────────
async def ml_predict(image_bytes: bytes) -> dict:
    try:
        from inference import predict_image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp = f.name
        result = predict_image(tmp)
        os.unlink(tmp)
        return {
            "diagnosis": result.get("diagnosis_ru", result.get("diagnosis", "")),
            "confidence": result.get("confidence", 0.0),
            "top3": result.get("top3", []),
            "reliable": result.get("reliable", False),
        }
    except Exception as e:
        log.warning(f"ML fallback failed: {e}")
        return {"diagnosis": "", "confidence": 0.0, "top3": [], "reliable": False}


# ─── Analysis pipeline ─────────────────────────────────────────────────────
async def analyze_image(b64: str, name: str = "друг"):
    user_context = f"Имя: {name}. Давность: не указано. Что пробовали: не указано."

    # STEP 1: Quality check
    log.info("STEP 1/5: Quality check")
    quality = {"usable": True}
    try:
        qp = rp("1_quality.txt", "Проверь качество фото кожи. JSON: {usable: bool, suggestion: string}")
        quality = await cj(
            [
                {"role": "system", "content": qp},
                {"role": "user", "content": [
                    {"type": "text", "text": "Оцени качество фото кожи"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            VISION_M, VIS_FB, 300,
        )
    except Exception as e:
        log.warning(f"Quality check failed: {e}")

    if not quality.get("usable", True):
        return {"status": "reshoot", "message": quality.get("suggestion", "Пересними при дневном свете, крупным планом.")}

    # STEP 2: Vision description
    log.info("STEP 2/5: Vision")
    vision = {"description": "не удалось получить описание"}
    try:
        vp = rp("2_vision.txt", "Опиши детально, что видно на фото кожи. JSON: {description: string, findings: array}")
        log.info(f"Vision prompt length: {len(vp)}, model: {VISION_M}, fallbacks: {VIS_FB}")
        vision = await cj(
            [
                {"role": "system", "content": vp},
                {"role": "user", "content": [
                    {"type": "text", "text": "Опиши что видишь на коже. Будь максимально точным и детальным."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            VISION_M, VIS_FB, 1200,
        )
        log.info(f"Vision OK: {vision}")
    except Exception as e:
        log.exception("Vision failed")

    # STEP 3: Dermatology reasoning
    log.info("STEP 3/5: Reasoner A")
    reason_a = {
        "hypotheses": [{"diagnosis": "требуется уточнение", "probability": 100, "reasoning": "не удалось провести анализ"}],
        "primary_diagnosis": "требуется уточнение",
        "stage": "unknown", "phase": "unknown", "severity": "unknown", "confidence": 0,
    }
    try:
        rp3 = rp("3_reasoning.txt", "Дифференциальная диагностика. JSON.")
        ctx = json.dumps({"vision": vision, "patient": user_context}, ensure_ascii=False)
        reason_a = await cj(
            [{"role": "system", "content": rp3}, {"role": "user", "content": ctx}],
            REASONER_A_M, TXT_FB, 1200,
        )
    except Exception as e:
        log.warning(f"Reasoner A failed: {e}")

    # STEP 4: Psychosomatic reasoning
    log.info("STEP 4/5: Reasoner B")
    reason_b = {"emotional_factor": "не удалось определить", "psychosomatic_pattern": "нет данных", "stress_level": "unknown"}
    try:
        rp3b = rp("reasoner_b_prompt.txt", "Психосоматический анализ. JSON.")
        ctx = json.dumps({"vision": vision, "reasoner_a": reason_a, "patient": user_context}, ensure_ascii=False)
        reason_b = await cj(
            [{"role": "system", "content": rp3b}, {"role": "user", "content": ctx}],
            REASONER_B_M, TXT_FB, 800,
        )
    except Exception as e:
        log.warning(f"Reasoner B failed: {e}")

    # STEP 5: Questions
    log.info("STEP 5/5: Questions")
    questions_data = {"questions": [], "intro": f"{name}, я проанализировал фото."}
    try:
        rp4 = rp("4_questions.txt", "Задай 1-3 уточняющих вопроса. JSON: {intro: string, questions: array}")
        ctx = json.dumps({"vision": vision, "reasoning": reason_a, "patient": user_context}, ensure_ascii=False)
        questions_data = await cj(
            [{"role": "system", "content": rp4}, {"role": "user", "content": ctx}],
            REASON_M, TXT_FB, 800,
        )
    except Exception as e:
        log.warning(f"Questions failed: {e}")

    return {
        "status": "success",
        "step": "questions",
        "intro": questions_data.get("intro", f"{name}, я проанализировал фото."),
        "vision": vision,
        "reasoning": reason_a,
        "reasoner_b": reason_b,
        "questions": questions_data.get("questions", []),
    }


# ─── Final recommendations ─────────────────────────────────────────────────
class AnswersRequest(BaseModel):
    vision: dict
    reasoning: dict
    reasoner_b: dict
    answers: str
    name: str = "друг"


async def generate_final(data: AnswersRequest):
    user_context = f"Имя: {data.name}. Давность: не указано. Что пробовали: не указано."

    all_data = json.dumps({
        "vision": data.vision,
        "reasoning": data.reasoning,
        "reasoner_b": data.reasoner_b,
        "patient_answers": data.answers,
        "patient": user_context,
    }, ensure_ascii=False)

    # Triage
    triage = {"risk_level": "green", "urgency": "routine"}
    try:
        rp5 = rp("5_triage.txt", "Определи уровень риска. JSON.")
        triage = await cj(
            [{"role": "system", "content": rp5}, {"role": "user", "content": all_data}],
            REASON_M, TXT_FB, 400,
        )
    except Exception as e:
        log.warning(f"Triage failed: {e}")

    # Recommendations
    recs = {"diagnosis_summary": "Анализ выполнен", "morning_routine": [], "evening_routine": [], "day_focus": "Следуй программе"}
    try:
        rp6 = rp("6_recommendations.txt", "Составь рекомендации. JSON.")
        ctx = json.dumps({"all_data": json.loads(all_data), "triage": triage}, ensure_ascii=False)
        recs = await cj(
            [{"role": "system", "content": rp6}, {"role": "user", "content": ctx}],
            REASONER_A_M, TXT_FB, 1500,
        )
    except Exception as e:
        log.warning(f"Recommendations failed: {e}")

    # Safety filter
    try:
        rp7 = rp("7_safety.txt", "Проверь безопасность. JSON.")
        await cj(
            [{"role": "system", "content": rp7},
             {"role": "user", "content": json.dumps({"recs": recs, "triage": triage, "reasoning": data.reasoning}, ensure_ascii=False)}],
            REASON_M, TXT_FB, 400,
        )
    except Exception:
        pass

    # Final response
    final = ""
    try:
        rp8 = rp("8_response.txt", "Синтезируй ответ.")
        rp8 = rp8.replace("{name}", data.name).replace("{day}", "1").replace("{week}", "1")
        ctx = json.dumps({
            "recommendations": recs, "triage": triage,
            "reasoner_a": data.reasoning, "reasoner_b": data.reasoner_b,
            "vision": data.vision, "name": data.name,
        }, ensure_ascii=False)
        final = await ct(
            [{"role": "system", "content": rp8}, {"role": "user", "content": ctx}],
            JUDGE_M, TXT_FB, 1500,
        )
    except Exception as e:
        log.warning(f"Final response failed: {e}")
        ds = recs.get("diagnosis_summary", "")
        mr = recs.get("morning_routine", [])
        er = recs.get("evening_routine", [])
        final = f"{ds}\n\nУтро:\n" + "\n".join(f"• {x}" for x in mr[:3]) + "\n\nВечер:\n" + "\n".join(f"• {x}" for x in er[:3])

    return {
        "status": "success",
        "step": "final",
        "triage": triage,
        "recommendations": recs,
        "final_text": final,
    }


# ─── API endpoints ─────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(photo: UploadFile = File(...), name: str = Form("друг")):
    if not OR_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    contents = await photo.read()
    b64 = base64.b64encode(contents).decode("utf-8")

    # Optional ML screening
    ml = await ml_predict(contents)

    result = await analyze_image(b64, name)
    result["ml"] = ml
    return JSONResponse(content=result)


@app.post("/api/final")
async def final_recommendations(body: AnswersRequest):
    result = await generate_final(body)
    return JSONResponse(content=result)


@app.get("/api/health")
async def health():
    return {"ok": True, "openrouter_configured": bool(OR_KEY)}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
