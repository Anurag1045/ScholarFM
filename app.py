import os
import json
import anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── PDF support (optional — graceful fallback if not installed) ────────────
try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    print("\n⚠️  WARNING: ANTHROPIC_API_KEY is not set. Requests will fail.\n")

client = anthropic.Anthropic(api_key=api_key)

SYSTEM_PROMPT = """You are a podcast script writer. When given a research paper, generate a natural, engaging conversation between two podcast hosts:
- Alex: enthusiastic, asks great questions, connects ideas to real-world applications
- Maya: the expert, explains concepts clearly, uses good analogies, occasionally nerdy

Output ONLY a valid JSON array of turns. Each turn: {"speaker": "Alex", "text": "..."} or {"speaker": "Maya", "text": "..."}
Make it 15-25 turns. Cover: what the paper is about, why it matters, key findings, limitations, future implications.
Keep each turn 1-3 sentences. Be conversational, not academic. No markdown, no preamble — just the JSON array starting with ["""

INTERRUPT_SYSTEM = """You are Alex and Maya, two podcast hosts who just got interrupted mid-episode by a listener question.
Answer the question naturally as a brief back-and-forth (4-6 turns total between both hosts), then end with Maya saying they'll get back to the paper.
Output ONLY a valid JSON array: [{"speaker": "Alex", "text": "..."}, ...]
No markdown, no preamble — just the JSON array starting with ["""


class QuestionRequest(BaseModel):
    question: str
    paper_text: str
    context_summary: Optional[str] = ""


def extract_pdf_text(file_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not installed. Run: pip install PyMuPDF")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Try pasting the text instead.")
    return text


async def stream_podcast_script(paper_text: str):
    accumulated = ""
    in_array = False

    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Research paper:\n\n{paper_text[:12000]}"}],
        ) as stream:
            for chunk in stream.text_stream:
                accumulated += chunk

                if not in_array and "[" in accumulated:
                    in_array = True

                if in_array:
                    while True:
                        start = accumulated.find("{")
                        if start == -1:
                            break
                        depth = 0
                        end = -1
                        for i, c in enumerate(accumulated[start:], start):
                            if c == "{":
                                depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0:
                                    end = i
                                    break
                        if end == -1:
                            break
                        obj_str = accumulated[start: end + 1]
                        accumulated = accumulated[end + 1:]
                        try:
                            turn = json.loads(obj_str)
                            if "speaker" in turn and "text" in turn:
                                yield f"data: {json.dumps(turn)}\n\n"
                        except json.JSONDecodeError:
                            pass
    except anthropic.AuthenticationError:
        yield 'data: {"error": "Invalid or missing API key. Set ANTHROPIC_API_KEY in your terminal."}\n\n'
    except anthropic.RateLimitError:
        yield 'data: {"error": "Rate limit hit. Wait a moment and try again."}\n\n'
    except Exception as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'

    yield "data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.post("/generate")
async def generate_podcast(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if file and file.filename:
        content = await file.read()
        paper_text = extract_pdf_text(content)
    elif text and text.strip():
        paper_text = text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide either text or a PDF file.")

    if len(paper_text) < 100:
        raise HTTPException(status_code=400, detail="Paper text too short (minimum 100 characters).")

    return StreamingResponse(
        stream_podcast_script(paper_text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/question")
async def answer_question(req: QuestionRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")

    context = f"Paper context: {req.paper_text[:3000]}"
    if req.context_summary:
        context += f"\n\nPodcast so far: {req.context_summary}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=INTERRUPT_SYSTEM,
            messages=[{"role": "user", "content": f"{context}\n\nListener question: {req.question}"}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    raw = message.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        turns = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract array from anywhere in the text
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            turns = json.loads(raw[start:end + 1])
        else:
            raise HTTPException(status_code=500, detail="Could not parse response from Claude.")

    return {"turns": turns}
