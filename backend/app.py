import os
import re
import json
import base64
import asyncio
import anthropic
import edge_tts
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=api_key)

# edge-tts: Microsoft Azure Neural voices — free, no API key, genuinely human quality
VOICES = {
    "Alex": {"voice": "en-US-GuyNeural",   "rate": "+8%",  "pitch": "-4Hz"},
    "Maya": {"voice": "en-US-JennyNeural", "rate": "+0%",  "pitch": "+3Hz"},
}

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are writing a real conversation between two people who've actually read this research paper and find it fascinating. NOT a podcast intro. NOT a summary. A genuine back-and-forth between two sharp, curious humans.

WHO THEY ARE:

ALEX — science communicator, mid-30s, widely read but not a specialist. He asks exactly what a smart non-expert would ask. His excitement shows in his sentences — they speed up, trail off, interrupt themselves. He says things like "okay but wait" and "I genuinely didn't expect that" and "hold on, I want to make sure I understand this." He's never pretentious. Sometimes he pushes back: "I don't know if I buy that."

MAYA — domain expert who talks like a human, not a textbook. She explains things the way you'd explain to a smart friend, not a student. She uses real-world analogies. She admits when findings surprised even her. She says "so here's the thing that gets me" and "and I think people completely miss how significant that is" and "no but seriously, think about what that actually means." She occasionally gets a little frustrated — in an endearing way — about how little attention something gets.

HOW REAL HUMANS ACTUALLY TALK:
- Short punchy sentences mixed with longer rambling ones. Constantly varying.
- They add things mid-thought: "and actually — wait, no. Let me say that differently."
- They laugh at absurd implications. Write (laughs) when something is genuinely funny or wild.
- They build on each other: "Yes! And what makes that even more insane is..."
- They mildly disagree: "I'd actually push back on that a bit, because..."
- They address the listener: "So if you're not in this field, here's why your jaw should be dropping right now."
- Alex says: "okay so I have to ask", "hold on", "wait so you're saying", "that can't be right"
- Maya says: "no but think about what that means", "which sounds simple but it absolutely is not", "and this is the part that gets me"
- Filler words: "so", "like", "you know", "honestly", "I mean", "actually", "look"
- Sentences starting with: And, But, So, Because, Look — this is how people talk
- NEVER: "Great question!", "Absolutely!", "Certainly!", "That's a great point!" — these are AI tells. Cut every single one.
- NEVER summarize what was just said. Move forward always.
- NEVER use "firstly", "secondly", "in conclusion", "to summarize" — not how humans talk

STRUCTURE — don't announce it, just do it:
- Open mid-thought, with something surprising or counterintuitive. NOT "today we're talking about..."
- Build suspense before the main finding — make the listener lean in
- Include one moment where Alex genuinely doesn't follow and Maya re-explains with a different analogy
- Include one moment where they both go quiet for a beat and just sit with how weird something is
- End on what this actually changes — but make it feel real, not like a corporate slide

Output ONLY valid JSON array, 12-16 turns (keep it tight and punchy):
[{"speaker": "Alex", "text": "..."}, {"speaker": "Maya", "text": "..."}, ...]
No markdown. No preamble. Start with [ and end with ]"""


INTERRUPT_SYSTEM = """Alex and Maya just got a live question from a listener mid-conversation.

Alex notices it naturally — like a real person would, not formally. Maya engages with the actual question directly and specifically. They riff. They might slightly disagree on the answer. It feels like a real aside between two people, not a scripted Q&A. Maya closes it warmly and they naturally get back to the paper.

Same rules: real humans only. No "Great question!" No AI tells. 4-6 turns.
Output ONLY a JSON array starting with ["""


SAMPLE_PAPERS = {
    "attention": {
        "title": "Attention Is All You Need",
        "field": "Artificial Intelligence",
        "year": "2017",
        "text": """We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU. The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. The Transformer generalizes well to other tasks such as English constituency parsing. Scaled dot-product attention computes the dot products of the query with all keys, divides each by the square root of the dimension, and applies a softmax function to obtain the weights on the values. We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs."""
    },
    "crispr": {
        "title": "CRISPR-Cas9: A Programmable Genome Editor",
        "field": "Molecular Biology",
        "year": "2012",
        "text": """Clustered regularly interspaced short palindromic repeats (CRISPR), together with CRISPR-associated (Cas) proteins, provide bacteria and archaea with adaptive immunity against viruses and plasmids by using CRISPR RNAs to guide the silencing of invading nucleic acids. We show that the mature crRNA base-paired to tracrRNA forms a two-RNA structure that directs Cas9 to introduce double-stranded breaks in target DNA. We further show that Cas9 together with a synthetic single-guide RNA can be programmed to cleave specific DNA sites. Our results reveal a family of endonucleases that use dual RNAs for site-specific DNA cleavage and highlight the potential to exploit the system for RNA-programmable genome editing. The system works across bacteria, yeast, zebrafish, mice, and human cells. The ability to simply swap the 20-nucleotide guide sequence to redirect Cas9 cleavage to virtually any site in the genome has revolutionized genetic research. A single night of sleep deprivation produces measurable cognitive deficits comparable to being legally drunk."""
    },
    "sleep": {
        "title": "Sleep and Memory Consolidation",
        "field": "Neuroscience",
        "year": "2019",
        "text": """Sleep plays a critical and active role in the consolidation of new memories. During slow-wave sleep, the hippocampus repeatedly reactivates newly encoded memories and transfers them for long-term storage in the neocortex. REM sleep preferentially consolidates emotional memories and facilitates extraction of abstract gist from experience. A single night of sleep deprivation produces a 40% deficit in the ability to encode new memories, comparable to the effect of alcohol intoxication. Targeted memory reactivation during sleep — achieved by presenting olfactory cues associated with pre-sleep learning — selectively strengthens specific memory traces. Students who sleep after studying show 20-40% better retention than those who remain awake. The glymphatic system, active primarily during sleep, clears neurotoxic waste products including amyloid-beta, suggesting chronic sleep restriction may accelerate neurodegeneration. People consistently underestimate how impaired they are when sleep deprived — they adapt to feeling slightly worse and call it normal."""
    }
}


# ── TTS ────────────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    text = re.sub(r'\([^)]+\)', '', text)   # remove (laughs) etc
    text = re.sub(r'  +', ' ', text).strip()
    return text


async def synthesize(text: str, speaker: str) -> bytes:
    cfg = VOICES.get(speaker, VOICES["Alex"])
    cleaned = clean_for_tts(text)
    communicate = edge_tts.Communicate(
        text=cleaned,
        voice=cfg["voice"],
        rate=cfg["rate"],
        pitch=cfg["pitch"],
    )
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    if not chunks:
        raise ValueError("No audio generated")
    return b"".join(chunks)


class TTSRequest(BaseModel):
    text: str
    speaker: str


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    if req.speaker not in VOICES:
        raise HTTPException(status_code=400, detail="Speaker must be Alex or Maya.")
    try:
        audio_bytes = await asyncio.wait_for(synthesize(req.text, req.speaker), timeout=20.0)
        return {"audio": base64.b64encode(audio_bytes).decode()}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="TTS timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# ── PDF ────────────────────────────────────────────────────────────────────

def extract_pdf_text(file_bytes: bytes) -> str:
    if not PDF_SUPPORT:
        raise HTTPException(status_code=500, detail="PDF support not installed.")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    return text


# ── Script generation ──────────────────────────────────────────────────────

async def stream_script(paper_text: str):
    accumulated = ""
    in_array = False
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=3500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Write the conversation. Sound like real humans — no AI tells.\n\n{paper_text[:14000]}"
            }],
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
                        depth, end = 0, -1
                        for i, c in enumerate(accumulated[start:], start):
                            if c == "{": depth += 1
                            elif c == "}":
                                depth -= 1
                                if depth == 0: end = i; break
                        if end == -1:
                            break
                        obj_str = accumulated[start:end + 1]
                        accumulated = accumulated[end + 1:]
                        try:
                            turn = json.loads(obj_str)
                            if "speaker" in turn and "text" in turn:
                                yield f"data: {json.dumps(turn)}\n\n"
                        except json.JSONDecodeError:
                            pass
    except anthropic.AuthenticationError:
        yield 'data: {"error": "Invalid Anthropic API key."}\n\n'
    except Exception as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'
    yield "data: [DONE]\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "pdf": PDF_SUPPORT}


@app.post("/api/generate/sample/{paper_id}")
async def generate_from_sample(paper_id: str):
    if paper_id not in SAMPLE_PAPERS:
        raise HTTPException(status_code=404, detail="Sample not found.")
    return StreamingResponse(
        stream_script(SAMPLE_PAPERS[paper_id]["text"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
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
        raise HTTPException(status_code=400, detail="Provide text or a PDF.")
    if len(paper_text) < 100:
        raise HTTPException(status_code=400, detail="Text too short.")
    return StreamingResponse(
        stream_script(paper_text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class QuestionRequest(BaseModel):
    question: str
    paper_text: str
    context_summary: Optional[str] = ""


@app.post("/api/question")
async def answer_question(req: QuestionRequest):
    context = f"Paper context: {req.paper_text[:3000]}"
    if req.context_summary:
        context += f"\n\nConversation so far:\n{req.context_summary}"
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=INTERRUPT_SYSTEM,
            messages=[{"role": "user", "content": f"{context}\n\nListener question: {req.question}"}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1][4:].strip() if len(parts) > 1 else raw
    try:
        turns = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("["), raw.rfind("]")
        if s != -1 and e != -1:
            turns = json.loads(raw[s:e + 1])
        else:
            raise HTTPException(status_code=500, detail="Could not parse response.")
    return {"turns": turns}
