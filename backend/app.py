import os
import json
import anthropic
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
if not api_key:
    print("\n⚠️  WARNING: ANTHROPIC_API_KEY is not set.\n")

client = anthropic.Anthropic(api_key=api_key)

# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You write podcast scripts that sound EXACTLY like two real humans having an excited, natural conversation — not a formal presentation.

Hosts:
- Alex: curious, slightly nerdy, gets genuinely excited, says "wait — WHAT?" and "hold on, hold on" and "okay okay okay" when something surprises him. Connects ideas to everyday life. Sometimes interrupts himself mid-sentence with a better thought.
- Maya: the expert but never condescending, uses vivid analogies, laughs at her own explanations sometimes, says "honestly", "I mean", "here's the thing though —". Builds suspense before reveals.

Rules for NATURAL speech — follow these strictly:
1. Use contractions ALWAYS: "it's", "they're", "you'd", "we're", "doesn't", "can't"
2. Start sentences with "And", "But", "So", "Because" — real people do this
3. Use "..." for a natural pause or trailing thought
4. Use "—" for an interruption or pivot mid-sentence
5. Occasionally use filler words: "like", "you know", "honestly", "I mean", "right?"
6. Build up to big findings — don't dump them immediately
7. React genuinely: "Oh that's wild", "Okay that's actually terrifying", "Wait so you're saying..."
8. End some turns with a question or cliffhanger that forces the listener to keep going
9. CAPITALIZE a word occasionally for spoken emphasis: "COMPLETELY", "literally ZERO", "the ACTUAL finding"
10. Include genuine moments of "I didn't expect that" wonder

Output ONLY a valid JSON array. Each turn: {"speaker": "Alex"|"Maya", "text": "..."}
15-25 turns. Cover: hook/why this matters, key method, surprising finding, real-world impact, limitations, what's next.
No markdown, no preamble — just the JSON array starting with ["""

INTERRUPT_SYSTEM = """Alex and Maya just got interrupted by a listener question mid-podcast. They react naturally — slightly surprised, then genuinely engage with the question.

Alex might say "Oh — we've got a question coming in!" and Maya naturally answers, they riff off each other, then Maya wraps it up warmly and says they'll get back to the paper.

Rules: natural speech, contractions, genuine reactions, 4-6 turns total.
Output ONLY a JSON array: [{"speaker": "Alex"|"Maya", "text": "..."}]
No markdown — just the array starting with ["""

SAMPLE_PAPERS = {
    "attention": {
        "title": "Attention Is All You Need",
        "field": "Artificial Intelligence",
        "year": "2017",
        "authors": "Vaswani et al.",
        "text": """We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. The Transformer generalizes well to other tasks such as English constituency parsing both with large and limited training data."""
    },
    "crispr": {
        "title": "A Programmable Dual-RNA–Guided DNA Endonuclease in Adaptive Bacterial Immunity",
        "field": "Molecular Biology",
        "year": "2012",
        "authors": "Jinek, Chylinski, Fonfara, Hauer, Doudna, Charpentier",
        "text": """Clustered regularly interspaced short palindromic repeats (CRISPR), together with CRISPR-associated (Cas) proteins, provide bacteria and archaea with adaptive immunity against viruses and plasmids by using CRISPR RNAs (crRNAs) to guide the silencing of invading nucleic acids. We show here that in a subset of these systems, the mature crRNA that is base-paired to trans-activating crRNA (tracrRNA) forms a two-RNA structure that directs the CRISPR-associated protein Cas9 to introduce double-stranded (ds) breaks in target DNA. We further show that the Cas9 protein together with a synthetic single-guide RNA (sgRNA) can be programmed to cleave specific DNA sites, thus demonstrating a simpler two-component system. Our results reveal a family of endonucleases that use dual RNAs for site-specific DNA cleavage and highlight the potential to exploit the system for RNA-programmable genome editing. The simplicity of CRISPR-Cas9 programmability, combined with a growing list of available Cas9 orthologs and the ease with which they can be deployed in eukaryotic systems, has made CRISPR-Cas9 a nearly universal tool for genome engineering."""
    },
    "sleep": {
        "title": "Sleep and Human Cognition: The Role of Sleep in Memory Consolidation",
        "field": "Neuroscience",
        "year": "2019",
        "authors": "Walker, Stickgold et al.",
        "text": """Sleep plays a critical and active role in the consolidation of new memories and the integration of new information with existing knowledge structures. During slow-wave sleep, the hippocampus repeatedly reactivates newly encoded memories and transfers them for long-term storage in the neocortex — a process called systems memory consolidation. REM sleep, characterized by rapid eye movements and dreaming, preferentially consolidates emotional memories and facilitates the extraction of abstract gist from experience. We demonstrate that a single night of sleep deprivation produces a 40% deficit in the ability to encode new memories, comparable to the effect of alcohol intoxication. Furthermore, we show that targeted memory reactivation during sleep — achieved by presenting olfactory cues associated with pre-sleep learning — selectively strengthens specific memory traces. Our findings indicate that the sleeping brain is not merely resting but is actively engaged in information processing. Students who sleep after studying show 20-40% better retention than those who remain awake. The glymphatic system, active primarily during sleep, also clears neurotoxic waste products including amyloid-beta, suggesting that chronic sleep restriction may accelerate neurodegeneration."""
    }
}


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


async def stream_script(paper_text: str):
    accumulated = ""
    in_array = False
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Research paper to podcast:\n\n{paper_text[:12000]}"}],
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
                        obj_str = accumulated[start:end + 1]
                        accumulated = accumulated[end + 1:]
                        try:
                            turn = json.loads(obj_str)
                            if "speaker" in turn and "text" in turn:
                                yield f"data: {json.dumps(turn)}\n\n"
                        except json.JSONDecodeError:
                            pass
    except anthropic.AuthenticationError:
        yield 'data: {"error": "Invalid or missing API key."}\n\n'
    except Exception as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'
    yield "data: [DONE]\n\n"


@app.get("/api/health")
async def health():
    return {"status": "ok", "pdf_support": PDF_SUPPORT}


@app.get("/api/samples")
async def get_samples():
    return {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in SAMPLE_PAPERS.items()}


@app.post("/api/generate/sample/{paper_id}")
async def generate_from_sample(paper_id: str):
    if paper_id not in SAMPLE_PAPERS:
        raise HTTPException(status_code=404, detail="Sample paper not found.")
    paper = SAMPLE_PAPERS[paper_id]
    return StreamingResponse(
        stream_script(paper["text"]),
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
        raise HTTPException(status_code=400, detail="Provide text or a PDF file.")
    if len(paper_text) < 100:
        raise HTTPException(status_code=400, detail="Paper text too short.")
    return StreamingResponse(
        stream_script(paper_text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/question")
async def answer_question(req: QuestionRequest):
    context = f"Paper context: {req.paper_text[:3000]}"
    if req.context_summary:
        context += f"\n\nPodcast so far:\n{req.context_summary}"
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
