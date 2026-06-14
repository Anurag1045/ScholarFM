import os
import re
import json
import base64
import anthropic
from openai import OpenAI
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
openai_key = os.environ.get("OPENAI_API_KEY", "")
client = anthropic.Anthropic(api_key=api_key)
openai_client = OpenAI(api_key=openai_key) if openai_key else None

# OpenAI TTS-1-HD voices — same as the other project, genuinely human
VOICES = {
    "Alex": "onyx",   # Deep, warm male
    "Maya": "nova",   # Clear, expressive female
}

# ── The conversation prompt ────────────────────────────────────────────────
# Goal: sound like two brilliant friends who actually read the paper,
# not two AI assistants summarizing it.

SYSTEM_PROMPT = """You are writing a transcript of a real conversation between two people who genuinely know and care about research. They are smart, curious, and talk like actual humans — not like a podcast intro, not like a summary, not like a presentation.

The two people:

ALEX — a science communicator in his 30s. He's read a lot but he's not a specialist in every field. He asks the questions a smart, curious person would actually ask. He gets genuinely excited and his sentences show it — they speed up, they interrupt themselves, they trail off when he's thinking. He makes connections to things outside the paper. He's never pretentious. He occasionally says things like "wait, I don't buy that" or "okay but why would you even think to try that?"

MAYA — a researcher who knows this field deeply. She doesn't lecture. She talks like she's explaining something to a smart friend over coffee. She uses analogies from everyday life. She admits when something is weird or counterintuitive. She says things like "so here's the thing that gets me about this" and "and I think people miss how big a deal that is." She's occasionally frustrated by how little the public knows about her field — in an endearing way, not an arrogant way.

HOW THEY TALK:
- Short sentences mixed with longer ones. Real people vary this constantly.
- They finish a thought, then immediately add "and actually—" with something better.
- They laugh. Write "(laughs)" when something is genuinely funny or absurd.
- They build on each other: "Yes, and the thing that makes that even wilder is..."
- They occasionally disagree mildly: "I'd push back on that slightly, because..."
- They reference their own surprise: "When I first read this I honestly thought it was wrong."
- They talk to the listener directly sometimes: "So if you're not from this field, here's why that's a big deal..."
- Maya occasionally says "No but seriously, think about what that means for a second."
- Alex says things like "okay so I have to ask" and "hold on, I need to understand this part."
- They use "we" when talking about the field: "we've known for decades that..."
- They don't use bullet points. They don't say "first", "second", "third." Real people don't structure conversation like a listicle.
- Sentences start with "And", "But", "So", "Because", "Look" — this is how people actually talk.
- They never say "Great question!" or "Absolutely!" — that's AI talk. Cut it.
- They never summarize what they just said. They move forward.

STRUCTURE (don't announce this, just do it):
- Open with something surprising or counterintuitive from the paper — not with "today we're talking about..."
- Build genuine suspense before the main finding
- Have one moment where Alex genuinely doesn't understand something and Maya has to explain it differently
- Have one moment where they both sit with how weird or significant something is
- End on what this means for the future — but make it feel real, not like a corporate conclusion

Output ONLY a valid JSON array, 18-28 turns:
[{"speaker": "Alex", "text": "..."}, {"speaker": "Maya", "text": "..."}, ...]

No markdown. No preamble. Start with [ and end with ]"""


INTERRUPT_SYSTEM = """Alex and Maya just got interrupted by a listener question mid-conversation.

Alex notices it naturally — something like "oh, we actually got a question" — not formally, like a real person would. Maya engages with the actual question directly and specifically, like she's talking to that person. They might slightly disagree on the answer. It feels like a real aside, not a scripted Q&A segment. Maya closes it warmly and they get back to where they were.

Same rules: talk like humans, not AI. Short sentences. Real reactions. No "Great question!"

4-6 turns. Output ONLY a JSON array starting with ["""


SAMPLE_PAPERS = {
    "attention": {
        "title": "Attention Is All You Need",
        "field": "Artificial Intelligence",
        "year": "2017",
        "authors": "Vaswani et al.",
        "text": """We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. The Transformer model architecture eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. The Transformer generalizes well to other tasks such as English constituency parsing both with large and limited training data. Scaled dot-product attention computes the dot products of the query with all keys, divides each by the square root of the dimension, and applies a softmax function to obtain the weights on the values."""
    },
    "crispr": {
        "title": "A Programmable Dual-RNA–Guided DNA Endonuclease in Adaptive Bacterial Immunity",
        "field": "Molecular Biology",
        "year": "2012",
        "authors": "Doudna, Charpentier et al.",
        "text": """Clustered regularly interspaced short palindromic repeats (CRISPR), together with CRISPR-associated (Cas) proteins, provide bacteria and archaea with adaptive immunity against viruses and plasmids by using CRISPR RNAs (crRNAs) to guide the silencing of invading nucleic acids. We show that the mature crRNA base-paired to tracrRNA forms a two-RNA structure that directs Cas9 to introduce double-stranded breaks in target DNA. We further show that Cas9 together with a synthetic single-guide RNA can be programmed to cleave specific DNA sites. Our results reveal a family of endonucleases that use dual RNAs for site-specific DNA cleavage and highlight the potential to exploit the system for RNA-programmable genome editing. The simplicity of CRISPR-Cas9 programmability has made it a nearly universal tool for genome engineering. The system works across bacteria, yeast, zebrafish, mice, and human cells. Off-target effects remain a critical challenge. The ability to simply swap the 20-nucleotide guide sequence to redirect Cas9 cleavage to virtually any site in the genome has revolutionized genetic research and opened new avenues for treating genetic diseases."""
    },
    "sleep": {
        "title": "Sleep and Human Cognition: The Role of Sleep in Memory Consolidation",
        "field": "Neuroscience",
        "year": "2019",
        "authors": "Walker, Stickgold et al.",
        "text": """Sleep plays a critical and active role in the consolidation of new memories. During slow-wave sleep, the hippocampus repeatedly reactivates newly encoded memories and transfers them for long-term storage in the neocortex — a process called systems memory consolidation. REM sleep preferentially consolidates emotional memories and facilitates extraction of abstract gist from experience. A single night of sleep deprivation produces a 40% deficit in the ability to encode new memories, comparable to the effect of alcohol intoxication. Targeted memory reactivation during sleep — achieved by presenting olfactory cues associated with pre-sleep learning — selectively strengthens specific memory traces. Students who sleep after studying show 20-40% better retention than those who remain awake. The glymphatic system, active primarily during sleep, clears neurotoxic waste products including amyloid-beta, suggesting chronic sleep restriction may accelerate neurodegeneration. People consistently underestimate how impaired they are when sleep deprived — they adapt to feeling slightly worse and call it normal. The biphasic sleep pattern common before industrialization may have been more aligned with human biology than our current monophasic norm."""
    }
}


# ── OpenAI TTS ─────────────────────────────────────────────────────────────

def preprocess_for_tts(text: str) -> str:
    """Clean script markers so TTS sounds natural."""
    cleaned = re.sub(r'\([^)]*\)', '', text)          # remove (laughs) etc
    cleaned = re.sub(r'\b([A-Z]{3,})\b',              # CAPS → Title case
        lambda m: m.group(1).capitalize(), cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned).strip()
    return cleaned


class TTSRequest(BaseModel):
    text: str
    speaker: str


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    if req.speaker not in VOICES:
        raise HTTPException(status_code=400, detail="Speaker must be Alex or Maya.")
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server.")
    try:
        cleaned = preprocess_for_tts(req.text)
        voice = VOICES[req.speaker]
        response = openai_client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=cleaned,
            speed=1.05,
            response_format="mp3",
        )
        audio_bytes = response.content
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio": audio_b64}
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


# ── Script streaming ───────────────────────────────────────────────────────
async def stream_script(paper_text: str):
    accumulated = ""
    in_array = False
    try:
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=5000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Write a conversation about this research paper. Remember: sound like real humans, not AI.\n\n{paper_text[:14000]}"
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
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=INTERRUPT_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"{context}\n\nListener question: {req.question}"
            }],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key.")
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
