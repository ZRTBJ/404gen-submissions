"""Three.js code generator using local vLLM models.

Runs VLM (Qwen2.5-VL-32B) and Code LLM (Qwen2.5-Coder-32B) locally via vLLM,
then validates the output with the Node.js validator.
"""

import base64
import json
import re
import subprocess
from pathlib import Path

from openai import OpenAI
from loguru import logger

VALIDATOR_PATH = Path("/app/validator/tools/validate.js") if Path("/app/validator/tools/validate.js").exists() else None
AGENTS_MD_PATH = Path("/app/AGENTS.md")

_agents_md = ""
if AGENTS_MD_PATH.exists():
    _agents_md = AGENTS_MD_PATH.read_text()

# VLM and Coder connect to local vLLM servers
VLM_CLIENT = None
CODER_CLIENT = None

VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
CODER_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


def init_clients(vlm_port: int = 8001, coder_port: int = 8002):
    """Initialize OpenAI clients pointing at local vLLM servers."""
    global VLM_CLIENT, CODER_CLIENT
    VLM_CLIENT = OpenAI(api_key="local", base_url=f"http://localhost:{vlm_port}/v1")
    CODER_CLIENT = OpenAI(api_key="local", base_url=f"http://localhost:{coder_port}/v1")
    logger.info(f"Initialized local clients: VLM@{vlm_port}, Coder@{coder_port}")


PLANNER_SYSTEM_PROMPT = """\
You are looking at a single 3D object. Produce a structured Object Structural
Description (OSD) as JSON that a Three.js code generator will use to rebuild
this object from procedural primitives.

Return a JSON object with: object_type, overall_description, proportions,
materials_overview, symmetry, and parts[] where each part has: name, primitive,
narrative, color, material, count_hint.

Decomposition: 3-8 parts (simple), 6-15 (medium), 12-25 (complex).
NEVER use "window" as a part name — use "pane", "glass_panel", "windshield".
Return ONLY JSON. No prose, no markdown fences.
"""

MATERIAL_TABLE = """\
Material normalization:
  polished_metal    MeshStandardMaterial  metalness=0.9  roughness=0.2
  brushed_metal     MeshStandardMaterial  metalness=0.8  roughness=0.5
  chrome            MeshStandardMaterial  metalness=0.9  roughness=0.1
  glossy_plastic    MeshStandardMaterial  metalness=0.0  roughness=0.3
  matte_plastic     MeshStandardMaterial  metalness=0.0  roughness=0.8
  rubber            MeshStandardMaterial  metalness=0.0  roughness=0.9
  wood              MeshStandardMaterial  metalness=0.0  roughness=0.6
  ceramic           MeshStandardMaterial  metalness=0.0  roughness=0.4
  clear_glass       MeshPhysicalMaterial  metalness=0.0  roughness=0.05  transmission=0.95  ior=1.5  transparent=true
  frosted_glass     MeshPhysicalMaterial  metalness=0.0  roughness=0.4   transmission=0.7   ior=1.5  transparent=true
  emissive          MeshStandardMaterial  emissive=<color>  emissiveIntensity=1.0
  generic           MeshStandardMaterial  metalness=0.0  roughness=0.7
"""

CODER_SYSTEM_PROMPT = (
    "You are a procedural Three.js code generator. Generate a validator-compatible "
    "JavaScript module from the OSD JSON.\n\n"
    "Rules: return ONLY raw JS code, one export default function generate(THREE), "
    "use fitToUnitCube (0.95/maxDim), deterministic, no forbidden identifiers "
    "(window, document, Date, Math.random, etc.), Uint8Array is plain JS not THREE.\n\n"
    + MATERIAL_TABLE + "\n\n"
    "AGENTS.md:\n" + _agents_md
)

MAX_RETRIES = 3


def download_image(url: str) -> bytes:
    """Download a prompt image from URL."""
    import httpx
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def generate_for_prompt(stem: str, image_url: str, seed: int) -> bytes:
    """Generate a Three.js .js file for a single prompt.

    Returns the .js source as bytes, or raises on failure.
    """
    if VLM_CLIENT is None or CODER_CLIENT is None:
        raise RuntimeError("Clients not initialized — call init_clients() first")

    # Download and encode image
    image_data = download_image(image_url)
    b64 = base64.b64encode(image_data).decode("utf-8")
    mime = "image/png" if image_url.endswith(".png") else "image/jpeg"

    # Stage 1: VLM → OSD
    osd_resp = VLM_CLIENT.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "Analyze this 3D object. Return the OSD JSON."},
            ]},
        ],
        max_tokens=4096,
        temperature=0.0,
        seed=seed,
    )
    osd = osd_resp.choices[0].message.content

    # Stage 2: Code LLM → Three.js
    code_resp = CODER_CLIENT.chat.completions.create(
        model=CODER_MODEL,
        messages=[
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate Three.js for '{stem}'.\n\nOSD:\n{osd}"},
        ],
        max_tokens=16384,
        temperature=0.0,
        seed=seed,
    )
    code = code_resp.choices[0].message.content.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:js|javascript)?\n", "", code)
        code = re.sub(r"\n```$", "", code)

    # Stage 3: Validate + retry
    for attempt in range(MAX_RETRIES):
        passed, error = _validate(code, stem)
        if passed:
            logger.info(f"[{stem}] PASSED (attempt {attempt + 1})")
            return code.encode("utf-8")

        logger.warning(f"[{stem}] FAILED attempt {attempt + 1}: {error[:80]}")
        if attempt < MAX_RETRIES - 1:
            fix_resp = CODER_CLIENT.chat.completions.create(
                model=CODER_MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Fix this Three.js code. Common issues: UNKNOWN_THREE_API (use plain "
                        "Uint8Array not THREE.Uint8Array), FORBIDDEN_IDENTIFIER (never use "
                        "'window' even as variable name), BOUNDING_BOX (fix fitToUnitCube). "
                        "Return ONLY fixed JS code."
                    )},
                    {"role": "user", "content": f"Error: {error}\n\nOSD:\n{osd}\n\nCode:\n{code}"},
                ],
                max_tokens=16384,
                temperature=0.0,
                seed=seed + attempt + 1,
            )
            code = fix_resp.choices[0].message.content.strip()
            if code.startswith("```"):
                code = re.sub(r"^```(?:js|javascript)?\n", "", code)
                code = re.sub(r"\n```$", "", code)

    # Return last attempt even if failed
    logger.error(f"[{stem}] FAILED after {MAX_RETRIES} attempts")
    return code.encode("utf-8")


def _validate(code: str, stem: str) -> tuple[bool, str]:
    """Run the Node.js validator on the generated code."""
    if VALIDATOR_PATH is None or not VALIDATOR_PATH.exists():
        return True, "No validator"

    tmp_path = Path(f"/tmp/{stem}.js")
    tmp_path.write_text(code)

    try:
        result = subprocess.run(
            ["node", str(VALIDATOR_PATH), "--json", str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(VALIDATOR_PATH.parent.parent),
        )
        if result.returncode == 0:
            return True, "PASSED"
        try:
            data = json.loads(result.stdout)
            failures = data.get("failures", [])
            msg = "; ".join(f"{f.get('rule','?')}: {f.get('detail','')}" for f in failures)
            return False, msg
        except json.JSONDecodeError:
            return False, result.stdout[:500] + result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        tmp_path.unlink(missing_ok=True)
