"""
Two independent guardrails required by REQUIREMENT.md section 5:

1. Groundedness: refuse if the top reranked chunk isn't confidently relevant
   (score-threshold check — lives inline in pipeline.py, same as the reference bot).
2. No-legal-verdict: this bot explains rights and law; it must never render a verdict
   like "yes, you were discriminated against." Detect verdict-seeking questions and
   force the prompt to explain-the-law-and-suggest-filing instead of judging facts.
"""

import re

VERDICT_PATTERNS = [
    r"\bwas\s+i\s+(definitely\s+)?discriminated\b",
    r"\bdid\s+(they|the bank|the lender)\s+discriminate\b",
    r"\bis\s+this\s+(illegal|discrimination)\b",
    r"\bam\s+i\s+a\s+victim\b",
    r"\bshould\s+i\s+sue\b",
    r"\bdo\s+i\s+have\s+a\s+(case|lawsuit)\b",
]

VERDICT_INSTRUCTION = (
    "This question is asking you to judge whether specific facts amount to illegal "
    "discrimination. You must NOT render that verdict. Instead: explain what the relevant "
    "law (ECOA/Regulation B for credit decisions, or the Fair Housing Act for housing/mortgage "
    "lending) says about the factor(s) mentioned, and suggest that if the facts match a "
    "prohibited basis, the person consider filing a complaint with the appropriate regulator "
    "(e.g., the CFPB or the Federal Reserve). Do not say whether discrimination 'definitely' "
    "or 'probably' occurred."
)


def is_verdict_seeking(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in VERDICT_PATTERNS)
