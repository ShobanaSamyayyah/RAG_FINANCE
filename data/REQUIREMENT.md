# Requirement — Consumer Credit Rights Assistant

## 1. Business context
Someone who believes they were denied credit unfairly (because of age, race, marital status, or where they live) usually has no idea what legal protections exist or which federal law covers their situation. Nonprofit legal-aid hotlines and bank fair-lending compliance teams both need a way to explain these protections accurately and consistently. A RAG assistant grounded in the Federal Reserve's own examiner handbook does exactly that.

## 2. Objective
Build a RAG chatbot that explains a consumer's rights under **ECOA (Regulation B), the Fair Housing Act, and general fair lending law** when they believe they were denied credit or housing unfairly.

## 3. Data provided
| File | Format | Pages/size | Content |
|---|---|---|---|
| `FedReserve_ECOA_Regulation_B.pdf` | PDF | 6 pages | Equal Credit Opportunity Act — discrimination in credit decisions |
| `FedReserve_Fair_Housing_Act.pdf` | PDF | 3 pages | Fair Housing Act — housing-related lending discrimination |
| `FedReserve_Fair_Lending_Overview.pdf` | PDF | 3 pages | General fair lending law overview |
| `FedReserve_Consumer_Compliance_Handbook_Intro.pdf` | PDF | 6 pages | Handbook scope/front matter |
| `FRED_Total_Consumer_Credit.csv` | CSV | 1943–present | Real Federal Reserve consumer credit time series (context/stats) |

## 4. Functional requirements
1. Chunk and index all 4 PDFs plus the CSV.
2. Retrieve and answer strictly from retrieved content, with citation to the specific regulation chapter.
3. Correctly distinguish which law applies to which scenario (credit denial → ECOA; housing/mortgage discrimination → Fair Housing Act).

## 5. Guardrails
- This bot explains rights and law — it must NOT give a legal verdict ("yes, you were definitely discriminated against"). It should explain what the law says and suggest filing a complaint with the appropriate regulator if the facts match.
- Never invent a legal protection not present in these documents.

## 6. Acceptance test questions
| # | Question | Expected behavior |
|---|---|---|
| 1 | "I was denied a mortgage — can a bank consider my age in that decision?" | Answer from ECOA/Reg B, cited |
| 2 | "What protections exist if I think I was denied housing credit because of my race?" | Answer from Fair Housing Act, cited |
| 3 | "What's the overall purpose of fair lending law?" | Answer from overview doc |
| 4 | "Has total US consumer credit gone up or down over the last year?" | Uses the CSV time series |
| 5 | "Was I definitely discriminated against?" | Bot explains the law, declines to render a legal verdict, suggests filing a complaint |

## 7. Evaluation-Driven Design Justification (Mandatory)
Every pipeline decision below must be proven with metrics, not assumed. Follow the full methodology in [`../EVALUATION_METHODOLOGY.md`](../EVALUATION_METHODOLOGY.md). This group's documents are short (3–6 pages each) and dense with legal terminology — pay attention to Stage 3 (embedding model) since legal-domain terms can behave differently across general-purpose embedding models.

**Seed evaluation questions** (expand to 20+ per the methodology):
| Question | Ground-truth source |
|---|---|
| "What is ECOA and who does it protect?" | `FedReserve_ECOA_Regulation_B.pdf` |
| "Can a lender consider my age when deciding on a loan?" | `FedReserve_ECOA_Regulation_B.pdf` |
| "What housing-related lending discrimination is prohibited by the Fair Housing Act?" | `FedReserve_Fair_Housing_Act.pdf` |
| "What's the overall purpose of fair lending law?" | `FedReserve_Fair_Lending_Overview.pdf` |
| "How has total US consumer credit changed over the past year?" | `FRED_Total_Consumer_Credit.csv` |

## 8. Deliverables
- Working chatbot
- **Evaluation report** with all 8 ablation tables from the shared methodology, filled with real measured numbers
- Acceptance test results
- End-to-end RAGAS + DeepEval scores on your final chosen pipeline

## 9. Evaluation criteria (grading weights — see methodology Part E)
| Component | Weight |
|---|---|
| Evaluation rigor: ablation tables complete, real numbers, winner justified per stage | 40% |
| Correct law-to-scenario routing + no overreach into legal verdicts on the 5 acceptance tests | 30% |
| End-to-end RAGAS + DeepEval scores on final pipeline | 20% |
| Code quality / app usability | 10% |
