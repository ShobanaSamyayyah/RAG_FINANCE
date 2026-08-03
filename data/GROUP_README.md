# Group 6 — Consumer Credit Rights Assistant

## Requirement
Build a RAG chatbot for a **consumer credit-rights helpline** (like a nonprofit legal-aid hotline, or a bank's internal fair-lending compliance training bot) that explains a consumer's rights under federal fair-lending law when they believe they were denied credit unfairly.

**Why this is coherent:** all 4 documents are chapters from the **same real source** — the Federal Reserve's Consumer Compliance Handbook, the actual manual bank examiners use. One agency, one mission (fair lending oversight), complementary chapters.

## Data sources (all real Federal Reserve regulatory chapters)
| File | Pages | Words | Covers |
|---|---|---|---|
| `FedReserve_ECOA_Regulation_B.pdf` | 6 | ~3,672 | Equal Credit Opportunity Act — discrimination in credit decisions |
| `FedReserve_Fair_Lending_Overview.pdf` | 3 | ~1,596 | General fair lending law overview |
| `FedReserve_Fair_Housing_Act.pdf` | 3 | ~1,478 | Fair Housing Act — mortgage/housing-related lending discrimination |
| `FedReserve_Consumer_Compliance_Handbook_Intro.pdf` | 6 | ~1,007 | Handbook front matter/scope |

**18 pages of real federal regulatory text, one source, one mission.**

## Bonus: real structured data
| File | What it is |
|---|---|
| `FRED_Total_Consumer_Credit.csv` | Real Federal Reserve Economic Data (FRED) time series — total outstanding US consumer credit, monthly, 1943–present |

## Sample questions to validate retrieval
- "I was denied a mortgage — can a bank consider my age in that decision?" (ECOA/Reg B)
- "What protections exist if I think I was denied housing credit because of my race?" (Fair Housing Act)
- "What's the overall purpose of fair lending law?" (overview)

## Manual download to extend this group (real, just blocked from this environment)
- CFPB Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/ — real complaint narratives that would pair naturally with these regulations (same consumer-rights mission). Download via a normal browser.
