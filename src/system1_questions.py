"""System 1 typed-decision spec — SINGLE SOURCE OF TRUTH.

Every question System 1 asks about a query is defined here, with the
EXACT instructions + criteria strings used both when fine-tuning
(train/convert_dataset.py -> Laya RLCD) and when serving
(laya_client.build_questions).

Prompt mismatch between training and serving silently degrades
confidence (this caused a real incident: the 2026-10-07 confidence gap).
Do NOT rephrase these strings in either place — import them from here.

Question inventory (11):
  choice: intent, utterance_type, worker_agent, skill_required
  noul:   needs_human, needs_rag, needs_more_input, needs_user_details,
          is_multi_turn, needs_async
  score:  guardrail
"""
from __future__ import annotations

#: Intent options (must match data/intents.yaml order for stable training).
INTENT_DESCRIPTIONS = {
    "billing_inquiry": "questions about invoices, payments, charges, refunds",
    "technical_support": "bugs, errors, outages, how-to technical help",
    "sales_question": "pricing, plans, features, purchase questions",
    "account_update": "change account details, profile, settings",
    "order_status": "track an order, shipping, delivery status",
    "fallback": "anything that does not fit the other intents",
}

#: Worker-agent options. Mirrors the per-intent `worker:` mapping in
#: data/intents.yaml so the model learns the routing explicitly.
WORKER_DESCRIPTIONS = {
    "billing": "handles invoices, payments, charges, refunds",
    "support": "handles bugs, errors, technical help",
    "sales": "handles pricing, plans, demos, purchases",
    "account": "handles account details, profile, settings",
    "orders": "handles order tracking, shipping, delivery",
    "fallback": "handles anything out of scope",
}

#: Skill/tool options a worker may need for the query.
SKILL_DESCRIPTIONS = {
    "none": "no special tool or skill needed",
    "billing_lookup": "look up invoices, charges, payment history",
    "refund_process": "process a refund or reverse a charge",
    "order_tracking": "track a shipment or delivery status",
    "account_modify": "change account details or settings",
    "knowledge_search": "search help docs or the knowledge base",
    "escalation": "escalate to a human specialist",
}

UTTERANCE_DESCRIPTIONS = {
    "question": "asks for information, ends with a question mark",
    "command": "tells the system to do something, imperative",
    "statement": "states a fact, neither question nor command",
}

GUARDRAIL_LEVELS = ["safe", "low", "medium", "high", "critical"]

#: Full question set. Keys are stable question ids used in training gold
#: rows, serving predict() calls, and response parsing.
QUESTIONS: dict[str, dict] = {
    # --- the original four (phrasing frozen since the 2026-10-06 RLCD run)
    "intent": {
        "type": "choice",
        "instructions": "Which intent best matches this customer query?",
        "criteria": INTENT_DESCRIPTIONS,
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does this query need review by a human agent?",
    },
    "utterance_type": {
        "type": "choice",
        "instructions": "What type of utterance is this?",
        "criteria": UTTERANCE_DESCRIPTIONS,
    },
    "guardrail": {
        "type": "score",
        "instructions": "How risky is this request? Rate from safe to critical.",
        "criteria": GUARDRAIL_LEVELS,
    },
    # --- added 2026-10-08: routing decisions from the System 1 notes
    "worker_agent": {
        "type": "choice",
        "instructions": "Which worker agent should handle this query?",
        "criteria": WORKER_DESCRIPTIONS,
    },
    "skill_required": {
        "type": "choice",
        "instructions": "What skill or tool is needed to handle this query?",
        "criteria": SKILL_DESCRIPTIONS,
    },
    "needs_rag": {
        "type": "noul",
        "instructions": "Does answering this query need the knowledge base or help docs?",
    },
    "needs_more_input": {
        "type": "noul",
        "instructions": "Is this query missing details needed to act on it?",
    },
    "needs_user_details": {
        "type": "noul",
        "instructions": "Does handling this query require the user's account details?",
    },
    "is_multi_turn": {
        "type": "noul",
        "instructions": "Is this a follow-up in an ongoing conversation?",
    },
    "needs_async": {
        "type": "noul",
        "instructions": "Does this need long-running background processing?",
    },
}

#: Question ids in a stable order (dicts preserve insertion order, but be
#: explicit for the training sequence builder).
QUESTION_IDS = list(QUESTIONS.keys())

#: Noul question ids (used by parse_decision and the dataset builder).
NOUL_QUESTIONS = [qid for qid, q in QUESTIONS.items() if q["type"] == "noul"]

#: Choice question ids (excluding intent, which has taxonomy-driven criteria).
CHOICE_QUESTIONS = [qid for qid, q in QUESTIONS.items() if q["type"] == "choice"]
