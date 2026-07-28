"""Document Retrieval Agent using RAG."""

from __future__ import annotations

from src.config import Settings
from src.llm_client import LLMService
from src.rag.retriever import TeaRetriever
from src.schemas import AgentMessage


class DocumentAgent:
    """Retrieve official evidence and create a grounded draft."""

    name = "DocumentAgent"

    def __init__(
        self,
        llm: LLMService,
        settings: Settings,
        retriever: TeaRetriever | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self._retriever = retriever

    @property
    def retriever(self) -> TeaRetriever:
        """Load the retriever only when it is needed."""
        if self._retriever is None:
            self._retriever = TeaRetriever()

        return self._retriever

    def handle(
        self,
        message: AgentMessage,
    ) -> AgentMessage:
        """Handle an official-document search request."""
        question = str(
            message.payload.get(
                "question",
                "",
            )
        ).strip()

        try:
            evidence = self.retriever.retrieve(
                question
            )

            if not evidence:
                payload = {
                    "status": "no_evidence",
                    "draft_answer": (
                        "No sufficiently relevant "
                        "official-document evidence "
                        "was found."
                    ),
                    "evidence": [],
                    "model": (
                        self.settings.reasoning_model
                    ),
                }

            else:
                evidence_text = "\n\n".join(
                    (
                        f"[{item['source_id']}] "
                        f"Document: {item['document']} | "
                        f"Page: {item['page']} | "
                        f"Similarity: "
                        f"{item['similarity_score']}\n"
                        f"{item['text']}"
                    )
                    for item in evidence
                )

                system_prompt = """
You are the Document Retrieval Agent for a Sri Lankan tea-industry
assistant.

Answer only from the retrieved evidence supplied by the application.

Rules:
1. Do not use your general knowledge.
2. Do not invent numbers, years, reasons, policies, or explanations.
3. Cite evidence using [S1], [S2], and similar source IDs.
4. Keep measurements and units exactly as shown in the evidence.
5. When evidence is incomplete, clearly state that the retrieved
   documents do not provide enough information.
6. Use simple, professional English.
""".strip()

                draft_answer = (
                    self.llm.complete_text(
                        model=(
                            self.settings
                            .reasoning_model
                        ),
                        system_prompt=(
                            system_prompt
                        ),
                        user_prompt=(
                            f"Question:\n{question}\n\n"
                            "Retrieved evidence:\n"
                            f"{evidence_text}"
                        ),
                        temperature=0.1,
                    )
                )

                payload = {
                    "status": "success",
                    "draft_answer": draft_answer,
                    "evidence": evidence,
                    "model": (
                        self.settings.reasoning_model
                    ),
                }

        except Exception as error:
            payload = {
                "status": "error",
                "draft_answer": "",
                "evidence": [],
                "error": str(error),
                "model": (
                    self.settings.reasoning_model
                ),
            }

        return AgentMessage(
            sender=self.name,
            receiver="Orchestrator",
            message_type="DOCUMENT_RESULT",
            payload=payload,
        )