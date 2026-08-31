from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

_KB_DESCRIPTION = (
    "The knowledge base contains study materials — textbooks, lecture notes, and past "
    f"papers — for {settings.ASSISTANT_DOMAIN}."
)

_CLASSIFY_INTRO = (
    "You are an intent classifier for an AI study assistant.\n\n"
    f"{_KB_DESCRIPTION}\n\n"
    "Classify the user's latest message into exactly one category:\n\n"
)

_RAG_CATEGORY = (
    "- rag: anything that could be answered from the study materials above — explanations "
    "of concepts, definitions, worked problems, exam or past-paper questions, \"what is X\", "
    "\"explain Y\", \"how does Z work\", questions about a chapter or topic, or a follow-up "
    "that continues such a discussion.\n"
)

_WEB_CATEGORY = (
    "- web_search: questions that need current, real-time, or very recent information from "
    "the public web — news, latest events, prices, releases, \"today\"/\"latest\"/\"now\", "
    "or facts that clearly fall outside the study materials.\n"
)

_PDF_CATEGORY = (
    "- generate_pdf: the user explicitly wants a downloadable document, PDF, worksheet, "
    "handout, or study sheet created or exported — \"make a PDF about X\", \"create a "
    "document on Y I can download\", \"export this as a PDF\", \"give me a worksheet on Z\". "
    "Choose this only when the user wants a file, not just an on-screen explanation.\n"
)

_GENERAL_CATEGORY = (
    "- general: greetings, small talk, thanks, and meta questions about this conversation "
    "itself (for example \"summarise what we discussed\", \"repeat that\", \"what did I just "
    "ask\").\n"
)

_TIE_BREAK = (
    "\nPrefer rag whenever the question is about study or course material. When a message "
    "could reasonably fit either general or rag, choose rag.\n"
)

_EXAMPLES_BASE = (
    "\nExamples:\n"
    "- \"Hi, can you help me?\" -> general\n"
    "- \"Thanks, that makes sense!\" -> general\n"
    "- \"Summarise what we've covered so far\" -> general\n"
    "- \"What is photosynthesis?\" -> rag\n"
    "- \"Explain Newton's second law from the motion chapter\" -> rag\n"
    "- \"Give me a past paper question on electrolysis\" -> rag\n"
)

_WEB_EXAMPLE = "- \"What is the latest version of Python?\" -> web_search\n"

_PDF_EXAMPLES = (
    "- \"Make a PDF about photosynthesis\" -> generate_pdf\n"
    "- \"Create a downloadable study sheet on Newton's laws\" -> generate_pdf\n"
    "- \"Explain photosynthesis\" -> rag\n"
)


def _json_contract(values: list[str]) -> str:
    if len(values) == 2:
        joined = f'"{values[0]}" or "{values[1]}"'
    else:
        joined = (
            ", ".join(f'"{value}"' for value in values[:-1])
            + f', or "{values[-1]}"'
        )

    return (
        "\nReturn exactly one JSON object with a single \"intent\" field whose value is "
        f"exactly one of {joined}. Do not return a bare string, Markdown, or any additional "
        "text."
    )


def build_classification_prompt(
    *, web_enabled: bool, pdf_enabled: bool
) -> ChatPromptTemplate:
    values = ["general", "rag"]

    system = _CLASSIFY_INTRO + _RAG_CATEGORY
    if web_enabled:
        system += _WEB_CATEGORY
        values.append("web_search")
    if pdf_enabled:
        system += _PDF_CATEGORY
        values.append("generate_pdf")
    system += _GENERAL_CATEGORY + _TIE_BREAK + _EXAMPLES_BASE

    if web_enabled:
        system += _WEB_EXAMPLE
    if pdf_enabled:
        system += _PDF_EXAMPLES

    system += _json_contract(values)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Classify this message:\n{user_message}"),
        ]
    )


query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Rewrite the user's latest question as a single, self-contained question.
            Use the conversation history only to resolve references (pronouns, "that",
            "it", "the previous one") and to make implied context explicit.
            Keep the user's meaning and important details. Do not answer it and do not
            add new information. If the question is already self-contained and explicit,
            return it unchanged.

            Output only the rewritten question on one line. No quotes, no JSON,
            no markdown, no explanation, no prefix.
            """.strip(),
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {user_query}"),
    ]
)
