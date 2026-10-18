from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

_KB_DESCRIPTION = (
    "The knowledge base contains study materials — textbooks, lecture notes, and past "
    f"papers — for {settings.ASSISTANT_DOMAIN}."
)

_CLASSIFY_INTRO = (
    "You analyse the latest message sent to an AI study assistant: you decide how it should "
    "be handled and, when the course material or the web will be searched, write the search "
    "queries.\n\n"
    f"{_KB_DESCRIPTION}\n\n"
    "Classify the latest message into exactly one category:\n\n"
)

_RAG_CATEGORY = (
    "- rag: anything that could be answered from the study materials above — explanations "
    "of concepts, definitions, worked problems, exam or past-paper questions, \"what is X\", "
    "\"explain Y\", \"how does Z work\", questions about a chapter or topic, or a follow-up "
    "that continues such a discussion.\n"
)

_WEB_CATEGORY = (
    "- web_search: the answer depends on current or real-time information from the public "
    "web — news, recent events, today's date, prices, this year's exam dates or "
    "announcements, latest releases, \"today\"/\"latest\"/\"now\". Facts that do not change "
    "over time — capitals, historical events, definitions, formulas, scientific values — are "
    "rag, never web_search. Never choose web_search for a study topic, even one you think "
    "the study materials may not cover: the assistant searches the web by itself when the "
    "materials have no answer.\n"
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
    "\nPrefer rag whenever the message is about a study topic, even if you are unsure the "
    "study materials cover it. When a message could reasonably fit either general or rag, "
    "choose rag.\n"
)

_EXAMPLES_BASE = (
    "\nExamples:\n"
    "- \"Hi, can you help me?\" -> general\n"
    "- \"Thanks, that makes sense!\" -> general\n"
    "- \"Summarise what we've covered so far\" -> general\n"
    "- \"What is photosynthesis?\" -> rag\n"
    "- \"Explain Newton's second law from the motion chapter\" -> rag\n"
    "- \"Give me a past paper question on electrolysis\" -> rag\n"
    "- \"Summarise the notes I uploaded\" -> rag\n"
)

_WEB_EXAMPLE = (
    "- \"What is the latest version of Python?\" -> web_search\n"
    "- \"When is this year's O/L exam?\" -> web_search\n"
    "- \"What is the boiling point of ethanol?\" -> rag\n"
    "- \"What is the capital of Australia?\" -> rag\n"
)

_PDF_EXAMPLES = (
    "- \"Make a PDF about photosynthesis\" -> generate_pdf\n"
    "- \"Create a downloadable study sheet on Newton's laws\" -> generate_pdf\n"
    "- \"Explain photosynthesis\" -> rag\n"
)

_PDF_QUERY = (
    " For generate_pdf, the topic the document should cover, written as a search query: drop "
    "the request wording (\"make a PDF\", \"worksheet\") and resolve \"this\" or \"what we "
    "discussed\" from the conversation."
)

_PDF_OUTPUT_EXAMPLE = (
    '{{"intent": "generate_pdf", "standalone_query": "Newton\'s laws of motion", '
    '"sub_queries": [], "subject": "Science"}}\n'
)


def _output_contract(values: list[str]) -> str:
    joined = ", ".join(f'"{value}"' for value in values)
    pdf_enabled = "generate_pdf" in values
    pdf_query = _PDF_QUERY if pdf_enabled else ""
    pdf_example = _PDF_OUTPUT_EXAMPLE if pdf_enabled else ""
    subjects = ", ".join(settings.KB_SUBJECTS)

    return f"""
Return exactly one JSON object with these four fields:
- "intent": exactly one of {joined}.
- "standalone_query": for rag and web_search, the latest message rewritten as one \
self-contained search query. Use the conversation only to resolve references ("it", "that", \
"the second one") and implied context. Keep the student's key terms, names, formulas, units \
and numbers exactly as written. Do not answer the question or add new facts.{pdf_query} For \
any other intent, use "".
- "sub_queries": only when the message asks about two or more distinct things that need \
separate look-ups ("compare X and Y", "differences between A and B", "explain P and Q"), one \
self-contained query per part, at most 3. Otherwise [].
- "subject": the school subject the question belongs to, written exactly as one of: \
{subjects}. Use the conversation to decide a follow-up's subject. Use null when the question \
is not clearly about one of these subjects, when it spans several of them, or for any intent \
other than rag{" and generate_pdf" if pdf_enabled else ""}.

Example outputs:
{{{{"intent": "rag", "standalone_query": "What is Newton's second law of motion?", "sub_queries": [], "subject": "Science"}}}}
{{{{"intent": "rag", "standalone_query": "Compare mitosis and meiosis", "sub_queries": ["What is mitosis?", "What is meiosis?"], "subject": "Science"}}}}
{{{{"intent": "rag", "standalone_query": "Explain ionic, covalent and metallic bonds", "sub_queries": ["What is an ionic bond?", "What is a covalent bond?", "What is a metallic bond?"], "subject": "Science"}}}}
{{{{"intent": "rag", "standalone_query": "What is a cell in a spreadsheet?", "sub_queries": [], "subject": "ICT"}}}}
{pdf_example}{{{{"intent": "general", "standalone_query": "", "sub_queries": [], "subject": null}}}}

Output only the JSON object, with no Markdown and no other text."""


def build_query_analysis_prompt(
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

    system += _output_contract(values)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{attachments_note}Latest message:\n{user_message}"),
        ]
    )


def attachments_note(names: dict[str, str]) -> str:
    filenames = list(dict.fromkeys(names.values()))

    if not filenames:
        return ""

    return (
        "Files the student uploaded earlier in this conversation: "
        f"{', '.join(filenames)}. A message that asks about, refers to, or follows up on "
        "these files is rag.\n\n"
    )
