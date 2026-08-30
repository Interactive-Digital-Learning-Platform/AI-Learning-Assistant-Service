from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.core.config import settings

_KB_DESCRIPTION = (
    "The knowledge base contains study materials — textbooks, lecture notes, and past "
    f"papers — for {settings.ASSISTANT_DOMAIN}."
)

_CLASSIFY_SYSTEM = (
    "You are an intent classifier for an AI study assistant.\n\n"
    f"{_KB_DESCRIPTION}\n\n"
    "Classify the user's latest message into exactly one category:\n\n"
    "- rag: anything that could be answered from the study materials above — explanations "
    "of concepts, definitions, worked problems, exam or past-paper questions, \"what is X\", "
    "\"explain Y\", \"how does Z work\", questions about a chapter or topic, or a follow-up "
    "that continues such a discussion.\n"
    "- general: greetings, small talk, thanks, and meta questions about this conversation "
    "itself (for example \"summarise what we discussed\", \"repeat that\", \"what did I just "
    "ask\").\n\n"
    "When a message could reasonably fit either category, choose rag.\n\n"
    "Examples:\n"
    "- \"Hi, can you help me?\" -> general\n"
    "- \"Thanks, that makes sense!\" -> general\n"
    "- \"Summarise what we've covered so far\" -> general\n"
    "- \"What is photosynthesis?\" -> rag\n"
    "- \"Explain Newton's second law from the motion chapter\" -> rag\n"
    "- \"Give me a past paper question on electrolysis\" -> rag\n\n"
    "Return exactly one JSON object with a single \"intent\" field whose value is either "
    "\"general\" or \"rag\". Do not return a bare string, Markdown, or any additional text."
)

_CLASSIFY_SYSTEM_WITH_WEB = (
    "You are an intent classifier for an AI study assistant.\n\n"
    f"{_KB_DESCRIPTION}\n\n"
    "Classify the user's latest message into exactly one category:\n\n"
    "- rag: anything that could be answered from the study materials above — explanations "
    "of concepts, definitions, worked problems, exam or past-paper questions, \"what is X\", "
    "\"explain Y\", \"how does Z work\", questions about a chapter or topic, or a follow-up "
    "that continues such a discussion.\n"
    "- web_search: questions that need current, real-time, or very recent information from "
    "the public web — news, latest events, prices, releases, \"today\"/\"latest\"/\"now\", "
    "or facts that clearly fall outside the study materials.\n"
    "- general: greetings, small talk, thanks, and meta questions about this conversation "
    "itself (for example \"summarise what we discussed\", \"repeat that\").\n\n"
    "Prefer rag whenever the question is about study or course material. Use web_search only "
    "when freshness or public-web coverage clearly matters. When a message could reasonably "
    "fit either general or rag, choose rag.\n\n"
    "Examples:\n"
    "- \"Hi, can you help me?\" -> general\n"
    "- \"Thanks, that makes sense!\" -> general\n"
    "- \"What is photosynthesis?\" -> rag\n"
    "- \"Explain Newton's second law from the motion chapter\" -> rag\n"
    "- \"Give me a past paper question on electrolysis\" -> rag\n"
    "- \"What is the latest version of Python?\" -> web_search\n\n"
    "Return exactly one JSON object with a single \"intent\" field whose value is exactly "
    "one of \"general\", \"rag\", or \"web_search\". Do not return a bare string, Markdown, "
    "or any additional text."
)

classification_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _CLASSIFY_SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Classify this message:\n{user_message}"),
    ]
)

classification_prompt_with_web = ChatPromptTemplate.from_messages(
    [
        ("system", _CLASSIFY_SYSTEM_WITH_WEB),
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
