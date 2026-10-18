from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings

NO_TOPIC = "NONE"

_TITLE_RULES = f"""You write the title shown in the sidebar for a conversation between a \
student and an AI study assistant for {settings.ASSISTANT_DOMAIN}.

Rules:
- 2 to 6 words in Title Case, naming the specific topic the student is asking about.
- Write it in English, even if the conversation uses another language.
- No quotation marks, no ending punctuation, no emoji, no markdown.
- Do not start with filler such as "Question About", "Help With" or "Explaining".
- If the student's message is only a greeting, thanks or small talk with no study topic, \
reply with exactly {NO_TOPIC}.

Examples:
- "What is photosynthesis?" -> Photosynthesis in Green Plants
- "how do i solve x^2 - 5x + 6 = 0" -> Solving Quadratic Equations
- "Make a PDF on the causes of World War I" -> Causes of World War I
- "Hi there!" -> {NO_TOPIC}

Reply with the title only."""

title_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _TITLE_RULES),
        (
            "human",
            "Student's message:\n{user_message}\n\nAssistant's reply:\n{assistant_reply}",
        ),
    ]
)
