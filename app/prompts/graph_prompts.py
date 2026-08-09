from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

classification_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """You are an intent classification assistant.
            Your task is to classify the user's query into one of these categories:

            - general: For casual questions, explanations, greetings, or questions that do not require external knowledge retrieval.

            - rag: For questions that require retrieving information from the knowledge base, documents, textbooks, or subject materials.

            Classify the query as either "general" or "rag".
            
            Return exactly one JSON object with a required "intent" field.
            The intent value must be either "general" or "rag".
            Do not return a bare string, Markdown, or additional text.

            """,
        ),
        (
            "human",
            """
            User Query:
            {user_message}

            Conversation History:
            {history}
            """,
        ),
    ]
)

query_rewrite_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
            Rewrite the latest question as a standalone search query.
            Use history only to resolve missing context.
            Preserve meaning and important details.
            Do not answer or add information.
            If already standalone, return it unchanged.
            """.strip(),
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {user_query}"),
    ]
)
