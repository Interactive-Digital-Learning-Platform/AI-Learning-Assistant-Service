from langchain_core.prompts import ChatPromptTemplate

classification_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """You are an intent classification assistant.
            Your task is to classify the user's query into one of these categories:

            - general: For casual questions, explanations, greetings, or questions that do not require external knowledge retrieval.

            - rag: For questions that require retrieving information from the knowledge base, documents, textbooks, or subject materials.

            Return only the classified intent.""",
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
            """You are a query rewriter. Given a conversation history and a user question,
            rewrite the question to be a clear standalone search query.
            Return ONLY the rewritten query, nothing else.
            """,
        ),
        ("placeholder", "{history}"),
        ("human", "Question: {user_query}"),
    ]
)
