from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential

from app.core.config import settings

text_translator = TextTranslationClient(
    credential=AzureKeyCredential(settings.AZURE_TRANSLATOR_KEY),
    region=settings.AZURE_TRANSLATOR_REGION,
    endpoint=settings.AZURE_TRANSLATOR_ENDPOINT.rstrip("/"),
)