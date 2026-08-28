import logging

from llama_cloud import AsyncLlamaCloud

from app.core.config import settings

logger = logging.getLogger(__name__)


class AttachmentIngestionService:

    def __init__(self):
        self.client = AsyncLlamaCloud(api_key=settings.LLAMA_CLOUD_API_KEY)


    async def extract_text(self, filename: str, content: bytes, content_type: str) -> tuple[str, str]:
        
        file = await self.client.files.create(
            file=(filename, content, content_type),
            purpose="parse"
        )

        result = await self.client.parsing.parse(
            file_id=file.id,
            tier="agentic",
            processing_options={
                "cost_optimizer": {
                    "enable": True
                }
            },
            version="latest",
            output_options={
                "markdown":{
                    "annotate_links": True,
                    "tables":{
                        "merge_continued_tables": True, 
                        "output_tables_as_markdown": True
                    },
                }
            },
            expand=["markdown_full"]
        )

        if result.markdown_full is None:
            logger.error("Parse result had no markdown for file_id=%s", file.id)
            return "", "llama_parse"


        return result.markdown_full, "llama_parse"
        