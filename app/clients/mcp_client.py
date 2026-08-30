import logging
from asyncio import wait_for

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection

logger = logging.getLogger(__name__)


class MCPClientManager:
    SERVER_NAME = "learning_assistant"

    def __init__(
        self,
        *,
        url: str | None,
        auth_token: str | None,
        timeout_seconds: float = 20.0,
        discovery_timeout_seconds: float = 10.0,
    ):
        self._enabled = bool(url and auth_token)
        self._discovery_timeout = discovery_timeout_seconds
        self._tools: dict[str, BaseTool] = {}

        if url and auth_token:
            connection: StreamableHttpConnection = {
                "transport": "streamable_http",
                "url": url,
                "headers": {"Authorization": f"Bearer {auth_token}"},
                "timeout": timeout_seconds,
            }
            self._client: MultiServerMCPClient | None = MultiServerMCPClient(
                {self.SERVER_NAME: connection}
            )
        else:
            self._client = None
            logger.warning(
                "MCP client disabled — MCP_SERVER_URL or MCP_SERVER_AUTH_TOKEN not set"
            )

    async def discover(self) -> None:
        if self._client is None:
            return

        try:
            tools = await wait_for(
                self._client.get_tools(server_name=self.SERVER_NAME),
                timeout=self._discovery_timeout,
            )
        except Exception:
            logger.exception(
                "MCP tool discovery failed; MCP-backed capabilities are unavailable"
            )
            return

        self._tools = {tool.name: tool for tool in tools}
        logger.info("MCP tools discovered: %s", sorted(self._tools))

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    async def aclose(self) -> None:
        self._client = None
        self._tools = {}
