# Using DuckDuckGo search requires configuring the DDGS_PROXY environment variable.
import os
import logging

import json
import asyncio
import requests

from toolkit.base import BaseTool

MAX_RETRIES = 3

DESC = """Search for information using the web search engine and return the search results"""

Q_DESC = """The query used to search for information. e.g. "PopMart latest news"."""

MAX_RESULTS_DESC = """The maximum number of results to return, ranging from 1 to 20. Default is 10."""

logger = logging.getLogger(__name__)


class WebSearch(BaseTool):
    name: str = "web_search"
    description: str = DESC
    parameters: dict = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": Q_DESC},
            "max_results": {"type": "integer", "description": MAX_RESULTS_DESC},
        },
        "required": ["q"],
    }

    async def _execute(self, data):
        for i in range(MAX_RETRIES):
            try:
                results = await self._serper_search(data)
                return {"observation": results, "success": True}
            except Exception as e:
                logger.error(f"Google search error: {e}")
                logger.error(f"Retrying {i + 1}/{MAX_RETRIES}...")
        return {"observation": [], "success": False}

    async def _serper_search(self, data):
        url = "https://google.serper.dev/search"

        payload = json.dumps({
        "q": data["q"]
        })
        max_results = data.get("max_results", 10)
        headers = {
        'X-API-KEY': os.environ.get("SERPER_KEY"),
        'Content-Type': 'application/json'
        }
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(url, headers=headers, data=payload)
            )
            results = json.loads(response.text)

            results = [
                    {
                        "url": item["link"],
                        "title": item["title"],
                        "content": item["snippet"],
                    }
                    for item in results['organic'][:max_results]
                ]
        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            return []
        
        return results

    async def execute(self, **kwargs):
        return await self._execute(kwargs)
