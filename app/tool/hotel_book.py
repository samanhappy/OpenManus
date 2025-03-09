import aiohttp
import json
import os
import time
import traceback
import uuid
from typing import Dict, Any, List
from pathlib import Path
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from app.logger import logger

from app.tool.base import BaseTool

# Get the absolute path to the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv()

SEARCH_AGENT_PROMPT_TEMPLATE = """
你是一个通用智能网络数据探索工具。你的目标是通过递归访问各种格式的数据（包括JSON-LD、YAML等），找到用户需要的信息、API，以完成指定的任务。

## 当前任务
{task_description}

## 重要说明
1. 你将收到一个起始URL({initial_url})，这是一个搜索智能体的描述文件
2. 你需要理解这个搜索智能体的结构、功能和API用法
3. 你需要像网络爬虫一样，不断从中发现并访问新的URL和API端点
4. 你可以使用fetch_url工具来获取任何URL的内容
5. 该工具可以处理多种格式的响应，包括：
   - JSON格式：将直接解析为JSON对象
   - YAML格式：将返回文本内容，你需要分析其结构
   - 其他文本格式：将返回原始文本内容
6. 阅读每个文档，寻找与任务相关的信息或API端点
7. 你需要自己决定爬取路径，不要等待用户指示

## 爬取策略
1. 首先获取初始URL内容，理解搜索智能体的结构和API
2. 识别文档中的所有URL和链接，特别是serviceEndpoint、url、@id等字段
3. 分析API文档，理解API的使用方法、参数和返回值
4. 根据API文档，构造合适的请求找到所需信息
5. 记录你访问过的所有URL，避免重复爬取
6. 总结发现的所有相关信息，提供详细的建议

## 工作流程
1. 获取起始URL内容，理解搜索智能体的功能
2. 分析内容，找出所有可能的链接和API文档
3. 解析API文档，理解API的使用方法
4. 根据任务需求，构造请求获取所需信息
5. 继续探索相关链接，直到找到足够的信息
6. 总结信息，提供最适合用户的建议

## JSON-LD数据解析提示
1. 注意@context字段，它定义了数据的语义上下文
2. @type字段表示实体类型，帮助你理解数据的含义
3. @id字段通常是一个可以进一步访问的URL
4. 寻找serviceEndpoint、url等字段，它们通常指向API或更多数据

提供详细信息和清晰解释，让用户理解你找到的信息和你的推荐理由。
"""

# 全局变量
progress_list = []
visited_urls = set()
initial_url = "https://agent-search.ai/ad.json"

# 初始化 Azure OpenAI 客户端
client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

# 定义可用的工具
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch content from a URL, supporting various formats like JSON, YAML, and plain text",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch content from",
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method to use (GET or POST)",
                        "enum": ["GET", "POST"],
                    },
                    "data": {
                        "type": "object",
                        "description": "Data to send in the request body (for POST requests)",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional headers to send with the request",
                    },
                },
                "required": ["url"],
            },
        },
    }
]


def add_progress_step(
    step_id: str, title: str, status: str = "pending", details: Dict = None
) -> Dict:
    """添加一个新的处理步骤"""
    step = {"id": step_id, "title": title, "status": status, "timestamp": time.time()}
    if details:
        step["details"] = details
    progress_list.append(step)
    logger.info(f"Added progress step: {step}")
    return step


def update_progress(step_id: str, status: str) -> None:
    """更新进度状态"""
    for step in progress_list:
        if step["id"] == step_id:
            step["status"] = status
            step["timestamp"] = time.time()
            logger.info(f"Updated progress step {step_id} to {status}: {step}")
            return


async def process_http_response(
    response: aiohttp.ClientResponse, url: str
) -> Dict[str, Any]:
    """处理HTTP响应"""
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()

    if "application/json" in content_type or url.endswith(".json"):
        return await response.json()
    elif (
        "application/x-yaml" in content_type
        or "text/yaml" in content_type
        or url.endswith((".yaml", ".yml"))
    ):
        text = await response.text()
        return {"content_type": "yaml", "text_content": text, "url": url}
    else:
        text = await response.text()
        return {"content_type": "text/plain", "text_content": text, "url": url}


async def fetch_url_content(
    url: str, method: str = "GET", data: Dict = None, headers: Dict = None
) -> Dict[str, Any]:
    """获取URL内容"""
    logger.info(f"Fetching document from URL: {url} with method: {method}")
    try:
        # Create a ClientSession with SSL verification disabled for testing
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            request_kwargs = {}
            if headers:
                request_kwargs["headers"] = headers
            if data and method == "POST":
                request_kwargs["json"] = data

            if method == "GET":
                async with session.get(url, **request_kwargs) as response:
                    return await process_http_response(response, url)
            elif method == "POST":
                async with session.post(url, **request_kwargs) as response:
                    return await process_http_response(response, url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
    except Exception as e:
        logger.error(f"Error fetching URL {url} with method {method}: {str(e)}")
        raise


async def handle_tool_call(tool_call: Any, messages: List[Dict]) -> None:
    """处理工具调用"""
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    if function_name == "fetch_url":
        url = function_args.get("url")
        method = function_args.get("method", "GET")
        data = function_args.get("data")
        headers = function_args.get("headers")

        if url in visited_urls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {
                            "error": f"你已经访问过这个URL: {url}",
                            "suggestion": "请尝试访问不同的URL，或者基于已有信息提供总结和建议。",
                        }
                    ),
                }
            )
            return

        random_id = str(uuid.uuid4())[:8]
        add_progress_step(
            f"fetch_url_{random_id}", f"获取URL内容: {url}", "in-progress", {"url": url}
        )

        try:
            result = await fetch_url_content(
                url, method=method, data=data, headers=headers
            )
            logger.info(f"HTTP response [url: {url}]:")
            logger.info(result)

            visited_urls.add(url)
            update_progress(f"fetch_url_{random_id}", "completed")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            update_progress(f"fetch_url_{random_id}", "error")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": f"Failed to fetch URL: {url}", "message": str(e)}
                    ),
                }
            )


async def process_user_input(
    user_input: str, task_type: str = "hotel_booking"
) -> Dict[str, Any]:
    """处理用户输入"""
    logger.info(f"Starting to process user input for task type: {task_type}")

    try:
        # 重置状态
        progress_list.clear()
        visited_urls.clear()

        # 添加初始步骤
        add_progress_step("create_agent", f"启动{task_type}搜索助手", "in-progress")
        add_progress_step(
            "fetch_initial_url",
            "获取搜索智能体描述",
            "in-progress",
            {"url": initial_url},
        )

        # 获取初始URL内容
        initial_content = await fetch_url_content(initial_url)
        visited_urls.add(initial_url)
        update_progress("fetch_initial_url", "completed")

        # 准备消息列表
        messages = prepare_initial_messages(user_input, initial_content)

        # 处理对话
        result = await process_conversation(messages, task_type)
        return result

    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}")
        logger.error(traceback.format_exc())
        for step in progress_list:
            if step["status"] == "in-progress":
                step["status"] = "error"
        raise e


def prepare_initial_messages(user_input: str, initial_content: Dict) -> List[Dict]:
    """准备初始消息列表"""
    formatted_prompt = SEARCH_AGENT_PROMPT_TEMPLATE.format(
        task_description=user_input, initial_url=initial_url
    )

    return [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": user_input},
    ]


async def process_conversation(messages: List[Dict], task_type: str) -> Dict[str, Any]:
    """处理对话流程"""
    max_iterations = 15
    current_iteration = 0

    while current_iteration < max_iterations:
        current_iteration += 1
        logger.info(f"Starting iteration {current_iteration}/{max_iterations}")

        logger.info(f"Current messages: {messages}")

        # 获取模型响应
        completion = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL"),
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
        )

        response_message = completion.choices[0].message
        logger.info(f"Model response: {response_message}")
        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": response_message.tool_calls,
            }
        )

        # 检查是否结束对话
        if should_end_conversation(response_message, current_iteration, max_iterations):
            update_progress("create_agent", "completed")
            return create_final_response(response_message, task_type)

        # 处理工具调用
        for tool_call in response_message.tool_calls:
            await handle_tool_call(tool_call, messages)


def should_end_conversation(
    response_message: Any, current_iteration: int, max_iterations: int
) -> bool:
    """判断是否应该结束对话"""
    if not response_message.tool_calls:
        return True

    if current_iteration >= max_iterations - 1:
        return True

    return False


def create_final_response(response_message: Any, task_type: str) -> Dict[str, Any]:
    """创建最终响应"""
    return {
        "content": response_message.content,
        "type": "text",
        "visited_urls": list(visited_urls),
        "task_type": task_type,
    }


class HotelBook(BaseTool):
    name: str = "hotel_book"
    description: str = """Search and book a hotel room based on location, date, and duration.
Use this tool when you need to find and book hotel accommodations.
The tool returns details about the selected hotel and room.
"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "coordinates": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate of the hotel location",
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate of the hotel location",
                    },
                },
                "required": ["longitude", "latitude"],
                "description": "(required) Geographic coordinates for hotel search",
            },
            "check_in_date": {
                "type": "string",
                "description": "(required) Check-in date in YYYY-MM-DD format",
            },
            "duration": {
                "type": "integer",
                "description": "(required) Duration of stay in nights",
            },
            "location_name": {
                "type": "string",
                "description": "(optional) Name of the city or area for hotel search",
            },
        },
        "required": ["coordinates", "check_in_date", "duration"],
    }

    async def execute(
        self,
        coordinates: Dict[str, float],
        check_in_date: str,
        duration: int,
        location_name: str = None,
    ) -> Dict[str, Any]:
        """
        Search for hotels and book a room based on the provided criteria.

        Args:
            coordinates (Dict[str, float]): Geographic coordinates with longitude and latitude.
            check_in_date (str): Check-in date in YYYY-MM-DD format.
            duration (int): Duration of stay in nights.
            location_name (str, optional): Name of the city or area.

        Returns:
            Dict[str, Any]: Details of the selected hotel and room booking.
        """
        try:
            # Format user input for the AI agent
            location_str = (
                location_name
                or f"coordinates ({coordinates['longitude']}, {coordinates['latitude']})"
            )
            task_description = f"我需要预订{location_str}的一个酒店：{check_in_date}，{duration}天的酒店。请一步步处理：第一步，你自己选择一个不错的酒店，第二步，帮我选择一个房间。最后告诉我你选择的详细信息"

            # Process the request using the AI agent
            result = await process_user_input(task_description, "hotel_booking")

            # Return the AI agent's response
            return result

        except Exception as e:
            logger.error(f"Error booking hotel: {str(e)}")
            return {"error": f"Failed to book hotel: {str(e)}"}
