import os
import asyncio
import logging
from typing import AsyncGenerator, Dict, List, Optional
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, APIError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMClient")

class InterruptionRequested(Exception):
    """自定义中断异常"""
    pass

class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-1106-preview")
        self.max_retries = 3
        self.timeout = 30.0
        
        # 状态管理
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.interrupt_events: Dict[str, asyncio.Event] = {}
        self.conversation_histories: Dict[str, List[Dict]] = {}

    def _format_messages(self, session_id: str, new_prompt: str) -> List[Dict]:
        """维护对话历史上下文"""
        history = self.conversation_histories.get(session_id, [])
        
        # 当收到中断时保留修订标记
        if history and "REVISION" in history[-1].get("content", ""):
            history[-1]["content"] = new_prompt
        else:
            history.append({"role": "user", "content": new_prompt})
        
        return history

    async def handle_api_errors(self, func):
        """API错误处理装饰器"""
        async def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    return await func(*args, **kwargs)
                except APIConnectionError:
                    logger.error("API连接错误，尝试重连...")
                    await asyncio.sleep(2 ** attempt)
                except RateLimitError:
                    logger.warning("速率限制触发，等待重试...")
                    await asyncio.sleep(5)
                except APIError as e:
                    logger.error(f"API错误: {e}")
                    break
            raise RuntimeError("API请求失败")
        return wrapper

    async def stream_generation(
        self,
        session_id: str,
        prompt: str,
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """带中断控制的流式生成"""
        # 初始化中断事件
        self.interrupt_events[session_id] = asyncio.Event()
        
        # 准备消息历史
        messages = self._format_messages(session_id, prompt)
        buffer = []
        stream_enabled = True  # 始终启用流式

        try:
            @self.handle_api_errors
            async def generate():
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream_enabled,
                    timeout=self.timeout
                )

                async for chunk in response:
                    if self.interrupt_events[session_id].is_set():
                        logger.info(f"检测到中断请求: {session_id}")
                        raise InterruptionRequested()

                    content = chunk.choices[0].delta.content
                    if content:
                        buffer.append(content)
                        yield content

                # 完成生成后保存历史
                full_response = ''.join(buffer)
                self.conversation_histories.setdefault(session_id, []).append({
                    "role": "assistant",
                    "content": full_response
                })

            async for chunk in generate():
                yield chunk

        except InterruptionRequested:
            logger.info(f"成功中断生成: {session_id}")
            yield "[INTERRUPTED]"  # 发送中断信号
        finally:
            self.cleanup_session(session_id)

    async def interrupt(self, session_id: str):
        """触发中断"""
        if session_id in self.interrupt_events:
            self.interrupt_events[session_id].set()
            logger.info(f"已发送中断信号: {session_id}")

            # 保留修订上下文
            if session_id in self.conversation_histories:
                last_message = self.conversation_histories[session_id][-1]
                last_message["content"] = f"[REVISION REQUESTED] {last_message['content']}"

    def cleanup_session(self, session_id: str):
        """清理会话资源"""
        if session_id in self.interrupt_events:
            del self.interrupt_events[session_id]
        if session_id in self.active_tasks:
            del self.active_tasks[session_id]

    async def close(self):
        """清理所有资源"""
        await self.client.close()