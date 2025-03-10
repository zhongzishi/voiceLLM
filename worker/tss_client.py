# tts_client.py
import os
import asyncio
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI, APIConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSClient")

class TTSGenerationError(Exception):
    """自定义TTS异常"""
    pass

class TTSClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("TTS_MODEL", "tts-1")
        self.voice = os.getenv("TTS_VOICE", "alloy")
        self.timeout = 30.0
        self.max_retries = 3
        
        # 状态管理
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.interrupt_flags: dict[str, asyncio.Event] = {}

    async def _handle_retry(self, func, *args, **kwargs):
        """带重试机制的请求处理"""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except APIConnectionError:
                logger.warning(f"连接失败，第{attempt+1}次重试...")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"未知错误: {str(e)}")
                break
        raise TTSGenerationError("TTS服务不可用")

    async def stream_audio(
        self,
        session_id: str,
        text: str,
        speed: float = 1.0
    ) -> AsyncGenerator[bytes, None]:
        """流式生成音频"""
        self.interrupt_flags[session_id] = asyncio.Event()
        
        try:
            response = await self._handle_retry(
                self.client.audio.speech.create,
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="opus",  # 使用流式优化格式
                speed=speed
            )
            
            # 分块流式传输
            async for chunk in await response.aiter_bytes(chunk_size=1024):
                if self.interrupt_flags[session_id].is_set():
                    logger.info(f"会话{session_id}音频生成已中断")
                    break
                yield chunk
                
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")
            raise TTSGenerationError("音频生成失败")
        finally:
            self.cleanup_session(session_id)

    async def interrupt(self, session_id: str):
        """中断指定会话的生成"""
        if session_id in self.interrupt_flags:
            self.interrupt_flags[session_id].set()
            logger.info(f"已标记中断: {session_id}")

    def cleanup_session(self, session_id: str):
        """清理会话资源"""
        if session_id in self.interrupt_flags:
            del self.interrupt_flags[session_id]
        if session_id in self.active_tasks:
            del self.active_tasks[session_id]

    async def close(self):
        """关闭客户端"""
        await self.client.close()

# 使用示例
async def main():
    tts = TTSClient()
    
    async def play_audio():
        try:
            async for chunk in tts.stream_audio("test123", "欢迎使用智能语音系统"):
                # 此处发送音频块到前端
                print(f"收到音频块: {len(chunk)}字节")
        except TTSGenerationError as e:
            print(str(e))

    task = asyncio.create_task(play_audio())
    await asyncio.sleep(0.5)
    await tts.interrupt("test123")  # 测试中断
    await task

asyncio.run(main())