from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import requests
import re
from bs4 import BeautifulSoup
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Summary API")

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Missing required environment variables")

class URLExtractor:
    @staticmethod
    def extract_url(text: str) -> Optional[str]:
        """Extract URL from text."""
        url_pattern = r'(https?://[^\s]+)'
        urls = re.findall(url_pattern, text)
        return urls[0] if urls else None

    @staticmethod
    def extract_youtube_id(url: str) -> Optional[str]:
        """Extract YouTube ID from URL."""
        try:
            query = urlparse(url)
            if query.hostname == 'youtu.be':
                return query.path[1:]
            elif query.hostname in ('www.youtube.com', 'youtube.com'):
                if query.path == '/watch':
                    return parse_qs(query.query)['v'][0]
                elif query.path.startswith(('/embed/', '/v/')):
                    return query.path.split('/')[2]
            return None
        except Exception as e:
            logger.error(f"Error extracting YouTube ID: {e}")
            return None

class TextProcessor:
    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)

    @staticmethod
    def format_transcript(transcript: list) -> str:
        """Convert YouTube transcript to formatted text."""
        return " ".join(item.get('text', '') for item in transcript)

class TextSummarizer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        self.summary_prompt = PromptTemplate(
            template="아래의 내용을 요약해줘. 요약된 내용은 3개의 <p>를 사용해서 요약해줘.: {text}",
            input_variables=["text"]
        )
        # Updated to use RunnablePassthrough instead of deprecated LLMChain
        self.chain = self.summary_prompt | self.llm

    async def summarize(self, text: str) -> str:
        """Summarize text using Gemini model."""
        try:
            response = await self.chain.ainvoke({"text": text})
            return response.content
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            raise HTTPException(status_code=500, detail="Summarization failed")

class YouTubeTranscriptExtractor:
    @staticmethod
    async def get_transcript(video_id: str) -> str:
        """Fetch transcript from YouTube video using youtube_transcript_api."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            return " ".join(item['text'] for item in transcript_list)
        except Exception as e:
            logger.error(f"Transcript extraction error: {e}")
            try:
                # Fallback to English if Korean is not available
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                return " ".join(item['text'] for item in transcript_list)
            except Exception as e:
                logger.error(f"Fallback transcript extraction error: {e}")
                raise HTTPException(status_code=500, detail="Failed to fetch transcript")

@app.post("/reply")
async def reply_message(request: Request):
    try:
        data = await request.json()
        room = data.get("room")
        msg = data.get("msg")
        sender = data.get("sender")

        logger.info(f"Received message - Room: {room}, Sender: {sender}, Message: {msg}")

        # URL 추출
        url_extractor = URLExtractor()
        url = url_extractor.extract_url(msg)
        
        # URL이 없는 일반 메시지는 빈 응답을 반환
        if not url:
            return JSONResponse(
                content={"response": ""},
                status_code=200
            )
            
        # URL이 있는 경우에만 처리
        text_processor = TextProcessor()
        summarizer = TextSummarizer()
        transcript_extractor = YouTubeTranscriptExtractor()

        try:
            youtube_id = url_extractor.extract_youtube_id(url)
            if youtube_id:
                # Handle YouTube video using youtube_transcript_api
                text = await transcript_extractor.get_transcript(youtube_id)
            else:
                # Handle regular webpage
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                text = text_processor.extract_text_from_html(response.text)

            summary = await summarizer.summarize(text)
            clean_summary = text_processor.extract_text_from_html(summary)

            return JSONResponse(
                content={"response": f"{sender}에게 답장합니다: \n {clean_summary}"},
                status_code=200
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch content")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)