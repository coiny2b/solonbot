# 베이스 이미지 선택
FROM python:3.12-slim

# 작업 디렉터리 설정
WORKDIR /solonbot

# 필요한 파일 복사
COPY ./main8.py /solonbot
COPY ./requirements.txt /solonbot

# 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
