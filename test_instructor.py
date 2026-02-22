import asyncio
import os
from pydantic import BaseModel
from openai import AsyncOpenAI
import instructor

class User(BaseModel):
    name: str
    age: int

async def main():
    client = instructor.from_openai(AsyncOpenAI(
        api_key="sk-598176acad634e448c5a5071928669b0",
        base_url="https://api.deepseek.com/v1"
    ))
    try:
        user = await client.chat.completions.create(
            model="deepseek-reasoner",
            response_model=User,
            messages=[{"role": "user", "content": "Extract Jason is 25 years old"}]
        )
        print(user)
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
