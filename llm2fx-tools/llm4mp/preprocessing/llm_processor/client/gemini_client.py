from typing import List, Dict, Optional
from google import genai
from pydantic import BaseModel

class ChatSchema(BaseModel):
    role: str
    content: str

class CotSchema(BaseModel):
    chain_of_thought: str

class EvaluationSchema(BaseModel):
    tool_alignment: int
    thought_quality: int

class ModelEvaluationSchema(BaseModel):
    instruction_following_quality: int
    chain_of_thought_quality: int

class GeminiClient:
    def __init__(self,api_key):
        """
        Args:
            api_key: Google API key for Gemini
        """
        self.client = genai.Client(api_key=api_key)

    def chat_completion(self,
                       message: str,
                       model_name= "gemini-2.5-flash-lite",
                       schema_type="chat",
        ):
        """
        Generate chat completion for a single message
        """
        if schema_type == "chat":
            response_schema = list[ChatSchema]
        elif schema_type == "cot":
            response_schema = CotSchema
        elif schema_type == "model_evaluation":
            response_schema = ModelEvaluationSchema
        elif schema_type == "evaluation":
            response_schema = EvaluationSchema
        response = self.client.models.generate_content(
            model=model_name,
            contents=message,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )
        return response.text
