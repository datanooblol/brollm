from typing import Any, Dict, List
import boto3

class BedrockChat:
    def __init__(
            self, 
            model_name: str = "us.meta.llama3-2-11b-instruct-v1:0",
            temperature: float = 0,
            region_name: str = "us-west-2",
            aws_access_key_id: str | None = None,
            aws_secret_access_key: str | None = None,
            aws_session_token: str | None = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

    def get_model(self, ):
        param = {
            "service_name": "bedrock-runtime",
            "region_name": self.region_name,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
        }
        return boto3.client(**param)

    def UserMessage(self, text: str, image_bytes=None, image_format: str | None = None) -> Dict[str, Any]:
        content = [{"text": text}]
        if image_bytes:
            image = {"image": {"format": image_format, "source": {"bytes": image_bytes}}}
            # content.append(image)
            content += [image]
        return {"role": "user", "content": content}

    def AIMessage(self, text: str) -> Dict[str, Any]:
        return {"role": "assistant", "content": [{"text": text}]}

    def SystemMessage(self, text: str) -> List[Dict[str, str]]:
        return [{"text": text}]

    def run(self, system_prompt: str, messages: List[Dict[str, Any]]) -> str:
        model = self.get_model()
        response = model.converse(
            modelId=self.model_name,
            messages=messages,
            system=self.SystemMessage(system_prompt),
            inferenceConfig={"temperature": self.temperature},
        )
        return response['output']['message']['content'][0]['text']
    
    def __call__(self, system_prompt:str, messages: List[Dict[str, Any]]) -> str:
        return self.run(system_prompt, messages)