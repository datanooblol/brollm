import asyncio
import httpx
import re
import time

def extract_code(text, language):
    """Extract code blocks between ```language and ```"""
    pattern = rf'```{language}\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return '\n'.join(matches)

async def ollama_code_async(prompt: str, language: str, model: str = "qwen2.5-coder:14b", base_url: str = "http://localhost:11434") -> str:
    system_prompt = f"""You are a senior {language} developer. You can do everything in {language}. 

Cautions:
- Return only {language} codeblock. Discard irrelevant conversation.

Always return your response in {language} codeblock like below example:
```{language}
<your code>
```"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f'{base_url}/api/chat', json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        })
        return response.json()['message']['content']

async def main():
    # Test prompts for different languages
    test_cases = [
        ("Create a function to calculate fibonacci numbers", "python"),
        ("Find top 5 customers by total purchase amount", "sql"),
        ("Create an async function to fetch user data from API", "typescript")
    ]
    
    print("Testing async code generation...\n")
    
    # Time the concurrent execution
    start_time = time.time()
    tasks = [ollama_code_async(prompt, lang) for prompt, lang in test_cases]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"⏱️  Total execution time: {total_time:.2f} seconds\n")
    
    # Display results
    for i, ((prompt, language), response) in enumerate(zip(test_cases, results), 1):
        print(f"=== Test {i}: {language.upper()} ===")
        print(f"Prompt: {prompt}")
        print(f"Response:\n{response}")
        
        # Extract and show clean code
        code = extract_code(response, language)
        if code:
            print(f"Extracted {language} code:")
            print(code)
        print("-" * 50)
    
    print(f"\n🚀 Completed {len(test_cases)} code generation requests in {total_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())