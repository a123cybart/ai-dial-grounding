import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient

# TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            "total_tokens": self.total_tokens,
            "batch_count": len(self.batch_tokens),
            "batch_tokens": self.batch_tokens,
        }


# TODO:
# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
llm = AzureChatOpenAI(
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    azure_deployment="gpt-4o",
    temperature=1.0,
)
# 2. Create TokenTracker
token_tracker = TokenTracker()


def join_context(context: list[dict[str, Any]]) -> str:
    # TODO:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...

    result = []
    for user in context:
        result.append("User:")
        for key, value in user.items():
            result.append(f"  {key}: {value}")
        result.append("")  # Empty line between users
    return "\n".join(result).strip()


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    # TODO:
    # 1. Create messages array with system prompt and user message
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    response = await llm.ainvoke(messages)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    usage = response.response_metadata.get("token_usage", {})
    total_tokens = usage.get("total_tokens", 0)
    # 4. Add tokens to `token_tracker`
    token_tracker.add_tokens(total_tokens)
    # 5. Print response content and `total_tokens`
    print(f"Response: {response.content}")
    print(f"Total tokens used: {total_tokens}")
    # 6. return response content
    return response.content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        # TODO:
        # 1. Get all users (use UserClient)
        user_client = UserClient()
        all_users = user_client.get_all_users()
        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        batch_size = 100
        user_batches = [
            all_users[i : i + batch_size] for i in range(0, len(all_users), batch_size)
        ]
        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        tasks = []
        for batch in user_batches:
            context_str = join_context(batch)
            user_message = USER_PROMPT.format(context=context_str, query=user_question)
            tasks.append(generate_response(BATCH_SYSTEM_PROMPT, user_message))
        # 4. Run task asynchronously, use method `gather` form `asyncio`
        batch_results = await asyncio.gather(*tasks)
        print("\n--- Processing batch results ---")
        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        filtered_results = [
            result for result in batch_results if result != "NO_MATCHES_FOUND"
        ]
        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        if filtered_results:
            combined_results = "\n\n".join(filtered_results)
            final_user_message = f"## SEARCH RESULTS:\n{combined_results}\n\n## ORIGINAL QUERY:\n{user_question}"
            final_response = await generate_response(
                FINAL_SYSTEM_PROMPT, final_user_message
            )
            print("\n--- Final Response ---")
            print(final_response)
        else:
            print("No users found matching the search criteria.")
        # 6. Otherwise prin the info that `No users found matching`
        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
        print("\n--- Token Usage Summary ---")
        usage_summary = token_tracker.get_summary()
        print(f"Total tokens used: {usage_summary['total_tokens']}")
        print(f"Number of batches processed: {usage_summary['batch_count']}")
        print(f"Tokens used per batch: {usage_summary['batch_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation
