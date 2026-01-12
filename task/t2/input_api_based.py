from enum import StrEnum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import BaseModel
from pydantic import Field, SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient

# TODO:
# Before implementation open the `api_based_grounding.png` to see the flow of app

QUERY_ANALYSIS_PROMPT = """You are a query analysis system that extracts search parameters from user questions about users.

## Available Search Fields:
- **name**: User's first name (e.g., "John", "Mary")
- **surname**: User's last name (e.g., "Smith", "Johnson") 
- **email**: User's email address (e.g., "john@example.com")

## Instructions:
1. Analyze the user's question and identify what they're looking for
2. Extract specific search values mentioned in the query
3. Map them to the appropriate search fields
4. If multiple search criteria are mentioned, include all of them
5. Only extract explicit values - don't infer or assume values not mentioned

## Examples:
- "Who is John?" → name: "John"
- "Find users with surname Smith" → surname: "Smith" 
- "Look for john@example.com" → email: "john@example.com"
- "Find John Smith" → name: "John", surname: "Smith"
- "I need user emails that filled with hiking" → No clear search parameters (return empty list)

## Response Format:
{format_instructions}
"""

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
- Be conversational and helpful in your responses.
- When presenting user information, format it clearly and include relevant details.
"""

USER_PROMPT = """## RAG CONTEXT:
{context}

## USER QUESTION: 
{query}"""


# TODO:
# 1. Create AzureChatOpenAI client
llm_client = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="",
    temperature=0.0,
)
# 2. Create UserClient
user_client = UserClient()


# TODO:
# Now we need to create pydentic models that will be user for search and their JSON schema will be passed to LLM by
# langchain. In response from LLM we expect to get response in such format (JSON by JSON Schema)
# 1. SearchField class, extend StrEnum and has constants: name, surname, email
class SearchField(StrEnum):
    NAME = "name"
    SURNAME = "surname"
    EMAIL = "email"


# 2. Create SearchRequest, extends pydentic BaseModel and has such fields:
#       - search_field (enum from above), also you can provide its `description` that will be provided with JSON Schema
#         to LLM that model will be better understand what you expect there
#       - search_value, its string, sample what we expect here is some name, surname or email to make search
class SearchRequest(BaseModel):
    search_field: SearchField = Field(
        ...,
        description="The field to search on. Can be 'name', 'surname', or 'email'.",
    )
    search_value: str = Field(
        ...,
        description="The value to search for in the specified field.",
    )


# 3. Create SearchRequests, extends pydentic BaseModel and has such fields:
#       - search_request_parameters, list of SearchRequest, by default empty list
class SearchRequests(BaseModel):
    search_request_parameters: list[SearchRequest] = Field(
        default_factory=list,
        description="A list of search request parameters extracted from the user query.",
    )


def retrieve_context(user_question: str) -> list[dict[str, Any]]:
    """Extract search parameters from user query and retrieve matching users."""
    # TODO:
    # 1. Create PydanticOutputParser with `pydantic_object=SearchRequests` as `parser`
    parser = PydanticOutputParser(pydantic_object=SearchRequests)
    # 2. Create messages array with:
    #       - use SystemMessagePromptTemplate and from template generate system message from QUERY_ANALYSIS_PROMPT
    #       - user message
    messages = [
        SystemMessagePromptTemplate.from_template(
            QUERY_ANALYSIS_PROMPT
        ).format_messages(format_instructions=parser.get_format_instructions())[0],
        HumanMessage(content=user_question),
    ]
    # 3. Generate `prompt`: `ChatPromptTemplate.from_messages(messages=messages).partial(format_instructions=parser.get_format_instructions())`
    prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
        format_instructions=parser.get_format_instructions()
    )
    # 4. Invoke it: `(prompt | llm_client | parser).invoke({})` as `search_requests: SearchRequests` (you are using LCEL)
    search_requests: SearchRequests = (prompt | llm_client | parser).invoke({})
    # 5. If `search_requests` has `search_request_parameters`:
    #       - create `requests_dict`
    #       - iterate through searched parameters and:
    #           - add to `requests_dict` the `search_request.search_field.value` as key and `search_request.search_value` as value
    #       - print `requests_dict`
    #       - search users (**requests_dict) with `user_client`
    #       - return users that you found
    if search_requests.search_request_parameters:
        requests_dict = {}
        for search_request in search_requests.search_request_parameters:
            requests_dict[search_request.search_field.value] = (
                search_request.search_value
            )
        print(f"Searching users with parameters: {requests_dict}")
        found_users = user_client.search_users(**requests_dict)
        return found_users
    # 6. Otherwise print 'No specific search parameters found!' and return empty array
    print("No specific search parameters found!")
    return []


def augment_prompt(user_question: str, context: list[dict[str, Any]]) -> str:
    """Combine user query with retrieved context into a formatted prompt."""
    # TODO:
    # 1. Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    context_strings = []
    for user in context:
        user_info = []
        for key, value in user.items():
            user_info.append(f"  {key}: {value}")
        context_strings.append("\n".join(user_info).strip())
    combined_context = "\n\n".join(context_strings).strip()
    # 2. Make augmentation for USER_PROMPT
    augmented_prompt = USER_PROMPT.format(context=combined_context, query=user_question)
    # 3. print augmented prompt
    print(augmented_prompt)
    # 4. return augmented prompt
    return augmented_prompt


def generate_answer(augmented_prompt: str) -> str:
    """Generate final answer using the augmented prompt."""
    # TODO:
    # 1. Create messages array with:
    #       - SYSTEM_PROMPT
    #       - augmented_prompt
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=augmented_prompt),
    ]
    # 2. Generate response, use invoke method with llm_client
    response = llm_client.invoke(messages)
    # 3. Return response content
    return response.content


def main():
    print("Query samples:")
    print(" - I need user emails that filled with hiking and psychology")
    print(" - Who is John?")
    print(" - Find users with surname Adams")
    print(" - Do we have smbd with name John that love painting?")

    # TODO:
    # 1. Create infinite loop
    while True:
        user_question = input("> ").strip()
        # 2. Get input from console as `user_question`
        # 2. If user types exit or quit, break the loop
        if user_question.lower() in ["quit", "exit"]:
            break
        # 3. retrieve context
        context = retrieve_context(user_question)
        # 4. if context is present:
        if context:
            #       - make augmentation
            augmented_prompt = augment_prompt(user_question, context)
            #       - generate answer with augmented prompt
            answer = generate_answer(augmented_prompt)
            print(f"Answer:\n{answer}\n")
        # 5. Otherwise print `No relevant information found`
        else:
            print("No relevant information found.\n")
    # 2. Get input from console as `user_question`
    # 3. retrieve context
    # 4. if context is present:
    #       - make augmentation
    #       - generate answer with augmented prompt
    # 5. Otherwise print `No relevant information found`


if __name__ == "__main__":
    main()


# The problems with API based Grounding approach are:
#   - We need a Pre-Step to figure out what field should be used for search (Takes time)
#   - Values for search should be correct (✅ John -> ❌ Jonh)
#   - Is not so flexible
# Benefits are:
#   - We fetch actual data (new users added and deleted every 5 minutes)
#   - Costs reduce
