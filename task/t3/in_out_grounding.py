import asyncio
import json
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import BaseModel, Field, SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient

# TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

# TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """
You are a Hobbies Searching Wizard that performs Named Entity Extraction (NEE) to identify hobbies from user queries and match them with users.

INSTRUCTIONS:
1. You will receive user data containing user IDs and their "about_me" sections from a RAG context
2. Analyze the user's query to extract hobby-related entities (e.g., "mountains" → rock climbing, hiking, camping)
3. Match users from the context whose hobbies align with the extracted entities
4. Group users by their specific hobbies

OUTPUT FORMAT:
Return ONLY a valid JSON object with hobbies as keys and arrays of user IDs as values:
{
  "hobby_name": [user_id1, user_id2, ...],
  "another_hobby": [user_id3, user_id4, ...]
}

RULES:
- Extract user IDs ONLY from the provided context - do NOT invent or hallucinate user IDs
- Group users by specific hobby categories based on their "about_me" descriptions
- If no relevant users found in context, return an empty JSON object: {}
- Output must be valid JSON format only, no additional text or explanations
"""

# TODO:
# Should consist retrieved context and user question
USER_PROMPT = """
## USER DATA:
{context}
## USER QUERY: 
{query}
"""


def format_user_document(user: dict[str, Any]) -> str:
    # TODO:
    # Prepare context from users JSONs in the same way as in `no_grounding.py` `join_context` method (collect as one string)
    # Only include id and about_me to reduce context window
    result = []
    for key in [
        "id",
        "about_me",
    ]:
        if key in user:
            result.append(f"  {key}: {user[key]}")
    return "\n".join(result).strip()


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore: Chroma | None = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        # TODO:
        # 1. Get all users (use UserClient)
        user_client = UserClient()
        all_users = user_client.get_all_users()
        print(f"✅ Fetched {len(all_users)} users.")
        print("🗂 Preparing vectorstore with user data...")
        # 2. Prepare array of Documents where page_content is `format_user_document(user)` (you need to iterate through users)
        documents = [
            Document(
                page_content=format_user_document(user),
                id=str(user["id"]),
            )
            for user in all_users
        ]
        # 3. call `_create_vectorstore_with_batching` (don't forget that its async) and setup it as obj var `vectorstore`
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(
        self, documents: list[Document], batch_size: int = 100
    ):
        # TODO:
        # 1. Split all `documents` on batches (100 documents in 1 batch). We need it since Embedding models have limited context window
        document_batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        # 2. Create Chroma vectorstore with first batch:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.afrom_documents
        vectorstore = await Chroma.afrom_documents(
            documents=document_batches[0],
            embedding=self.embeddings,
        )
        # 3. Add remaining batches to the vectorstore:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
        for batch in document_batches[1:]:
            await vectorstore.aadd_documents(documents=batch)
        # 4. Return vectorstore
        return vectorstore

    async def refresh_users(self):
        """
        Refresh the vectorstore by comparing current API users with stored users.
        Add new users and remove deleted users.
        """
        print("🔄 Refreshing user data...")
        # 1. Get all current users from API
        user_client = UserClient()
        current_users = user_client.get_all_users()
        current_user_ids = {str(user["id"]) for user in current_users}

        # 2. Get all IDs currently in vectorstore
        # https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
        vectorstore_data = self.vectorstore.get()
        stored_user_ids = (
            set(vectorstore_data["ids"]) if vectorstore_data["ids"] else set()
        )

        # 3. Identify new and deleted users
        new_user_ids = current_user_ids - stored_user_ids
        deleted_user_ids = stored_user_ids - current_user_ids

        # 4. Remove deleted users
        if deleted_user_ids:
            print(f"🗑️  Removing {len(deleted_user_ids)} deleted users...")
            # https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
            self.vectorstore.delete(ids=list(deleted_user_ids))

        # 5. Add new users
        if new_user_ids:
            print(f"➕ Adding {len(new_user_ids)} new users...")
            new_users = [
                user for user in current_users if str(user["id"]) in new_user_ids
            ]
            new_documents = [
                Document(
                    page_content=format_user_document(user),
                    metadata={"user_id": str(user["id"])},
                    id=str(user["id"]),
                )
                for user in new_users
            ]
            # https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
            await self.vectorstore.aadd_documents(documents=new_documents)

        print(
            f"✅ Vectorstore refreshed: {len(new_user_ids)} added, {len(deleted_user_ids)} removed"
        )

    async def retrieve_context(
        self, query: str, k: int = 10, score: float = 0.1
    ) -> str:
        # Refresh vectorstore to ensure up-to-date data
        await self.refresh_users()

        # TODO:
        # 1. Make similarity search:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.asimilarity_search_with_relevance_scores
        all_docs = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query,
            k=k,
        )
        # Filter by score threshold manually (Chroma doesn't support score_threshold parameter)
        relevant_docs = [
            (doc, score_val) for doc, score_val in all_docs if score_val >= score
        ]
        # 2. Create `context_parts` empty array (we will collect content here)
        context_parts = []
        # 3. Iterate through retrieved relevant docs (pay attention that its tuple (doc, relevance_score)) and:
        #       - add doc page content to `context_parts` and then print score and content
        for doc, relevance_score in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"Score: {relevance_score}\nContent:\n{doc.page_content}\n")
        # 4. Return joined context from `context_parts` with `\n\n` spliterator (to enhance readability)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        # TODO: Make augmentation for USER_PROMPT via `format` method
        augmented_prompt = USER_PROMPT.format(context=context, query=query)
        return augmented_prompt

    async def generate_answer(self, augmented_prompt: str) -> str:
        # TODO:
        # 1. Create messages array with:
        #       - system prompt
        #       - user prompt
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        # 2. Generate response
        #    https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.azure.AzureChatOpenAI.html#langchain_openai.chat_models.azure.AzureChatOpenAI.invoke
        response = await self.llm_client.ainvoke(messages)
        # 3. Return response content
        return response.content

    async def ground_output(self, llm_response: str) -> dict[str, list[dict[str, Any]]]:
        """
        Perform output grounding by verifying user IDs and fetching full user info.
        Replaces user IDs with complete user data from the API.
        """

        try:
            # Parse the JSON response from LLM
            hobby_user_ids = json.loads(llm_response)
        except json.JSONDecodeError:
            print("⚠️  Failed to parse LLM response as JSON")
            return {}

        user_client = UserClient()
        grounded_response = {}

        # Iterate through each hobby and its user IDs
        for hobby, user_ids in hobby_user_ids.items():
            grounded_users = []

            # Fetch full info for each user ID
            for user_id in user_ids:
                try:
                    # Verify user exists and get full info
                    user_info = await user_client.get_user(user_id)
                    if user_info:
                        grounded_users.append(user_info)
                    else:
                        print(f"⚠️  User ID {user_id} not found in API")
                except Exception as e:
                    print(f"⚠️  Error fetching user {user_id}: {e}")

            # Only include hobby if we found valid users
            if grounded_users:
                grounded_response[hobby] = grounded_users

        return grounded_response


async def main():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small-1",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        dimensions=384,
    )
    llm_client = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="",
        temperature=0.0,
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ["quit", "exit"]:
                break
            # TODO:
            # 1. Retrieve context
            context = await rag.retrieve_context(user_question, k=10, score=0.1)
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            # 3. Generate answer (returns JSON with hobby: [user_ids])
            answer = await rag.generate_answer(augmented_prompt)
            print("\n📋 LLM Response (hobbies with user IDs):")
            print(answer)
            # 4. Ground output: verify IDs and fetch full user info
            print(
                "\n🔍 Grounding output (verifying IDs and fetching full user data)..."
            )
            grounded_result = await rag.ground_output(answer)
            print("\n✅ Final Grounded Response:")

            print(json.dumps(grounded_result, indent=2))


asyncio.run(main())
