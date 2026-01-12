import asyncio
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from task._constants import API_KEY, DIAL_URL
from task.user_client import UserClient

# TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app

# TODO:
# Provide System prompt. Goal is to explain LLM that in the user message will be provide rag context that is retrieved
# based on user question and user question and LLM need to answer to user based on provided context
SYSTEM_PROMPT = """
You are a helpful assistant that provides comprehensive answers based on the provided user data.
INSTRUCTIONS:
1. Review the retrieved user data from the context
2. Use the information to answer the user's query accurately and thoroughly
3. If the context does not contain relevant information, respond with "INSUFFICIENT_DATA" and suggest possible alternatives or clarifications
OUTPUT FORMAT:
- If you can answer based on the context: Provide a detailed and informative response
- If the context is insufficient: Respond with exactly "INSUFFICIENT_DATA"
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
    result = []
    for key, value in user.items():
        result.append(f"  {key}: {value}")
    return "\n".join(result).strip()


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore: FAISS | None = None

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
            Document(page_content=format_user_document(user)) for user in all_users
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
        # 2. Iterate through document batches and create array with tasks that will generate FAISS vector stores from documents:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.afrom_documents
        tasks = []
        for batch in document_batches:
            task = FAISS.afrom_documents(
                documents=batch,
                embedding=self.embeddings,
            )
            tasks.append(task)
        # 3. Gather tasks with asyncio
        vectorstores = await asyncio.gather(*tasks)
        # 4. Create `final_vectorstore` via merge of all vector stores:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.merge_from
        final_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            final_vectorstore.merge_from(vs)
        # 6. Return `final_vectorstore`
        return final_vectorstore

    async def retrieve_context(
        self, query: str, k: int = 10, score: float = 0.1
    ) -> str:
        # TODO:
        # 1. Make similarity search:
        #    https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.similarity_search_with_relevance_scores
        relevant_docs = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score,
        )
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

    def generate_answer(self, augmented_prompt: str) -> str:
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
        response = self.llm_client.invoke(messages)
        # 3. Return response content
        return response.content


async def main():
    # TODO:
    # 1. Create AzureOpenAIEmbeddings
    #    embedding model 'text-embedding-3-small-1'
    #    I would recommend to set up dimensions as 384
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
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ["quit", "exit"]:
                break
            # TODO:
            # 1. Retrieve context
            context = await rag.retrieve_context(user_question, k=10, score=0.1)
            # 2. Make augmentation
            augmented_prompt = rag.augment_prompt(user_question, context)
            # 3. Generate answer and print it
            answer = rag.generate_answer(augmented_prompt)
            print(answer)


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce
