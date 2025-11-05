"""Interactive chatbot for querying the vector store."""
import os
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Config
from app.vectorstore import Vectorstore


def run_chatbot():
    """Run the interactive chatbot for querying the vector store."""
    print("🤖 Starting Soccer RAG Chatbot...")
    print("=" * 60)
    
    # Get configuration
    index_name = Config.get_pinecone_index_name()
    default_k = Config.get_default_k()
    pinecone_api_key = Config.get_pinecone_api_key()
    openai_api_key = Config.get_openai_api_key()

    # Initialize vector store
    print("\n[1/2] Initializing vector store connection...")
    vectorstore = Vectorstore(
        index_name=index_name,
        api_key=pinecone_api_key,
    )
    
    # Initialize embeddings model
    print("✓ Pinecone API key found")
    print("✓ OpenAI API key found")
    print(f"✓ Index name: {index_name}")
    
    embeddings_model = OpenAIEmbeddings(
        model=Config.get_embeddings_model_name(),
        openai_api_key=openai_api_key,
    )
    
    print("\n[2/3] Connecting to vector store...")
    vectorstore.initialize_vectorstore(embeddings_model, show_progress=False)
    print("✓ Vector store connected successfully")
    
    # Initialize LLM for answer generation
    print("\n[3/3] Initializing LLM for answer generation...")
    llm = ChatOpenAI(
        model=Config.get_llm_model_name(),
        temperature=Config.get_llm_temperature(),
        api_key=openai_api_key,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(k=default_k)
    
    # Create prompt template with conversation history support
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about Premier League annual reports.

Use ONLY the following context from the Premier League annual reports to answer the question. 
If the answer is not in the context, say "I don't have enough information in the provided documents to answer this question."

Context from Premier League documents:
{context}

Answer based on the context above. Provide a clear, concise, and well-structured answer."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    # Format documents for context using LCEL RunnableLambda
    format_docs = RunnableLambda(
        lambda docs: "\n\n".join(doc.page_content for doc in docs)
    )
    
    # Initialize chat history for session memory
    chat_history = InMemoryChatMessageHistory()
    
    # Create function to get formatted chat history
    def get_chat_history(inputs: dict) -> list:
        """Get chat history formatted for the prompt."""
        return chat_history.messages
    
    # Create RAG chain using LCEL (LangChain Expression Language)
    # This uses idiomatic LCEL patterns:
    # - Dict automatically becomes RunnableParallel for parallel execution
    # - Pipe operator (|) chains runnables together
    # - RunnableLambda for custom transformations
    # - Chat history for conversation context
    def create_qa_chain(retriever_instance):
        # Create the chain using LCEL pipe operator
        # The dict syntax automatically creates a RunnableParallel for combining inputs
        chain = (
            {
                "context": retriever_instance | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(get_chat_history),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain
    
    qa_chain = create_qa_chain(retriever)
    
    print("✓ LLM initialized successfully")
    
    print("\n" + "=" * 60)
    print("💬 Chatbot ready! Type your questions below.")
    print("Commands:")
    print("  - Type 'exit' or 'quit' to exit")
    print("  - Type 'clear' to clear the screen")
    print("  - Type 'k=<number>' to change number of results (default: 4)")
    print("=" * 60)
    print()
    
    k = default_k
    
    while True:
        try:
            # Get user query
            query = input("🔍 Your question: ").strip()
            
            # Handle empty queries
            if not query:
                continue
            
            # Handle exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Thanks for using Soccer RAG Chatbot!")
                break
            
            # Handle clear command
            if query.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            
            # Handle k setting
            if query.lower().startswith('k='):
                try:
                    new_k = int(query.split('=')[1].strip())
                    if new_k > 0:
                        k = new_k
                        print(f"✓ Number of results set to {k}")
                    else:
                        print("⚠️  Number of results must be greater than 0")
                except ValueError:
                    print("⚠️  Invalid format. Use 'k=<number>' (e.g., 'k=5')")
                continue
            
            # Generate answer using RAG chain
            print(f"\n🔎 Searching and generating answer...")
            print("-" * 60)
            
            try:
                # Update retriever k if changed
                if k != default_k:
                    retriever = vectorstore.as_retriever(k=k)
                    qa_chain = create_qa_chain(retriever)
                    default_k = k
                
                # Get source documents first (for display)
                source_docs = retriever.invoke(query)
                
                # Get answer from RAG chain (includes previous chat history automatically)
                # The current question is passed separately, not in history
                answer = qa_chain.invoke(query)
                
                # Store conversation in history for future context
                # (after getting answer, so history contains previous turns only)
                chat_history.add_user_message(query)
                chat_history.add_ai_message(answer)
                
                print(f"\n💡 Answer:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
                # Display source information (deduplicated by document)
                if source_docs:
                    # Group documents by source file and collect page numbers
                    source_map = {}
                    for doc in source_docs:
                        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                        source_file = metadata.get('file_name', metadata.get('source_file', 'Unknown'))
                        page_number = metadata.get('page_number')
                        
                        if source_file not in source_map:
                            source_map[source_file] = set()
                        
                        if page_number is not None:
                            source_map[source_file].add(page_number)
                    
                    # Display unique sources
                    unique_sources = list(source_map.items())[:5]  # Show top 5 unique documents
                    print(f"\n📚 Sources ({len(unique_sources)} unique document(s) from {len(source_docs)} chunks):")
                    for i, (source_file, pages) in enumerate(unique_sources, 1):
                        file_name = Path(source_file).name
                        if pages:
                            # Sort pages and format
                            sorted_pages = sorted(pages)
                            if len(sorted_pages) == 1:
                                page_info = f"Page {sorted_pages[0]}"
                            elif len(sorted_pages) <= 3:
                                page_info = f"Pages {', '.join(map(str, sorted_pages))}"
                            else:
                                page_info = f"Pages {sorted_pages[0]}-{sorted_pages[-1]}"
                            print(f"  {i}. {file_name} ({page_info})")
                        else:
                            print(f"  {i}. {file_name}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error generating answer: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Soccer RAG Chatbot!")
            break
        except EOFError:
            print("\n\n👋 Goodbye! Thanks for using Soccer RAG Chatbot!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.\n")



