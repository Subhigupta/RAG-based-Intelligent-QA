import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM  

def distance_to_similarity(distance):
    return 1 / (1 + distance) 

def run_rag_pipeline(vstore, tokenizer, model, query):

    # Find the similiar chunks from the database
    searchDocs = vstore.similarity_search_with_score(query, k=3)
    for doc, score in searchDocs:
        print("Distance:", score, "| Content:", doc.page_content[:100])

    # for doc, score in searchDocs:
    #     sim = distance_to_similarity(score)
    #     print(f"Similarity: {sim:.4f} | Distance: {score:.2f}")  # lower is better
    #     print(f"Content: {doc.page_content[:200]}...\n")

    docs_with_sim = [(doc, distance_to_similarity(score)) for doc, score in searchDocs]
    threshold = 0.005
    filtered_docs = [doc for doc, sim in docs_with_sim if sim > threshold]

    if not filtered_docs:
        return "The provided documents do not contain any relevant information for your question."

    
    # Create the prompt
    context_text = "\n\n".join([doc.page_content for doc, score in searchDocs])
    print("Context Text:", context_text)
    # context_text = "\n\n".join([doc.page_content for doc in searchDocs])
    # prompt = f"""Based on the following context, please answer the question. Answer the question in descriptive way 
    #              atleast in 4-5 lines.
    #              Context: {context_text}
    #              Question: {query}
    #              Answer:"""
    
    prompt = f"""INSTRUCTIONS:
                Answer the users QUESTION using the CONTEXT text below.
                Keep your answer ground in the facts of the CONTEXT.
                Keep your answer descriptive MAXIMUM of 5 lines.
                Write your answer in complete sentences and ensure it ends properly.
                If the CONTEXT does not contain any relevant facts to answer the QUESTION, strictly return "NONE" and nothing else.

                CONTEXT:
                {context_text}

                QUESTION:
                {query}

                ANSWER:"""
    
    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens = 100, min_new_tokens = 50, do_sample = True,
                            temperature = 0.7, top_p = 0.9, pad_token_id = tokenizer.eos_token_id,
                            stop_strings = ["\n\nQuestion:", "\nQuestion:", "Question:"],
                            tokenizer = tokenizer)
    
    # Extract just the answer
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response[len(prompt):].strip()
    print("Answer:", answer)
    
    stop_patterns = [
    "\nContext:",
    "\nQuestion:", 
    "\n\nQuestion:",
    "\nQ:",
    "Context:",
    "Question:",
    "\n\n\n"]

    for pattern in stop_patterns:
        if pattern in answer:
            answer = answer.split(pattern)[0].strip()
            break
    
    return answer