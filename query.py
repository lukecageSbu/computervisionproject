from pymilvus import Collection, connections
from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI(api_key='')

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def connect_milvus():
    connections.connect(host="abc.ca-central-1.compute.amazonaws.com", port="19530")

def query_milvus(text):
    query_embedding = generate_embedding(text)
    collection = Collection(name="textbook")
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "L2"},
        limit=1,
        output_fields=["id", "text"]
    )
    return results

def format_context(results):
    context = ""
    collection = Collection(name="textbook")
    for result in results:
        for hit in result:
            expr = f"id == {hit.id}"
            entities = collection.query(expr, output_fields=["text"])  # Only retrieve the 'text' field
            for entity in entities:
                context += entity.get("text", "") + "\n"
    return context.strip()  # Remove trailing newline if necessary

def generate_response_with_context(context, question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI Tutor, a graduate student teaching a Computer Vision course. Use the provided context to answer questions accurately."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
        ],
        temperature=0,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

def main():
    connect_milvus()
    
    question = "Define stereo vision"
    # Query Milvus and retrieve context
    results = query_milvus(question)
    context = format_context(results)
    
    # Generate a response using the context

    answer = generate_response_with_context(context, question)
    print("Question:",question)
    print("Generated Answer:", answer)
    print('*'*100)
if __name__ == "__main__":
    main()