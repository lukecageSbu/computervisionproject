import fitz  # PyMuPDF
import io
from PIL import Image
from tqdm import tqdm
import re
from sklearn.decomposition import PCA
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from openai import OpenAI

# Initialize OpenAI API client
client = OpenAI(api_key='')

# Functions to extract images and text from PDF
def extract_images_from_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save the image
            image_path = f"{output_folder}/page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
            image.save(image_path)
            # print(f"Image saved: {image_path}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        full_text += text
    
    return full_text

def generate_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def chunk_text(text, max_tokens=500):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    chunk = ""

    for sentence in sentences:
        # Check if adding the next sentence would exceed the max token limit
        if len(chunk.split()) + len(sentence.split()) > max_tokens:
            chunks.append(chunk.strip())
            chunk = sentence
        else:
            chunk += " " + sentence

    # Add the last chunk
    if chunk:
        chunks.append(chunk.strip())
    
    return chunks

def connect_milvus():
    connections.connect(host="abc.ca-central-1.compute.amazonaws.com", port="19530")

# Main code
pdf_path = "chapter12.pdf"
output_folder = "extracted_images"

# Extract Images
extract_images_from_pdf(pdf_path, output_folder)

# Extract Text
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the text and formulae
text_chunks = chunk_text(extracted_text)

# Connect to Milvus
connect_milvus()
collection = Collection(name="textbook")
collection.drop()
# Define the schema for the collection
schema = CollectionSchema(
    fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
)

# Create the collection
collection = Collection(name="textbook", schema=schema)

ids = []
embeddings = []
texts = []
for i, text in enumerate(text_chunks):
    embedding = generate_embedding(text)
    ids.append(i)
    embeddings.append(list(embedding))
    texts.append(text)
data = [ids, embeddings, texts]
collection.insert(data)

index_param = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_param)
collection.load()