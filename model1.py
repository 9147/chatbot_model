import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


# Extracting Text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text

text = extract_text_from_pdf("CollieryControlOorder2000.pdf")

# Step 1: Remove special characters and symbols
cleaned_text = ''.join(e for e in text if (e.isalnum() or e.isspace() or e in ['.', ',', ';', ':', '(', ')']))

# Step 2: Remove extra spaces and line breaks
cleaned_text = ' '.join(cleaned_text.split())

# Step 3: Join lines
cleaned_text = cleaned_text.replace('\n', ' ')


def process_text_chunk(chunk_text, question, max_seq_length=512):
    tokenized_question = tokenizer.encode(question, add_special_tokens=True, return_tensors="pt")

    all_answers = []
    print("len=", len(chunk_text))
    for start in range(0, len(chunk_text), max_seq_length):
        end = start + max_seq_length
        chunk = chunk_text[start:end]
        print("count: ", start)
        tokenized_chunk = tokenizer.encode(chunk, add_special_tokens=True, return_tensors="pt")

        input_ids = torch.cat([tokenized_question, tokenized_chunk], dim=1)

        output = model(input_ids)
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

        # Compute confidence level based on the sum of start and end logits
        confidence = output.start_logits[0][answer_start] + output.end_logits[0][answer_end - 1]

        all_answers.append({"answer": answer, "confidence": confidence.item()})

    return all_answers


def getAnswer(question):
    # Example chunk text and question
    chunk_text = cleaned_text

    # Process the chunk with the question
    answers = process_text_chunk(chunk_text, question)

    # Sort answers by confidence level (higher confidence first)
    answers.sort(key=lambda x: x["confidence"], reverse=True)
    result=""
    # Print answers with confidence levels
    for i, answer in enumerate(answers):
        result+=f"Answer {i + 1}: {answer['answer']} (Confidence: {answer['confidence']:.2f})\n"

    return answers[0]['answer']