from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

def text_classification(input_text):
    inputs = bert_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1)
    return predicted_class.item()

def chatbot_response(user_input):
    # You can use text_classification for certain queries
    if user_input.lower() == "classify":
        input_for_classification = input("Provide text for classification: ")
        predicted_class = text_classification(input_for_classification)
        return "Predicted Class: " + str(predicted_class)
    else:
        # Handle other responses or return a default response
        return "I'm not sure how to respond to that."

if __name__ == "__main__":
    print("Personal Assistant Bot: Hello! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Personal Assistant Bot: Goodbye!")
            break

        response = chatbot_response(user_input)
        print("Personal Assistant Bot:", response)
