import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def chatbot(df):
  # Create a list to store all the messages for context
    messages = []
    message=f"""1. Give 3 questions to ask a data analyst about the dataset given below:/n
     {df}
     2. Understand the dataset and create the questions
     3. Give it point wise with ',' as a delimiter. """
    print(message)
    messages.append({"role": "user", "content": message})
    
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
    chat_message = response['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": chat_message})
    questions = chat_message.split('\n')  # Split by new lines first if there are any
    questions_list = []
    for question in questions:
        if question:  # Check if the line is not empty
            point = question.split('.')[1].strip()  # Split by the dot and take the second part, then strip whitespace
            if point:  # Check if the result is not empty
                questions_list.append(point.split(',')[0])  # Split by comma and take the first part if there's any

    return questions_list