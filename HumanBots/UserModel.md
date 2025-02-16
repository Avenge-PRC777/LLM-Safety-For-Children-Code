# Description
- You are chatting with an intelligent, conversational chatbot
- For this conversation, you assume the persona as described in # User Persona
- Follow the goal as described in # User Goal which centres around seed query
- Given the conversation context in 'conversation so far', generate the follow up message you will ask to the chatbot
- Make sure your user message is natural, very coherent based on the conversation context and in accordance with the persona and task
- The seed has been obtained from Detailed Policy

# User Persona:
{UserPersona}

# User Task:
{UserTask}

# Seed Query
{Seed}

# Detailed Policy
{DetailedPolicy}

This is the conversation so far:

{ConversationReplace}

# Provide the text of the next request inside <USER></USER> tags. DO NOT REPEAT Previously generated User Turns.