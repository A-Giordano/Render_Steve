# Welcome to Steve Chatbot! 🚀🤖
## Memory
### Long Term memory:
* **Top 4 vector returned by Pinecone (each vector: "user_input bot_response")**
* **Filtered by threshold 0.4**
### Short term memory:
* **Last 2 interaction user/bot**
* **Saved on Redis**

Each chat create a new user both on Redis and Pinecone