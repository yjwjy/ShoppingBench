# Task
Write a query to simulate a user searching for <|task|> on the e-commerce platform based on the specified requirements.

# Specified Requirements
<|requirements|>

# Output Format
Your output should be a valid JSON object in the following format:
```json
{
    "query": "..."
}
```

# Important Notes
1. Include all specified requirements, except for attributes that appear only in the title.
2. Only use the basic product category name (no more than 5 tokens) from the title.
3. Replace a few words in the requirements with more conversational synonyms.
4. Don't repeat the product title in the query.