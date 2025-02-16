The Results excel has 2 main sheets to follow for results:
1. KidsAdultCombined: Captures defect rates by running conversation scrapes thru refusal prompt
2. refusal_output_with_mappings: Captures refusal rates by running conversation scrapes thru refusal prompt

The columns descriptions are:
Filename: The name of the model
ConversationScrape: The conversation scrape in <USER>user message</USER><AI>ai response</AI> format
Category: High level harm area category
Threat: Sub level harm area category
DetailedPolicy: Detailed policy lines of what the harm area comprises of
Seed: Seed query
PersonalityInventory: High level personality trait
PersonalityAdjectives: Adjectives set for the personality trait
Sentiment: Positive or negative sentiment of adjective set
DomainOfInterest: Sub level interest segment
DescriptionOfInterest: Description of interest segment
OverallCategory: High level interest segment
UserPersona: The persona created by persona creator prompt by considering personality, interest and seed query
UserTask: The task created by persona creator prompt by considering personality, interest and seed query
GPTXResult: Evaluation prompt results on the conversation scrape as input to evaluation prompt
Leak: Whether conversation is leaking or not
SumTotal: Identifier column <can be ignored>
ConvNature: Whether conversation is coming from kids user model or adult user model
Turns: Number of turns in conversation scrape
DefectTurn: The turn where defect occurs according to evaluation prompt