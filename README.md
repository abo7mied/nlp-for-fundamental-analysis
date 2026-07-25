Tasks for the next week:

-- FIRST OBJECTIVE: Merge ABD -- 

Day 9:

0- Replace one-hot inputs with trainable token embeddings

0.5- Add positional embeddings using nn.Embedding

1- Understand the part of ABD branch code consisting of TransformerLanguageModel and what follows (blocks 13-19) and write a report/presentation that the other person can read to understand fully what's going on

2- Merge the part of ABD branch code consisting of everything that precedes TransformerLanguageModel (blocks 1-12)

Day 10:

3- Validate that, after merging blocks 1-12, the code works well, and debug if necessary

4- Merge blocks 13-19

5- Validate that, after merging blocks 13-19, the code works well, and debug if necessary (you have to read the report/presentation made by the other person)

Note: the person that validates should be different from the one that merged.

person A takes tasks 1,3,4

person B takes tasks 2,5


-- SECOND OBJECTIVE: Streamline data processing from FNSPID (via huggingface probably) up to a list of strings (or an alternative) --

Day 11:

6- Write code to load FNSPID and extract links

7- Write code to extract text from a link

8- Write code to convert extracted text to the format we expect in the current code (a list of strings) or an alternative
